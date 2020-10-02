#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# The following kernel shows how to run a regression using [LightGBM](https://lightgbm.readthedocs.io/en/latest/). It also runs some feature engineering to improve the score.

# In[ ]:


import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from scipy.stats import norm
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, Imputer
from sklearn.model_selection import cross_val_score, cross_val_predict
import os
print(os.listdir("../input"))


# ## Data Exploration
# The following is a brief exploratory data analysis (EDA) of the data. There are other kernels on the competition page which do this more justice. Here I just focus on a summary of the data, the missing values and looking at the target. Normally we would also look at correlations, this has been done outside of this notebook and is used later in aspects of the feature engineering.

# In[ ]:


df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")


# In[ ]:


df_train.sample(3)


# In[ ]:


df_test.sample(3)


# In[ ]:


print("Shape train: %s, test: %s" % (df_train.shape, df_test.shape))


# So we have ~1500 records with 80 features. Lets look at some basic statistics of the current numerical features.

# In[ ]:


pd.options.display.max_columns = None # Show all cols
df_train.describe()


# ### Missing Data
# From the above we can see any missing values (in the numeric fields), that will later need imputing and any significant outliers will also need to be dealt with. For example "LotFrontage" is missing on 259 rows. The max sales price is over 7 standard deviations from the mean which would suggest outliers. Lets look at the top N features with missing values.

# In[ ]:


df_na = (df_train.isnull().sum() / len(df_train)) * 100
df_na = df_na.drop(df_na[df_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :df_na})
missing_data.head(10)


# ### Target analysis
# Target distribution is skewed and can be seen in the plots below - this is generally not a good thing. So we will need to adjust it so its normally distributed. This can be achieved with a log transform or something more powerful like box cox. On the diagram below, the left pane shows the original (skewed) data. The right pane show the transformed data.

# In[ ]:


fig, ax = plt.subplots(1,2)
width, height = fig.get_size_inches()
fig.set_size_inches(width*2, height)
sns.distplot(df_train['SalePrice'], ax=ax[0], fit=norm)
sns.distplot(np.log(df_train[('SalePrice')]+1), ax=ax[1], fit= norm)


# # Data Engineering
# We have to do some work to get the data into a format that will work with LightGBM. This covers:
# * Handling categoricals
# * Handling numericals
# * Feature engineering - To generate new features
# 
# This would normally be packaged into some form of utility library as a separate step in the ML pipeline. In production setups this would typically be either Python or perhaps Spark for larger data sets.
# ## Basic data engineering
# First lets define some useful functions. Again this *should* be encapsulated in an external function library. For simplicity these are defined here.

# In[ ]:


def fill_missing(df, cols, val):
    """ Fill with the supplied val """
    for col in cols:
        df[col] = df[col].fillna(val)

def fill_missing_with_mode(df, cols):
    """ Fill with the mode """
    for col in cols:
        df[col] = df[col].fillna(df[col].mode()[0])
        
def addlogs(res, cols):
    """ Log transform feature list"""
    m = res.shape[1]
    for c in cols:
        res = res.assign(newcol=pd.Series(np.log(1.01+res[c])).values)   
        res.columns.values[m] = c + '_log'
        m += 1
    return res


# Some basic calculated cols

# In[ ]:


df_train['TotalSF'] = df_train['TotalBsmtSF'] + df_train['1stFlrSF'] + df_train['2ndFlrSF']


# Add Log transform columns for simple integer features.

# In[ ]:


loglist = ['LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF',
                 'TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea',
                 'BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr',
                 'TotRmsAbvGrd','Fireplaces','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF',
                 'EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','YearRemodAdd','TotalSF']

df_train = addlogs(df_train, loglist)


# For sale price we have effectively a real valued number, so we need to use [log1p](https://docs.scipy.org/doc/numpy/reference/generated/numpy.log1p.html) to ensure the log transform is accurate. This is particularly important when the numbers are small but is just good practice for real numbers.

# In[ ]:


df_train["SalePrice"] = np.log1p(df_train["SalePrice"])


# Now impute the missing values with something sensible

# In[ ]:


fill_missing(df_train, ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu", 
                        "GarageType", "GarageFinish", "GarageQual", "GarageCond",
                       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                       "MasVnrType", "MSSubClass"], "None")
fill_missing(df_train, ["GarageYrBlt", "GarageArea", "GarageCars",
                       'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath',
                       "MasVnrArea"], 0)
fill_missing_with_mode(df_train, ["MSZoning", "KitchenQual", "Exterior1st", "Exterior2nd", "SaleType"])
fill_missing(df_train, ["Functional"],"Typ")
# Utils is pointless as there is only one row with a value
df_train.drop(['Utilities'], axis=1, inplace=True)
# For lot frontage we take the median of the neighbourhood. In general this would be a good approximation as most 
# house co located are similar in size 
df_train["LotFrontage"] = df_train.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))


# Remove outliers. These can seriously mess up a model so its best to either cap them, or drop them. Here we drop them.

# In[ ]:


df_train.drop(df_train[(df_train['OverallQual']<5) & (df_train['SalePrice']>200000)].index, inplace=True)
df_train.drop(df_train[(df_train['GrLivArea']>4000) & (df_train['SalePrice']<300000)].index, inplace=True)
df_train.reset_index(drop=True, inplace=True)


# And finally there are some fields that are categorical and we should not treat them as numbers. So we have to convert non-numeric -> string where approriate 

# In[ ]:


df_train['MSSubClass'] = df_train['MSSubClass'].apply(str)
df_train['YrSold'] = df_train['YrSold'].astype(str)
df_train['MoSold'] = df_train['MoSold'].astype(str)


# ## Handle categoricals
# First some util functions to dummy encode the categoricals. LightGBM can handle these natively but for now we do it manually as this could then easily be applied a pre process step for other algorithms.

# In[ ]:


def fix_missing_cols(in_train, in_test):
    missing_cols = set(in_train.columns) - set(in_test.columns)
    # Add a missing column in test set with default value equal to 0
    for c in missing_cols:
        in_test[c] = 0
    # Ensure the order of column in the test set is in the same order than in train set
    in_test = in_test[in_train.columns]
    return in_test

def dummy_encode(in_df_train, in_df_test):
    df_train = in_df_train
    df_test = in_df_test
    categorical_feats = [
        f for f in df_train.columns if df_train[f].dtype == 'object'
    ]
    print(categorical_feats)
    for f_ in categorical_feats:
        prefix = f_
        df_train = pd.concat([df_train, pd.get_dummies(df_train[f_], prefix=prefix)], axis=1).drop(f_, axis=1)
        df_test = pd.concat([df_test, pd.get_dummies(df_test[f_], prefix=prefix)], axis=1).drop(f_, axis=1)
        df_test = fix_missing_cols(df_train, df_test)
    return df_train, df_test


# In[ ]:


df_train, df_test = dummy_encode(df_train, df_test)
print("Shape train: %s, test: %s" % (df_train.shape, df_test.shape))


# ## Additional Feature Engineering
# Additional daa engineering often involves some complex computations. For small data sets like this its not a problem. For larger data sets you need to verify the performance vs the change in performance / accuracy after adding the features. This should be in terms of both accuracy and time to train. If there is a real benefit, then these can be applied but could be perhaps be done as a pre procesing step (eg via a Spark Job). This depends on the feature / data but the point is to ensure you test with and without the new feature.
# 
# ### Interaction Terms
# First generate some interaction terms based on the highest correlated features (these were pre-computed). 
# 
# See https://en.wikipedia.org/wiki/Interaction_(statistics)
# 
# Analysis of the features selected here showed they were more correlated (either +ve or -ve) with the sales price. This can be seen by looking using the [dataframe.corr](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.corr.html) method, then selecting the top N features. All the ones here had a positive correlation over 0.6.

# In[ ]:


def load_poly_features(df_train, df_test, cols):
    """
    USeful function to generate poly terms
    :param df_train: The training data frame
    :param df_test: The test data frame
    :return: df_poly_features, df_poly_features_test - The training polynomial features + the test
    """
    print('Loading polynomial features..')
    # Make a new dataframe for polynomial features
    poly_features = df_train[cols]
    poly_features_test = df_test[cols]

    # imputer for handling missing values
    imputer = Imputer(strategy='median')

    # Need to impute missing values
    poly_features = imputer.fit_transform(poly_features)
    poly_features_test = imputer.transform(poly_features_test)

    # Create the polynomial object with specified degree
    poly_transformer = PolynomialFeatures(degree=3)
    # Train the polynomial features
    poly_transformer.fit(poly_features)

    # Transform the features
    poly_features = poly_transformer.transform(poly_features)
    poly_features_test = poly_transformer.transform(poly_features_test)
    print('Polynomial Features shape: %s' % str(poly_features.shape))

    df_poly_features = pd.DataFrame(poly_features,
                                    columns=poly_transformer.get_feature_names(cols))
    df_poly_features_test = pd.DataFrame(poly_features_test,
                                         columns=poly_transformer.get_feature_names(cols))
    df_poly_features['Id'] = df_train['Id']
    df_poly_features_test['Id'] = df_test['Id']
    print('Loaded polynomial features')
    return df_poly_features, df_poly_features_test


# In[ ]:


correlated_cols = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF']
df_train_poly, df_test_poly =  load_poly_features(df_train, df_test, cols=correlated_cols)
print("Shape train: %s, test: %s" % (df_train_poly.shape, df_test_poly.shape))


# In[ ]:


df_train = df_train.merge(right=df_train_poly.reset_index(), how='left', on='Id')
df_test = df_test.merge(right=df_test_poly.reset_index(), how='left', on='Id')


# In[ ]:


print("Shape train: %s, test: %s" % (df_train.shape, df_test.shape))


# So in the end our features have gone from the original 80 to 446. The number of records has slightly reduced as we dropped the outliers.[](http://)

# # Light GBM
# Now lets run our regression!
# 
# First lets split up the data into our training data (X_train), our testing data frame (X_test) and our target variables that we want to predict for both training and testing (y_train and y_test respectively).

# In[ ]:


y = df_train["SalePrice"]
y.sample(3)


# In[ ]:


df_train.drop(["SalePrice"], axis=1, inplace=True)
# The fix missing cols above will have added the target column to the test data frame, so this is a workaround to remove it
df_test.drop(["SalePrice"], axis=1, inplace=True) 


# In[ ]:


print("Shape train: %s, test: %s" % (df_train.shape, df_test.shape))


# Split the data set into training and testing data with a fixed random value.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split( df_train, y, test_size=0.2, random_state=42)


# The hyper parameter settings are below. The settings below are best on a few iterations of training with some guided attempts driven by the documentation on the LightGBM website. They are far from the optimum..
# 
# In a real application we would adjust these to see the impact on loss - over numerous iterations. This would either be manual or more likely use a tool like [Optunity](https://optunity.readthedocs.io/en/latest/) or Hyperopt, to run automated hyper parameter tuning.

# In[ ]:


hyper_params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': ['l2', 'auc'],
    'learning_rate': 0.005,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.7,
    'bagging_freq': 10,
    'verbose': 0,
    "max_depth": 8,
    "num_leaves": 128,  
    "max_bin": 512,
    "num_iterations": 100000,
    "n_estimators": 1000
}


# In[ ]:


gbm = lgb.LGBMRegressor(**hyper_params)


# In[ ]:


gbm.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='l1',
        early_stopping_rounds=1000)


# In[ ]:


y_pred = gbm.predict(X_train, num_iteration=gbm.best_iteration_)


# In[ ]:


# Basic RMSE
print('The rmse of prediction is:', round(mean_squared_log_error(y_pred, y_train) ** 0.5, 5))


# ## Results
# First lets create the prediction CSV required for model submission. This could be submitted via the Kaggle command line but for simplicity, was uploaded via the web UI.

# In[ ]:


test_pred = np.expm1(gbm.predict(df_test, num_iteration=gbm.best_iteration_))


# In[ ]:


df_test["SalePrice"] = test_pred
df_test.to_csv("results.csv", columns=["Id", "SalePrice"], index=False)


# The model developed above is a first draft to highlight the code required to implement LightGBM on a regression problem. Its current performance can be seen on the [leaderboard](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/leaderboard). As of writing this kernel the score was 0.13302, which gets to around the top 40% of the leaderboard (position 1917).
# 
# 
# ## Conclusion
# LightGBM provides a robust implementation of gradient boosting for decision trees. The training times are comparably short and out of the box and with minimal tuning you can achieve excellent model accuracy. There is an good write up about LightGBM on [Analytics Vidhya](https://www.analyticsvidhya.com/blog/2017/06/which-algorithm-takes-the-crown-light-gbm-vs-xgboost/).
# 
# To take this model to the next level of performance, other areas to try would be:
# * Different transformations to normalise the skewed data (eg Box Cox transform)
# * Stacking or ensembling of multiple models together. Other possible models could be the [SKlearn XGBOOST](https://github.com/dmlc/xgboost) or something completely different like an ANN.
# * Hyperparameter tuning should be applied. Currently only a few parameters have been tried. The parameter space is large for LightGBM with numerous possibilities. Something like Optunity would be able to automate finding a much better set, albeit taking some time to run.
# 
# With a few tweaks and some additional time on feature enrichment, then significant advances in accuracy could be achieved.
