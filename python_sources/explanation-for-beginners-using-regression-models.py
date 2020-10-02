#!/usr/bin/env python
# coding: utf-8

# Our objective is to obtain a relation between the Sale Price of the houses in Ames, Iowa and the variables given in our dataset such that we may be able to predict the house prices of any other house based on the dataset.
# 
# **Our Code consists of these parts:**
# <br><br>
# **1. Importing packages and the datasets** <br>
# **2. Data visualisation**<br>
# **3. Data Analysis**<br>
# **4. Handling missing values**<br>
# **5. Feature Engineering**<br>
# **6. Pre-processing the data**<br>
# **7. Creating models**<br>
# **8. Stacking**<br>
# **9. Scores based on RMSE values**<br>
# **10. Blending our models**<br>
# **11. Ensembling with outputs of better performing models**

# ## 1. Importing packages and the datasets

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# We start off by importing the basic data preprocessing, analysis and visualisation packages.<br><br>
# Next we read the training & test set files. 

# In[ ]:


train=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
test=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")


# The Id column is unnecessary as it is just the serial number of every data entry and will have no relation with the sale price whatsoever.<br><br>So we drop it from our dataset.

# In[ ]:


train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)
train.shape, test.shape


# In[ ]:


train.describe().T #transposes the actual describe


# ## 2. Data visualisation
# ## 3. Data Analysis 
# 
# These will be performed simultaneously as plots will give us insights to the importance and distribution of features with respect to the __SalePrice__ target variable

# In[ ]:


sns.distplot(train['SalePrice']);


# This function gave us the distribution of our **SalePrice** values.<br><br>As you can see; the histogram is skewed to the right.<br><br>
# Therefore, we need to do something to normalise the data distribution because most of the machine learning models work best on normally distributed data.

# In[ ]:


#correlation matrix

corrmat = train.corr()
f, ax = plt.subplots(figsize=(15, 10))
sns.heatmap(corrmat, vmax=.8, square=True);


# We generated a heatmap to tell us about the correlation between different variables.

# In[ ]:


#saleprice correlation matrix
#k = 10 #number of variables for heatmap

cols = corrmat.nlargest(10, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# The above graph tells us about the variables with the **10 highest values of correlation with SalePrice values** 

# In[ ]:


#Graph for SalePrice v/s OverallQual

var = 'OverallQual'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));


# In[ ]:


#Graph for SalePrice v/s GrLivArea

var = 'GrLivArea'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));


# In[ ]:


#Deleting outliers
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

#Graph for SalePrice v/s GrLivArea after deleting outliers

var = 'GrLivArea'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));


# In[ ]:


#Graph for SalePrice v/s GarageCars

var = 'GarageCars'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));


# In[ ]:


#Graph for SalePrice v/s GarageArea

var = 'GarageArea'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));


# In[ ]:


#Graph for SalePrice v/s TotalBsmtSF

var = 'TotalBsmtSF'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));


# In[ ]:


#Graph for SalePrice v/s 1stFlrSF

var = '1stFlrSF'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));


# Know more about boxcox transform here : http://blog.minitab.com/blog/applying-statistics-in-quality-projects/how-could-you-benefit-from-a-box-cox-transformation

# In[ ]:


#Graph for SalePrice v/s FullBath

var = 'FullBath'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));


# In[ ]:


#Graph for SalePrice v/s TotRmsAbvGrd

var = 'TotRmsAbvGrd'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));


# We generated scatter plots of the Sale Price v/s the 9 variables with highest correlation values with it.

# In[ ]:


train["SalePrice"] = np.log1p(train["SalePrice"])
y = train['SalePrice'].reset_index(drop=True)

sns.distplot(train['SalePrice']);


# In many cases, taking the log greatly reduces the variation of a variable making estimates less prone to outlier influence.<br><br>
# That justifies a logarithmic transformation. Taking the log of saleprice as new SalePrice values removes to a great extent the skewness of the SalePrice distribution. Now we have a somewhat normally distributed histogram.

# In[ ]:


train.shape, test.shape


# Combining the training and testing sets allows us to clean and pre-process the data together and hence, efficiently.

# In[ ]:


combine = pd.concat([train, test], sort = False).reset_index(drop=True)
combine.drop(['SalePrice'], axis=1, inplace=True)
print("Size of combined data set is : {}".format(combine.shape))


# In[ ]:


combine.describe()


# ## 4. Handling missing values
# 
# Having observed the datasets we know that there are a number of missing data entries for every house in them.<br><br>Handling them may be important because they might cause problems in our model. [Read about it here](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3668100/)<br><br>We defined the __miss_perc__ function to tell us about which variables have missing data and quantises it.

# In[ ]:


def miss_perc(df):
  df_null_data = (df.isnull().sum() / len(combine)) * 100
  df_null_data = df_null_data.drop(df_null_data[df_null_data == 0].index).sort_values(ascending=False)[:30]
  return pd.DataFrame({'Missing Percentage' :df_null_data})

miss_perc(combine)


# These features are categorical in nature and the model may mistakenly consider them as numerical features. Thence, we establish them as type __String__

# In[ ]:


combine['MSSubClass'] = combine['MSSubClass'].apply(str)
combine['YrSold'] = combine['YrSold'].astype(str)
combine['MoSold'] = combine['MoSold'].astype(str)


# Now we will fill our missing data entries with values most suitable for their type.<br><br>
# 
# First, we fill the numerical features with the value 0, because given the data description and upon some thinking, it is likely that these values are missing because the feature they are associated to is not a feature of the house.<br><br>
# Second, these are the categorical features, which need to have an object type data type such as string and similarly from observation, we place in the missing entries the value 'None'<br><br>
# Third, there are very few empty values in these remaining columns, so we will us the mode to fill them out.<br><br>
# The __LotFrontage__ was filled with the median of the values of LotFrontage of houses of every __Neighborhood__.<br><br>
# The __MSZoning__ was filled with the median of the values of MSZoning of their respective __MSSubClass__.

# In[ ]:


for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 
            'BsmtHalfBath','GarageYrBlt', 'GarageArea','GarageCars','MasVnrArea'):
    combine[col] = combine[col].fillna(0)

for col in ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
            'Fence','PoolQC','MiscFeature','Alley','FireplaceQu','Fence','GarageType',
            'GarageFinish', 'GarageQual', 'GarageCond']:
    combine[col] = combine[col].fillna('None')

for col in ['Utilities','Exterior1st','Exterior2nd','SaleType','Functional','Electrical',
            'KitchenQual', 'GarageFinish', 'GarageQual', 'GarageCond','MasVnrType']:
    combine[col] = combine[col].fillna(combine[col].mode()[0])

combine['LotFrontage'] = combine.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

combine['MSZoning'] = combine.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))

miss_perc(combine)


# Now when we called __miss_perc__, we saw that all the missing values had been handled.

# The next code cells groups together the categorical and numerical features. 

# In[ ]:


categorical_features = combine.dtypes[combine.dtypes == "object"].index

combine.update(combine[categorical_features].fillna('None'))

categorical_features


# In[ ]:


numerical_features = combine.dtypes[combine.dtypes != "object"].index

combine.update(combine[numerical_features].fillna(0))

numerical_features


# ## 5. Feature Engineering
# 
# Importing packages to help with transformation of the features as more of them may be skewed.

# In[ ]:


from scipy import stats
from scipy.stats import norm, skew, boxcox_normmax # for statistics
from scipy.special import boxcox1p


# Grouping the features having skewness together

# In[ ]:


skewed_features = combine[numerical_features].apply(lambda x: skew(x)).sort_values(ascending=False)
skewed_features


# Box-Cox transformation is used here, to make the features with high skewness normally distributed. <br><br>Box Cox is useful for highly skewed non-positive data. See [here](https://stats.stackexchange.com/questions/339589/box-cox-log-or-arcsine-transformation) or [here](https://stats.stackexchange.com/a/1452) for more explanation.

# In[ ]:


high_skew_feat = skewed_features[abs(skewed_features) > 0.5]
skewed_features = high_skew_feat.index

for feature in skewed_features:
  combine[feature] = boxcox1p(combine[feature], boxcox_normmax(combine[feature] + 1))


# __Utilities, Street and PoolQC__ are observed to be uninfluential to the SalePrice.<br><br>
# We have created new features here, by the name of __TotalSF__ etc. which are self explanatory.

# In[ ]:


combine = combine.drop(['Utilities', 'Street', 'PoolQC',], axis=1)

combine['TotalSF'] = combine['TotalBsmtSF'] + combine['1stFlrSF'] + combine['2ndFlrSF']

combine['YrBltAndRemod'] = combine['YearBuilt']+ combine['YearRemodAdd']

combine['Total_sqr_footage'] = (combine['BsmtFinSF1'] + combine['BsmtFinSF2'] + combine['1stFlrSF'] + combine['2ndFlrSF'])

combine['Total_Bathrooms'] = (combine['FullBath'] + (0.5 * combine['HalfBath']) + combine['BsmtFullBath'] + (0.5 * combine['BsmtHalfBath']))

combine['Total_porch_sf'] = (combine['OpenPorchSF'] + combine['3SsnPorch'] + combine['EnclosedPorch'] + combine['ScreenPorch'] +
                             combine['WoodDeckSF'])

combine['haspool'] = combine['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

combine['has2ndfloor'] = combine['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

combine['hasgarage'] = combine['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

combine['hasbsmt'] = combine['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

combine['hasfireplace'] = combine['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)


# ## 6. Pre-processing the data
# 
# Encoding the categorical features. These features have data entries which are text format, not understandable by the model.
# This cell converts the different text categories in to numeric categories.<br><br>
# The features below have more than two types of categories.They are not merely columns to store the data for presence or absence of a feature of a house.

# In[ ]:


from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(combine[c].values)) 
    combine[c] = lbl.transform(list(combine[c].values))


# Categorical features are converted into dummy/indicator variables by means of this function. At this point, get_dummies and LabelEncoder seem similar. Please check out this [link](https://stats.stackexchange.com/questions/369428/deciding-between-get-dummies-and-labelencoder-for-categorical-variables-in-a-lin) to get rid of confusion.

# In[ ]:


combine = pd.get_dummies(combine)
print(combine.shape)


# Used for location using index of the data in the feature. <br><br>
# X is the feature input file and X_sub is a similar file which will be used later for the submision file.

# In[ ]:


X = combine.iloc[:len(y), :]
X_sub = combine.iloc[len(y):, :]


# In[ ]:


X.shape, y.shape, X_sub.shape


# Removing the features that overfit.

# In[ ]:


overfit = []
for i in X.columns:
    counts = X[i].value_counts()
    zeros = counts.iloc[0]
    if zeros / len(X) * 100 > 99.94:
        overfit.append(i)

overfit = list(overfit)

X = X.drop(overfit, axis=1).copy()
X_sub = X_sub.drop(overfit, axis=1).copy()
overfit


# Looks like there aren't any

# ## 7. Creating models
# 
# Importing the necessary packages for defining the models we wish to use. They are all regression models as is required for this problem

# In[ ]:


from datetime import datetime

from sklearn.linear_model import ElasticNetCV, Lasso, ElasticNet, LassoCV, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.svm import SVR

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error

from mlxtend.regressor import StackingCVRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

import sklearn.linear_model as linear_model


# The Root Mean Squared Error is defined here to find the accuracy of our predictions. <br><br> KFold is a method of cross-validating the model ability on new data. See [here](https://machinelearningmastery.com/k-fold-cross-validation/)

# In[ ]:


kfolds = KFold(n_splits=10, shuffle=True, random_state=42)

def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

def cv_rmse(model, X=X):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kfolds))
    return (rmse)


# Alphas arrays; i.e., the list of parameters that we use here within ridge, lasso and Elastic net models as a **regularisation penalty parameter.**  

# In[ ]:


alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]
alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]
e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]


# These models are all different methods of regression. Please read the documnetation of each to understand the details and the choice of parameters. <br><br>
# The __RobustScaler()__ is used because we have not handled the outliers very well.<br><br>
# We do not want the models to succumb to inaccuracy in prediction due to non-robustness of our model arising due to the presence of these outliers.<br><br>
# **make_pipeline** can be understood as an entity that allows for sequential assembly of multiple transforms on a dataset.<br<br>In our case we combine the robust scaler and estimator and cross validator module.

# In[ ]:


ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=alphas_alt, cv=kfolds))


# In[ ]:


lasso = make_pipeline(RobustScaler(), LassoCV(max_iter=1e7, alphas=alphas2, random_state=42, cv=kfolds))


# In[ ]:


ENet = make_pipeline(RobustScaler(), ElasticNetCV(max_iter=1e7, alphas=e_alphas, cv=kfolds, l1_ratio=e_l1ratio))


# In[ ]:


svr = make_pipeline(RobustScaler(), SVR(C= 20, epsilon= 0.008, gamma=0.0003,))


# In[ ]:


XGBoostR = XGBRegressor(learning_rate=0.01,n_estimators=3460, max_depth=3, min_child_weight=0, gamma=0, subsample=0.7, colsample_bytree=0.7, 
                       objective='reg:squarederror', nthread=-1, scale_pos_weight=1, seed=27, reg_alpha=0.00006, silent = True)


# In[ ]:


CatBoostR = CatBoostRegressor(iterations=500, learning_rate=0.05, depth=10, eval_metric='RMSE', random_seed = 42,
                        bagging_temperature = 0.2, od_type='Iter', metric_period = 50, od_wait=20)


# In[ ]:


LightGBMR = LGBMRegressor(objective='regression', num_leaves=4, learning_rate=0.01, n_estimators=5000, max_bin=200, bagging_fraction=0.75,
                                       bagging_freq=5, bagging_seed=7, feature_fraction=0.2, feature_fraction_seed=7, verbose=-1, )


# ## 8. Stacking 
# ##            and
# ## 9. Scores based on RMSE values

# In[ ]:


StackCVR_gen = StackingCVRegressor(regressors=(ridge, lasso, ENet, CatBoostR, XGBoostR, LightGBMR), 
                                meta_regressor=XGBoostR, use_features_in_secondary=True)


# In[ ]:


# Using various prediction models that we just created 

score = cv_rmse(ridge , X)
print("Ridge: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

score = cv_rmse(lasso , X)
print("LASSO: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

score = cv_rmse(ENet)
print("elastic net: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

score = cv_rmse(svr)
print("SVR: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

score = cv_rmse(LightGBMR)
print("lightgbm: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

score = cv_rmse(CatBoostR)
print("catboost: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

score = cv_rmse(XGBoostR)
print("xgboost: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )


# In[ ]:


print('START Fit')

print('stack_gen')
stack_gen_model = StackCVR_gen.fit(np.array(X), np.array(y))

print('elasticnet')
elastic_model_full_data = ENet.fit(X, y)

print('Lasso')
lasso_model_full_data = lasso.fit(X, y)

print('Ridge')
ridge_model_full_data = ridge.fit(X, y)

print('Svr')
svr_model_full_data = svr.fit(X, y)

print('catboost')
cbr_model_full_data = CatBoostR.fit(X, y)

print('xgboost')
xgb_model_full_data = XGBoostR.fit(X, y)

print('lightgbm')
lgb_model_full_data = LightGBMR.fit(X, y)


# ## 10. Blending our models

# In[ ]:


def blend_models_predict(X):
    return ((0.15 * elastic_model_full_data.predict(X)) +             (0.15 * lasso_model_full_data.predict(X)) +             (0.1 * ridge_model_full_data.predict(X)) +             (0.1 * svr_model_full_data.predict(X)) +             (0.05 * cbr_model_full_data.predict(X)) +             (0.05 * xgb_model_full_data.predict(X)) +             (0.1 * lgb_model_full_data.predict(X)) +             (0.3 * stack_gen_model.predict(np.array(X))))


# In[ ]:


print('RMSLE score on train data:')
print(rmsle(y, blend_models_predict(X)))


# In[ ]:


print('Predict submission')
submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")
submission.iloc[:,1] = np.floor(np.expm1(blend_models_predict(X_sub)))


# ## 11. Ensembling with outputs of better performing models

# In[ ]:


print('Blend with Top Kernels submissions\n')
sub_1 = pd.read_csv('https://raw.githubusercontent.com/AnujPR/Kaggle-Hybrid-House-Prices-Prediction/master/masum_rumia-detailed-regression-guide-with-house-pricing%20submission.csv')
sub_2 = pd.read_csv('https://raw.githubusercontent.com/AnujPR/Kaggle-Hybrid-House-Prices-Prediction/master/serigne_stacked-regressions-top-4-on-leaderboard_submission.csv')
sub_3 = pd.read_csv('https://raw.githubusercontent.com/AnujPR/Kaggle-Hybrid-House-Prices-Prediction/master/jesucristo1-house-prices-solution-top-1_new_submission.csv')
submission.iloc[:,1] = np.floor((0.25 * np.floor(np.expm1(blend_models_predict(X_sub)))) + 
                                (0.25 * sub_1.iloc[:,1]) + 
                                (0.25 * sub_2.iloc[:,1]) + 
                                (0.25 * sub_3.iloc[:,1]))


# In[ ]:


q1 = submission['SalePrice'].quantile(0.0042)
q2 = submission['SalePrice'].quantile(0.99)
# Quantiles helping us get some extreme values for extremely low or high values 
submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x > q1 else x*0.77)
submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x < q2 else x*1.1)
submission.to_csv("submission.csv", index=False)


# In[ ]:


submission.head()

