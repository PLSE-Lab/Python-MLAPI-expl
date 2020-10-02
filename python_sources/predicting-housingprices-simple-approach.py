#!/usr/bin/env python
# coding: utf-8

# ## Introduction 
# ![featured image](https://storage.googleapis.com/idx-acnt-gs.ihouseprd.com/AR658975/file_manager/increase.jpg)
# This Kernel is meant to solve the Housing Prices Challenge in a simple way, very much understandable for beginners at the same time being able to produce a great LB score.  
# 
# Predicting Housing Prices is one of the very first things you must be able to do as at the start of your Data Science career. Our goal here is to build a machine learning model that can predict housing prices with good accuracy provided a great set of features which we'll extract from the training data for this competition. The performance of our model is evaluated based on a metric called Root-Mean-Squared-Error(RMSE). If you haven't yet read the competition desciption I strongly suggest you to go out and read it first up [here](http://https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview/description). One of primary objectives of this kernel is to make things look simple enough code-wise for beginners so that they can understand, replicate and reproduce better models in the future. Without further ado, let's jump right in. 
# 
# **And by the way if you like this Kernel please give it an upvote.**

# ## Import all the necessary packages

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option('display.width', 1000)

# Plotting Tools
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns

# Import Sci-Kit Learn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, StratifiedKFold, learning_curve, KFold

# Ensemble Models
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Package for stacking models
from vecstack import stacking

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))

from IPython.display import display, HTML
display(HTML("""
<style>
.output_png {
    display: table-cell;
    text-align: center;
    vertical-align: middle;
}
</style>
"""))


# In[ ]:


def show_all(df):
    #This fuction lets us view the full dataframe
    with pd.option_context('display.max_rows', 100, 'display.max_columns', 100):
        display(df)


# ## Importing Data

# In[ ]:


# Bring training data into the environment
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')

# Bring test data into the environment
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')

show_all(train.head())


# Oh looks like there are a ton of columns(features).

# ## Data Exploration, Engineering and Cleaning 

# In[ ]:


train.info()


# We can see that there are a lot of missing values in a number of columns. I'll come on to fixing that soon. The target variable for us is the SalePrice. 

# In[ ]:


train.describe()


# #### Lets make a plot to intuitively understand what's missing in our data and where it is missing.

# In[ ]:


# Plot missing values 
def plot_missing(df):
    # Find columns having missing values and count
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    #missing.sort_values(inplace=True)
    
    # Plot missing values by count 
    missing.plot.bar(figsize=(12,8))
    plt.xlabel('Columns with missing values')
    plt.ylabel('Count')
    
    # search for missing data
    import missingno as msno
    msno.matrix(df=df, figsize=(16,8), color=(0,0.2,1))
    
plot_missing(train)


# #### Fixing the missing data

# In[ ]:


# # IMPUTING MISSING VALUES
def fill_missing_values(df):
    ''' This function imputes missing values with median for numeric columns 
        and most frequent value for categorical columns'''
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    for column in list(missing.index):
        if df[column].dtype == 'object':
            df[column].fillna(df[column].value_counts().index[0], inplace=True)
        elif df[column].dtype == 'int64' or 'float64' or 'int16' or 'float16':
            df[column].fillna(df[column].median(), inplace=True)


# In[ ]:


fill_missing_values(train)
train.isnull().sum().max()


# #### Now the test set also needs attention

# In[ ]:


# Using the function written above to visualize missing values
plot_missing(test)


# #### Fixing missing data in test set

# In[ ]:


fill_missing_values(test)
test.isnull().sum().max()


# ### Imputing Categorical variables
# As we have quite a lot of columns with textual and categorical data we have to impute them with numerics because ML models can only work with numbers.

# In[ ]:


def impute_cats(df):
    '''This function converts categorical and non-numeric 
       columns into numeric columns to feed into a ML algorithm'''
    # Find the columns of object type along with their column index
    object_cols = list(df.select_dtypes(exclude=[np.number]).columns)
    object_cols_ind = []
    for col in object_cols:
        object_cols_ind.append(df.columns.get_loc(col))

    # Encode the categorical columns with numbers    
    label_enc = LabelEncoder()
    for i in object_cols_ind:
        df.iloc[:,i] = label_enc.fit_transform(df.iloc[:,i])


# In[ ]:


# Impute the missing values
impute_cats(train)
impute_cats(test)
print("Train Dtype counts: \n{}".format(train.dtypes.value_counts()))
print("Test Dtype counts: \n{}".format(test.dtypes.value_counts()))


# ### Understanding our data

# In[ ]:


corr_mat = train[["SalePrice","MSSubClass","MSZoning","LotFrontage","LotArea", "BldgType",
                       "OverallQual", "OverallCond","YearBuilt", "BedroomAbvGr", "PoolArea", "GarageArea",
                       "SaleType", "MoSold"]].corr()
# corr_mat = train.corr()
f, ax = plt.subplots(figsize=(16, 8))
sns.heatmap(corr_mat, vmax=1 , square=True)


# We can see that the features OverallQual, GarageArea and YearBuilt are closely correlated with the sale price. That means these features play an important role in determining the SalePrice of a house. 
# Similarly a lot of inferences can be made just by looking at the heatmap but we're not going to go through each one of them. If you don't know what a heatmap is and wish to learn, I strongly recommend checking out [this video](https://www.youtube.com/watch?v=oMtDyOn2TCc).

# #### Visulaizing the relationship between SalePrice and YearBuilt

# In[ ]:


f, ax = plt.subplots(figsize=(16, 8))
sns.lineplot(x='YearBuilt', y='SalePrice', data=train)


# You might have noticed a significant increase in SalePrices just after the start of the 21st century, which is pretty interesting. What's even more surprising is the late 1800s saw phenomenal increase in SalePrices but dropping way below even before the end of that century.

# #### Visualizing the relationship between Sale prices and Overall Quality 

# In[ ]:


f, ax = plt.subplots(figsize=(12, 8))
sns.lineplot(x='OverallQual', y='SalePrice', color='green',data=train)


# We can see that the SalePrices increase rapidly with houses with better overall quality which is pretty reasonable.

# #### Visulizing the Distribution of SalePrice

# In[ ]:


f, ax = plt.subplots(figsize=(12, 6))
sns.distplot(train['SalePrice'])


# It's clear that most of the houses, about 80% are sold for a price in the region on 100,000 and 200,000 dollars. Looks like there are a lot of houses sold in the mid 100,000s which makes up most of the data.

# #### Visualizing the relationship between SalePrice and SaleType

# In[ ]:


sns.catplot(x='SaleType', y='SalePrice', data=train, kind='bar', palette='muted')


# It is clear that the selling prices for Sale types 2 and 6 are significantly higher than the others.

# ### Preparing the Data to do Machine Learning

# In[ ]:


X = train.drop('SalePrice', axis=1)
y = np.ravel(np.array(train[['SalePrice']]))
print(y.shape)


# In[ ]:


# Use train_test_split from sci-kit learn to segment our data into train and a local testset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# ## Define Evaluation Metric

# Submissions are evaluated on Root-Mean-Squared-Error (RMSE) between the logarithm of the predicted value and the logarithm of the observed sales price. (Taking logs means that errors in predicting expensive houses and cheap houses will affect the result equally.)
# 
# ![RMSE formula](https://gisgeography.com/wp-content/uploads/2014/07/rmse-formula1-300x96.png)
#  
# We will write a function named rmse to perform this task.

# In[ ]:


def rmse(y, y_pred):
    return np.sqrt(mean_squared_error(np.log(y), np.log(y_pred)))


# ## Modelling

# We'll start by building standalone models, validating their performance and picking the right ones. Later we will stack all our models into an ensemble for better accuracy.
# 
# I just played with a number of models and ended up picking the following models which gave me best results personally. If you find a better way please let me know :)
# 
# I tuned the hyperparameters by manually experimenting a lot based on previous experiences, saving you a bunch of time hopefully. Anyways if you find a set of parameters that can work even better let me know in the comments down below. 

# ### Random Forest

# In[ ]:


# Initialize the model
random_forest = RandomForestRegressor(n_estimators=1200,
                                      max_depth=15,
                                      min_samples_split=5,
                                      min_samples_leaf=5,
                                      max_features=None,
                                      random_state=42,
                                      oob_score=True
                                     )

# Perform cross-validation to see how well our model does 
kf = KFold(n_splits=5)
y_pred = cross_val_score(random_forest, X, y, cv=kf, n_jobs=-1)
y_pred.mean()


# In[ ]:


# Fit the model to our data
random_forest.fit(X, y)


# In[ ]:


# Make predictions on test data
rf_pred = random_forest.predict(test)


# ### XG Boost

# Note: I found that our standalone XGBoost model in itself gives a good score. I suggest you to check it out.

# In[ ]:


# Initialize our model
xg_boost = XGBRegressor( learning_rate=0.01,
                         n_estimators=6000,
                         max_depth=4, min_child_weight=1,
                         gamma=0.6, subsample=0.7,
                         colsample_bytree=0.2,
                         objective='reg:linear', nthread=-1,
                         scale_pos_weight=1, seed=27,
                         reg_alpha=0.00006
                       )

# Perform cross-validation to see how well our model does 
kf = KFold(n_splits=5)
y_pred = cross_val_score(xg_boost, X, y, cv=kf, n_jobs=-1)
y_pred.mean()


# In[ ]:


# Fit our model to the training data
xg_boost.fit(X,y)


# In[ ]:


# Make predictions on the test data
xgb_pred = xg_boost.predict(test)


# ### Gradient Boost Regressor(GBM)

# In[ ]:


# Initialize our model
g_boost = GradientBoostingRegressor( n_estimators=6000, learning_rate=0.01,
                                     max_depth=5, max_features='sqrt',
                                     min_samples_leaf=15, min_samples_split=10,
                                     loss='ls', random_state =42
                                   )

# Perform cross-validation to see how well our model does 
kf = KFold(n_splits=5)
y_pred = cross_val_score(g_boost, X, y, cv=kf, n_jobs=-1)
y_pred.mean()


# In[ ]:


# Fit our model to the training data
g_boost.fit(X,y)


# In[ ]:


# Make predictions on test data
gbm_pred = g_boost.predict(test)


#  ### LightGBM

# In[ ]:


# Initialize our model
lightgbm = LGBMRegressor(objective='regression', 
                                       num_leaves=6,
                                       learning_rate=0.01, 
                                       n_estimators=6400,
                                       verbose=-1,
                                       bagging_fraction=0.80,
                                       bagging_freq=4, 
                                       bagging_seed=6,
                                       feature_fraction=0.2,
                                       feature_fraction_seed=7,
                                    )

# Perform cross-validation to see how well our model does
kf = KFold(n_splits=5)
y_pred = cross_val_score(lightgbm, X, y, cv=kf)
print(y_pred.mean())


# In[ ]:


# Fit our model to the training data
lightgbm.fit(X,y)


# In[ ]:


# Make predictions on test data
lgb_pred = lightgbm.predict(test)


# ## Model Stacking

# Stacking (also called meta ensembling) is a model ensembling technique used to combine information from multiple predictive models to generate a new model which usually performs better.
# In this project we use python package called **vecstack** that helps us stack our models which we have imported earlier. It's actually very easy to use, you can have a look at the [documentation](https://github.com/vecxoz/vecstack) for more information.  

# In[ ]:


# List of the models to be stacked
models = [g_boost, xg_boost, lightgbm, random_forest]


# In[ ]:


# Perform Stacking
S_train, S_test = stacking(models,
                           X_train, y_train, X_test,
                           regression=True,
                           mode='oof_pred_bag',
                           metric=rmse,
                           n_folds=5,
                           random_state=25,
                           verbose=2
                          )


# In[ ]:


# Initialize 2nd level model
xgb_lev2 = XGBRegressor(learning_rate=0.1, 
                        n_estimators=500,
                        max_depth=3,
                        n_jobs=-1,
                        random_state=17
                       )

# Fit the 2nd level model on the output of level 1
xgb_lev2.fit(S_train, y_train)


# In[ ]:


# Make predictions on the localized test set
stacked_pred = xgb_lev2.predict(S_test)
print("RMSE of Stacked Model: {}".format(rmse(y_test,stacked_pred)))


# ## Prediction

# Now it is finally time to make predictions on the real world test data. The approach here might look strange to you. You can visit this [link](https://github.com/vecxoz/vecstack/issues/4) to understand how it is done. 

# #### Prediction on Stacked Layer 1

# In[ ]:


y1_pred_L1 = models[0].predict(test)
y2_pred_L1 = models[1].predict(test)
y3_pred_L1 = models[2].predict(test)
y4_pred_L1 = models[3].predict(test)
S_test_L1 = np.c_[y1_pred_L1, y2_pred_L1, y3_pred_L1, y4_pred_L1]


# #### Final Prediction

# In[ ]:


test_stacked_pred = xgb_lev2.predict(S_test_L1)


# In[ ]:


# Save the predictions in form of a dataframe
submission = pd.DataFrame()

submission['Id'] = np.array(test.index)
submission['SalePrice'] = test_stacked_pred


# #### Blending with top kernels

# In[ ]:


top_public = pd.read_csv('../input/top-submissions/submission (9).csv')


# In[ ]:


final_blend = (0.8*top_public.SalePrice.values + 0.2*test_stacked_pred)

blended_submission = pd.DataFrame()

blended_submission['Id'] = np.array(test.index)
blended_submission['SalePrice'] = final_blend


# ### Create Submission Files

# In[ ]:


submission.to_csv('submission.csv', index=False)
blended_submission.to_csv('blended_submission.csv', index=False) # Best LB Score


# i hope that was helpful. Thanks for being here, if you like this kernel **please upvote**. If you have any suggestions drop them down below. 
# 
# Until next time, Happy Kaggling! :)
