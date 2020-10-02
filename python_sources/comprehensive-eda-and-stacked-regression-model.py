#!/usr/bin/env python
# coding: utf-8

# This notebook begins with some exploratory data analysis and follows with feature engineering. Several simple regression models were applied and finally a stacked model built based on those simple models. It should be a great place to start if you are a beginner. Feel free to use this kernel, please upvote if you find this is helpful and share your comment. Great thanks to the following kernels:
# 
# https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python
# 
# https://www.kaggle.com/shaygu/house-prices-begginer-top-7 
# 
# https://www.kaggle.com/lavanyashukla01/how-i-made-top-0-3-on-a-kaggle-competition

# # Table of Contents
# 1. [Importing libraries](#Importing libraries)
# 2. [Importing data](#Importing data)
# 3. [Exploring predicted variable](#Exploring predicted variable)
# 4. [Exploring numerical features](#Exploring numerical features)
# 5. [Features](#Features)
# 6. [Train a model](#Train a model)
# 7. [Submission](#Submission)

# # Importing libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy.stats import skew
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from sklearn.preprocessing import StandardScaler
from scipy import stats
get_ipython().run_line_magic('matplotlib', 'inline')

# Ignore useless warnings
import warnings
warnings.filterwarnings(action="ignore")
pd.options.display.max_seq_items = 8000
pd.options.display.max_rows = 8000

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Importing data

# In[ ]:


train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')


# In[ ]:


train.head()


# # Exploring predicted variable

# In[ ]:


train['SalePrice'].describe()


# In[ ]:


print(train['SalePrice'].skew())
print(train['SalePrice'].kurt())


# In[ ]:


#histogram and normal probability plot
sns.distplot(train['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)


# The sale price was not normally distributed. We can tranform it using log function.

# In[ ]:


train['SalePrice'] = np.log(train['SalePrice'])


# In[ ]:


sns.distplot(train['SalePrice'], fit=norm);

fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)


# # Exploring numerical features

# ## Heat map of correlations

# In[ ]:


# correlation matrix
fig, ax = plt.subplots(figsize=(12,9))
sns.heatmap(train.corr(),square=True)


# From the above heat map, we can find some of the features have higher correlations with sale price. Let's find the top ten highest correlation and plot the zoomed heat map.

# In[ ]:


# zoomed heatmap - selected variables
cols = train.corr().nlargest(10, 'SalePrice').index


# In[ ]:


plt.subplots(figsize=(10,10))
sns.set(font_scale=1.25)
sns.heatmap(train[cols].corr(),square=True, annot=True)


# In[ ]:


cols


# In[ ]:


sns.pairplot(train[cols])


# Let's closely take a look at several features.

# ## Remove outliers

# In[ ]:


var = 'GrLivArea'
plt.scatter(x=train[var], y=train['SalePrice'])


# We can find that there are two outliers at the bottom right. Let's locate them.

# In[ ]:


train[train['GrLivArea'] > 4500].index


# They will be deleted later.

# In[ ]:


var = 'GarageArea'
plt.scatter(x=train[var], y=train['SalePrice'])


# In[ ]:


train[train['GarageArea'] > 1220].index


# In[ ]:


var = 'TotalBsmtSF'
plt.scatter(x=train[var], y=train['SalePrice'])


# In[ ]:


train[train['TotalBsmtSF'] > 5000].index


# In[ ]:


var = '1stFlrSF'
plt.scatter(x=train[var], y=train['SalePrice'])


# In[ ]:


train[train['TotalBsmtSF'] > 4000].index


# In[ ]:


var = 'OverallQual'
plt.subplots(figsize=(10,6))
sns.boxplot(x=train[var], y=train['SalePrice'])


# In[ ]:


var = 'YearBuilt'
fig, ax = plt.subplots(figsize=(15,6))
fig = sns.boxplot(x=train[var], y=train['SalePrice'])


# Remove outliers.

# In[ ]:


train = train.drop([523, 581, 1061, 1190, 1298])


# In[ ]:


"""train.drop(train[(train['OverallQual']<5) & (train['SalePrice']>200000)].index, inplace=True)
train.drop(train[(train['OverallQual']==8) & (train['SalePrice']>500000)].index, inplace=True)
train.drop(train[(train['OverallQual']==9) & (train['SalePrice']>500000)].index, inplace=True)
train.drop(train[(train['OverallQual']==10) & (train['SalePrice']>700000)].index, inplace=True)"""
train.reset_index(drop=True, inplace=True)


# # Features

# ## Drop ID and combine train/test

# In[ ]:


#Split train and labes
y_train = train['SalePrice'].reset_index(drop=True)
train = train.drop(['SalePrice'], axis=1)


# In[ ]:


#Delete ID
train.drop(['Id'], axis=1, inplace=True)
test.drop(['Id'], axis=1, inplace=True)


# In[ ]:


#concatenate train and test
all_features = pd.concat((train,test)).reset_index(drop=True)
all_features.shape


# ## Dealing with missing values
# ### Overview of missing values

# In[ ]:


# missing values
total = all_features.isnull().sum().sort_values(ascending=False)
percent = (all_features.isnull().sum()/all_features.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total,percent], axis=1, keys=['Total','Percent'])
missing_data.head(40)


# ### String features

# Some of the non-numeric features are stored as numbers. They should be converted to strings.

# In[ ]:


all_features['MSSubClass'] = all_features['MSSubClass'].apply(str)
all_features['YrSold'] = all_features['YrSold'].apply(str)
all_features['MoSold'] = all_features['MoSold'].apply(str)


# In[ ]:


# Some features have only a few missing value. Fill up using most common value
common_vars = ['Exterior1st','Exterior2nd','SaleType','Electrical','KitchenQual']
for var in common_vars:
    all_features[var] = all_features[var].fillna(all_features[var].mode()[0])

all_features['MSZoning'] = all_features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
# data description says NA means typical
all_features['Functional'] = all_features['Functional'].fillna('Typ')


# The rest string features can be filled with None.

# In[ ]:


col_str = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond','BsmtQual',
            'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',"PoolQC"
           ,'Alley','Fence','MiscFeature','FireplaceQu','MasVnrType','Utilities']
for col in col_str:
    all_features[col] = all_features[col].fillna('None')


# ### Numerical features

# The missing values of numerical features can be filled with 0 or median.

# In[ ]:


# Replacing missing data with 0 (Since No garage = no cars in such garage.)
col_num = ['GarageYrBlt','GarageArea','GarageCars','MasVnrArea','BsmtFinSF1','BsmtFinSF2'
           ,'BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BsmtUnfSF','TotalBsmtSF']
for col in col_num:
    all_features[col] = all_features[col].fillna(0)
    
# group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
all_features['LotFrontage'] = all_features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))


# In[ ]:


# Check if there is still missing value
all_features.isnull().sum().sort_values(ascending=False).head(5)


# Now there is no missing value in the whole dataset.

# ## Transforming skewed numerical features

# In[ ]:


# Find all numerical features
num_features = all_features.select_dtypes(exclude='object').columns


# In[ ]:


# Create box plots for all numeric features
sns.set_style("white")
f, ax = plt.subplots(figsize=(8, 12))
ax.set_xscale("log")
ax = sns.boxplot(data=all_features[num_features], orient="h", palette="Set1")
ax.xaxis.grid(False)
ax.set(ylabel="Feature names")
ax.set(xlabel="Numeric values")
ax.set(title="Numeric Distribution of Features")
sns.despine(trim=True, left=True)


# In[ ]:


# Find skewed numerical features
skewness = all_features[num_features].apply(lambda x: skew(x)).sort_values(ascending=False)
high_skewness = skewness[abs(skewness) > 0.5]

print("There are {} numerical features with Skew > 0.5 :".format(high_skewness.shape[0]))
high_skewness.sort_values(ascending=False)


# Next step, we can use Box Cox transformation on skewed data.

# In[ ]:


high_skewness.index


# In[ ]:


from scipy.special import boxcox1p
skewed_features = high_skewness.index
for feat in skewed_features:
    all_features[feat] = boxcox1p(all_features[feat], boxcox_normmax(all_features[feat] + 1))


# Let't check the skewness after transformation.

# In[ ]:


new_skewness = all_features[num_features].apply(lambda x: skew(x)).sort_values(ascending=False)
new_high_skewness = new_skewness[abs(new_skewness) > 0.5]
print("There are {} skewed numerical features after Box Cox transform".format(new_high_skewness.shape[0]))
print("Mean skewnees: {}".format(np.mean(new_high_skewness)))
new_high_skewness.sort_values(ascending=False)


# There are still a lot of skewness. They will be dealt with later.

# ## Creating new features

# In[ ]:


#  Adding total sqfootage feature 
all_features['TotalSF']=all_features['TotalBsmtSF'] + all_features['1stFlrSF'] + all_features['2ndFlrSF']
#  Adding total bathrooms feature
all_features['Total_Bathrooms'] = (all_features['FullBath'] + (0.5 * all_features['HalfBath']) +
                               all_features['BsmtFullBath'] + (0.5 * all_features['BsmtHalfBath']))
#  Adding total porch sqfootage feature
all_features['Total_porch_sf'] = (all_features['OpenPorchSF'] + all_features['3SsnPorch'] +
                              all_features['EnclosedPorch'] + all_features['ScreenPorch'] +
                              all_features['WoodDeckSF'])

all_features['haspool'] = all_features['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
all_features['hasgarage'] = all_features['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
all_features['hasbsmt'] = all_features['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
all_features['hasfireplace'] = all_features['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)


# Not normaly distributed can not be normalised and has no central tendecy
all_features = all_features.drop(['MasVnrArea', 'OpenPorchSF', 'WoodDeckSF', 'BsmtFinSF1','2ndFlrSF'], axis=1)


# ## Encode categorical features

# In[ ]:


all_features = pd.get_dummies(all_features).reset_index(drop=True)
all_features.shape


# ## Split train and test sets

# In[ ]:


X = all_features.iloc[:len(y_train), :]
X_test = all_features.iloc[len(y_train):, :]
X.shape, y_train.shape, X_test.shape


# In[ ]:


# Removes colums where the threshold of zero's is (> 99.95), means has only zero values 
overfit = []
for i in X.columns:
    counts = X[i].value_counts()
    zeros = counts.iloc[0]
    if zeros / len(X) * 100 > 99.95:
        overfit.append(i)

overfit = list(overfit)
overfit.append('MSZoning_C (all)')

X = X.drop(overfit, axis=1).copy()
X_test = X_test.drop(overfit, axis=1).copy()

print(X.shape,y_train.shape,X_test.shape)


# # Train a model

# In[ ]:


from datetime import datetime
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from mlxtend.regressor import StackingCVRegressor
from sklearn.linear_model import LinearRegression

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


# ## Set up cross validation

# In[ ]:


# setup cross validation folds
kfolds = KFold(n_splits=16, shuffle=True, random_state=42)

# define error metrics
def cv_rmse(model, X=X):
    rmse = np.sqrt(-cross_val_score(model, X, y_train, scoring="neg_mean_squared_error", cv=kfolds))
    return (rmse)

def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


# ## Defining models

# In[ ]:


# LightGBM regressor
lightgbm = LGBMRegressor(objective='regression', 
                       num_leaves=4,
                       learning_rate=0.01, 
                       n_estimators=9000,
                       max_bin=200, 
                       bagging_fraction=0.75,
                       bagging_freq=5, 
                       bagging_seed=7,
                       feature_fraction=0.2,
                       feature_fraction_seed=7,
                       min_sum_hessian_in_leaf = 11,
                       verbose=-1,
                       random_state=42)

# XGBoost Regressor
"""xgboost = XGBRegressor(learning_rate=0.01,
                       n_estimators=6000,
                       max_depth=4,
                       min_child_weight=0,
                       gamma=0.6,
                       subsample=0.7,
                       colsample_bytree=0.7,
                       objective='reg:linear',
                       nthread=-1,
                       scale_pos_weight=1,
                       seed=27,
                       reg_alpha=0.00006,
                       random_state=42)"""

"""gbr = GradientBoostingRegressor(n_estimators=6000,
                                learning_rate=0.01,
                                max_depth=4,
                                max_features='sqrt',
                                min_samples_leaf=15,
                                min_samples_split=10,
                                loss='huber',
                                random_state=42)  
"""

# setup models hyperparameters using a pipline
# This is a range of values that the model considers each time in runs a CV
ridge_alpha = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]
lasso_alpha = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]

elastic_alpha = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]

# Ridge Regression: robust to outliers using RobustScaler
ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=ridge_alpha, cv=kfolds))

# Lasso Regression: 
lasso = make_pipeline(RobustScaler(), LassoCV(max_iter=1e7, 
                    alphas=lasso_alpha,random_state=42, cv=kfolds))

# Elastic Net Regression:
elasticnet = make_pipeline(RobustScaler(), ElasticNetCV(max_iter=1e7, 
                         alphas=elastic_alpha, cv=kfolds, l1_ratio=e_l1ratio))

# Support Vector Regression
svr = make_pipeline(RobustScaler(), SVR(C= 20, epsilon= 0.008, gamma=0.0003))

# Stack up all the models above
stack_gen = StackingCVRegressor(regressors=(ridge, lasso, elasticnet, svr, lightgbm),
                                meta_regressor=elasticnet,
                                use_features_in_secondary=True)


# ## Training models

# Get cross validation score for each model.

# In[ ]:


# Store scores of each model
scores = {}

score = cv_rmse(lightgbm)
print("lightgbm: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['lightgbm'] = (score.mean(), score.std())


# In[ ]:


score = cv_rmse(ridge)
print("ridge: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['ridge'] = (score.mean(), score.std())


# In[ ]:


score = cv_rmse(lasso)
print("lasso: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['lasso'] = (score.mean(), score.std())


# In[ ]:


score = cv_rmse(elasticnet)
print("elasticnet: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['elasticnet'] = (score.mean(), score.std())


# In[ ]:


score = cv_rmse(svr)
print("svr: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['svr'] = (score.mean(), score.std())


# In[ ]:


"""score = cv_rmse(gbr)
print("gbr: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['gbr'] = (score.mean(), score.std())"""


# ## Fitting the models

# In[ ]:


print('----START Fit----',datetime.now())
print('Elasticnet')
elastic_model = elasticnet.fit(X, y_train)
print('Lasso')
lasso_model = lasso.fit(X, y_train)
print('Ridge')
ridge_model = ridge.fit(X, y_train)
print('lightgbm')
lgb_model = lightgbm.fit(X, y_train)
print('svr')
svr_model = svr.fit(X, y_train)

print('stack_gen')
stack_gen_model = stack_gen.fit(np.array(X), np.array(y_train))


# ## Blend the models

# In[ ]:


def blend_predictions(X):
    return ((0.12  * elastic_model.predict(X)) +             (0.12 * lasso_model.predict(X)) +             (0.12 * ridge_model.predict(X)) +             (0.22 * lgb_model.predict(X)) +             (0.1 * svr_model.predict(X)) +             (0.32 * stack_gen_model.predict(np.array(X))))


# In[ ]:


# Get final precitions from the blended model
blended_score = rmsle(y_train, blend_predictions(X))
scores['blended'] = (blended_score, 0)
print('RMSLE score on train data:')
print(blended_score)


# ## Find the best model

# In[ ]:


# Plot the predictions for each model
sns.set_style("white")
fig = plt.figure(figsize=(24, 12))

ax = sns.pointplot(x=list(scores.keys()), y=[score for score, _ in scores.values()], markers=['o'], linestyles=['-'])
for i, score in enumerate(scores.values()):
    ax.text(i, score[0] + 0.002, '{:.6f}'.format(score[0]), horizontalalignment='left', size='large', color='black', weight='semibold')

plt.ylabel('Score (RMSE)', size=20, labelpad=12.5)
plt.xlabel('Model', size=20, labelpad=12.5)
plt.tick_params(axis='x', labelsize=13.5)
plt.tick_params(axis='y', labelsize=12.5)

plt.title('Scores of Models', size=20)

plt.show()


# # Submission

# In[ ]:


submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")
submission.iloc[:,1] = np.floor(np.expm1(blend_predictions(X_test)))


# In[ ]:


submission.to_csv("submission.csv", index=False)

