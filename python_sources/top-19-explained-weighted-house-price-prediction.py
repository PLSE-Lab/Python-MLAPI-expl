#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # House Pricing Prediction

# ## 1. Background

# > "Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. But this notebook proves that much more influences price negotiations than the number of bedrooms or a white-picket fence."
# 
# 

# ## 2. Problem

# With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, USA. the goal is to predict the final price of each home.

# ## 3. Import Libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.gaussian_process.kernels import ExpSineSquared
from sklearn.preprocessing import LabelEncoder, RobustScaler
from scipy.stats import skew
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV,learning_curve
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
get_ipython().run_line_magic('matplotlib', 'inline')


# ## 4. Gathering Data

# In[ ]:


train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')


# ## 5. EDA - Exploratory Data Analysis

# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


train.info()


# statistical summary

# In[ ]:


train.describe()


# let see mean price

# In[ ]:


train['SalePrice'].dropna().mean()


# In[ ]:


sns.distplot(train['SalePrice'])


# we can see that the ```SalePrice``` data is skewed - we will fix it later.

# let us do the same for year built

# In[ ]:


sns.distplot(train['YearBuilt'])


# we can see that more building were built afther the year 2000

# ## 6. Data Cleaning

# First of all let us explore the outliers by checking the correlation between living area and  house price

# In[ ]:


sns.lmplot(x='GrLivArea', y='SalePrice', data=train)


# we can see here 2 prominent outliers, the two in the bottom with a high living (> 4,000) area but a low price (< 200,000).

# In[ ]:


train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 200000)]


# In[ ]:


train.drop([523,1298], inplace=True)


# In[ ]:


sns.lmplot(x='GrLivArea', y='SalePrice', data=train)


# Now let us combine both data sets - train and test

# In[ ]:


train_length = len(train)
combined = pd.concat([train, test])


# In[ ]:


sns.heatmap(combined.isnull())


# we have alot of missing data

# In[ ]:


combined.isnull().sum().sort_values(ascending=False)[:40]


# In general, i will fill NA value for columns of type object with the defined value for the NA values in the data description, missing values of numeric columns i will fill them with 0.

# PoolQC - Pool quality, possible values are :
# *  Ex   : Excellent
# *  Gd   : Good
# *  TA   : Average/Typical
# *  Fa   : Fair
# *  NA   : No Pool
# 
# NA values defined as ```No Pool``` for the specific house, i will do the same for other features with missing values

# In[ ]:


combined['PoolQC'].fillna('No Pool', inplace=True)
combined['MiscFeature'].fillna('None', inplace=True)
combined['Alley'].fillna('No alley access', inplace=True)
combined['Fence'].fillna('No Fence', inplace=True)
combined['FireplaceQu'].fillna('No Fireplace', inplace=True)


# LotFrontage - Linear feet of street connected to property
# i will fill missing value with the neighborhood mean value 

# In[ ]:


combined["LotFrontage"] = combined.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.dropna().median()))


# In[ ]:


combined[['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']] = combined[['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']].fillna('No Garage')


# since GarageYrBlt, GarageArea, GarageCars are numeric values, i will fill their missing values with 0

# In[ ]:


combined[['GarageYrBlt', 'GarageArea', 'GarageCars']] = combined[['GarageYrBlt', 'GarageArea', 'GarageCars']].fillna(0)


# In[ ]:


combined[['BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtCond', 'BsmtQual']] = combined[['BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtCond', 'BsmtQual']].fillna('No Basement')


# In[ ]:


combined['MasVnrArea'] = combined['MasVnrArea'].fillna(0)
combined['MasVnrType'] = combined['MasVnrType'].fillna('None')


# Sine there is no defined value for NA value for Electrical, i will fill missing value with the mode.

# In[ ]:


combined['Electrical'] = combined['Electrical'].fillna(combined['Electrical'].mode()[0])
combined['MSZoning'] = combined['MSZoning'].fillna(combined['MSZoning'].mode()[0])
combined['Functional'] = combined['Functional'].fillna(combined['Functional'].mode()[0])
combined['Exterior1st'] = combined['Exterior1st'].fillna(combined['Exterior1st'].mode()[0])
combined['Exterior2nd'] = combined['Exterior2nd'].fillna(combined['Exterior2nd'].mode()[0])
combined['KitchenQual'] = combined['KitchenQual'].fillna(combined['KitchenQual'].mode()[0])
combined['SaleType'] = combined['SaleType'].fillna(combined['SaleType'].mode()[0])


# In[ ]:


combined[['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']] = combined[['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']].fillna(0) 


# In[ ]:


sns.countplot(x='Utilities', data=combined)


# almost all of the Utilities values is ```AllPub```, so this will not effect the machine learning process, let us drop this column.

# In[ ]:


combined.drop('Utilities', axis=1, inplace=True)


# In[ ]:


combined['TotalBsmtSF'].fillna(0, inplace=True)


# In[ ]:


combined.isnull().sum().sort_values(ascending=False)[:20]


# No missing data we are good to go.

# ## 6. Feature Engineering

# now let us fix the skewed data of the ```SalePrice```

# In[ ]:


train["SalePrice"] = train["SalePrice"].map(lambda i: np.log1p(i))


# In[ ]:


sns.distplot(train['SalePrice'])


# now we can see that the data is well distributed. 

# now let us find categorical colmns and numeric columns

# In[ ]:


numeric_columns = []
categorical_columns = []
for column in combined.columns:
    if(combined[column].dtype == np.object):
        categorical_columns.append(column)
    else :
        numeric_columns.append(column)


# In[ ]:


len(categorical_columns)


# i will transform categorical columns into numeric labels using ```LabelEncoder```

# In[ ]:


for column in categorical_columns:
    combined[column] = LabelEncoder().fit_transform(combined[column])


# In[ ]:


len(numeric_columns)


# let us check the skewness of the numeric columns

# In[ ]:


skewed_columns = combined[numeric_columns].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)


# In[ ]:


skewed_columns = skewed_columns.apply(abs)


# In[ ]:


for column in skewed_columns[skewed_columns > 0.75].index:
    combined[column] = combined[column].apply(lambda x : np.log1p(x))


# ## 7. Machine Learning Modeling

# In[ ]:


train = combined[:train_length]
test = combined[train_length:].drop('SalePrice', axis=1)


# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


X = train.drop('SalePrice', axis=1)
y = train['SalePrice']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# ### 7.1 Model Defining

# In[ ]:


KRR = KernelRidge()
GBR = GradientBoostingRegressor()
XGB = XGBRegressor()
LGBM = LGBMRegressor()
ENET =  ElasticNet()
LASS =  Lasso()


# In[ ]:


models = [KRR, GBR, XGB, LGBM, ENET, LASS]


# ### 7.2 Cross Validation

# let us define our validation score method

# In[ ]:


k_folds = KFold(5, shuffle=True, random_state=42)

def cross_val_rmse(model):
    return np.sqrt(-1*cross_val_score(model, X, y,scoring="neg_mean_squared_error",cv=k_folds))


# In[ ]:


corss_val_score = []
for model in models:
    model_name = model.__class__.__name__
    corss_val_score.append((model_name,cross_val_rmse(model).mean()))


# In[ ]:


sorted(corss_val_score, key=lambda x : x[1], reverse=True)


# we can see that KernelRidge, GradientBoostingRegressor, XGBRegressor has the best cross validation result. 
# let us continue, plotting learning curve and  finding best estimators after the hyperparameter tuning process.

# let us see the learning curve.

# In[ ]:


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    """Generate a simple plot of the test and training learning curve"""
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


# In[ ]:


for model in models:
    plot_learning_curve(model,model.__class__.__name__ + " mearning curves",X,y,cv=5)


# ### 7.3 Hyperparameter Tuning

# In[ ]:


# lass_param_grid = {
#     'alpha' : [0.001, 0.0005],
#     'random_state':[1,3,42]
# }

# enet_param_grid = {
#     'alpha' : [0.001, 0.0005],
#     'random_state':[1,3,42],
#     'l1_ratio' : [.1,.9] 
# }

# gboost_param_grid ={
#     'n_estimators':[100,3000],
#     'learning_rate': [0.1, 0.05],
#     'max_depth':[4,6],
#     'max_features':['sqrt'],
#     'min_samples_leaf' :[3,9,15],
#     'min_samples_split':[3,10],
#     'loss':['huber'],
#     'random_state':[5,42]
# }

# xgb_param_grid = {
#     'colsample_bytree':[0.1,0.5],
#     'gamma' :[0.01,0.04],
#     'reg_alpha':[0.1,0.5],
#     'reg_lambda':[0.1,0.9],
#     'subsample':[0.1,0.5],
#     'silent':[1],
#     'random_state':[1,7],
#     'nthread':[-1],
#     'learning_rate': [0.1, 0.05],
#     'max_depth': [3,6],
#     'min_child_weight':[1.5,1.4,1.8],
#     'n_estimators': [100,2000]}




# krl_param_grid = {"alpha": [0.1, 0.6],"degree": [2,4], "kernel":['polynomial'], "coef0":[0.5,2.5]}



# lgbm_param_grid = {
#     'n_estimators':[100],
#     'learning_rate': [0.1, 0.05, 0.01],
#     'max_depth':[4,6],
#     'max_leaves':[3,9,17],
# }

# models = [
#     (KernelRidge,krl_param_grid),
#     (XGBRegressor,xgb_param_grid),
#     (GradientBoostingRegressor,gboost_param_grid),
#     (Lasso,lass_param_grid),
#     (ElasticNet,enet_param_grid)
# ]


# In[ ]:


# best_models = []
# for model, param in models:
#     print("Fitting ", model.__class__.__name__)
#     grid_search = GridSearchCV(model(),
#                                scoring='neg_mean_squared_error',
#                                param_grid=param,
#                                cv=5,
#                                verbose=2,
#                                n_jobs=-1)
#     grid_search.fit(X, y)
#     print(grid_search.best_params_)


# After tuning all above models, we can find the best parameters to give the more **negative** root mean squared error, which means the less **positive** root mean squared error.

# In[ ]:


GBR =  GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)



XGB = XGBRegressor(gamma=0.04, learning_rate=0.05, colsample_bytree=0.5,  
                              max_depth=3, 
                             min_child_weight=1.8, n_estimators=2000,
                             reg_alpha=0.5, reg_lambda=0.9,
                             subsample=0.5, silent=1,
                             random_state =7, nthread = -1)
KRR = KernelRidge(kernel='polynomial', alpha=0.6, coef0=2.5, degree=2)

LASS = Lasso(alpha =0.0005, random_state=1)
ENET = ElasticNet( l1_ratio=.9,alpha=0.0005, random_state=3)


# In[ ]:


best_models = [GBR,XGB,KRR,LASS,ENET]


# In[ ]:


corss_val_score = []
for model in best_models:
    model_name = model.__class__.__name__
    print("Fitting ",model_name)
    corss_val_score.append((model_name,cross_val_rmse(model).mean()))


# In[ ]:


sorted(corss_val_score, key=lambda x : x[1], reverse=True)


# ### 7.4 Model Weighted Averaging Ensemble

# In[ ]:


corss_val_score


# In[ ]:


total_rmse = sum([x[1] for x in corss_val_score])


# In[ ]:


weighted_val_score = {}
for k,v in corss_val_score:
    weighted_val_score[k] = round((v/total_rmse)*100)


# In[ ]:


weighted_val_score


# In[ ]:


submission = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv")


# In[ ]:


for model in best_models:
    model.fit(X,y)
    y_pred = model.predict(test)
    submission[model.__class__.__name__] = model.predict(test)
    submission[model.__class__.__name__] = submission[model.__class__.__name__].apply(lambda x: np.expm1(x))


# In[ ]:


submission


# In[ ]:


submission['AVG2'] = submission[['GradientBoostingRegressor','XGBRegressor', 'Lasso', 'ElasticNet']].mean(axis=1)
weighted_average = submission[['Id', 'AVG2']]
weighted_average.rename(columns={'AVG2':'SalePrice'}, inplace=True)
weighted_average.to_csv("AVG2.csv", index=False)


# In[ ]:


submission['AVG3'] = submission[['GradientBoostingRegressor','XGBRegressor', 'ElasticNet']].mean(axis=1)
weighted_average = submission[['Id', 'AVG3']]
weighted_average.rename(columns={'AVG3':'SalePrice'}, inplace=True)
weighted_average.to_csv("AVG3.csv", index=False)


# In[ ]:


submission['AVG4'] = submission[['GradientBoostingRegressor', 'ElasticNet']].mean(axis=1)
weighted_average = submission[['Id', 'AVG4']]
weighted_average.rename(columns={'AVG4':'SalePrice'}, inplace=True)
weighted_average.to_csv("AVG4.csv", index=False)


# In[ ]:


submission['AVG'] = submission[['GradientBoostingRegressor','XGBRegressor', 'KernelRidge', 'Lasso', 'ElasticNet']].mean(axis=1)
weighted_average = submission[['Id', 'AVG']]
weighted_average.rename(columns={'AVG':'SalePrice'}, inplace=True)
weighted_average.to_csv("AVG.csv", index=False)


# In[ ]:


submission['weighted_average'] = (submission['GradientBoostingRegressor']*(0.18))+(submission['XGBRegressor']*(0.18))+(submission['KernelRidge']*(0.28))+(submission['Lasso']*(0.18))+(submission['ElasticNet']*(0.18)) 


# In[ ]:


submission


# In[ ]:


weighted_average = submission[['Id', 'weighted_average']]
weighted_average.rename(columns={'weighted_average':'SalePrice'}, inplace=True)
weighted_average.to_csv("weighted_average.csv", index=False)


# In[ ]:


GRB = submission[['Id', 'GradientBoostingRegressor']]
GRB.rename(columns={'GradientBoostingRegressor':'SalePrice'}, inplace=True)
GRB.to_csv("GRB.csv", index=False)


# In[ ]:


XGBR = submission[['Id', 'XGBRegressor']]
XGBR.rename(columns={'XGBRegressor':'SalePrice'}, inplace=True)
XGBR.to_csv("XGBR.csv", index=False)


# In[ ]:


KRR = submission[['Id', 'KernelRidge']]
KRR.rename(columns={'KernelRidge':'SalePrice'}, inplace=True)
KRR.to_csv("KRR.csv", index=False)


# In[ ]:


LASS = submission[['Id', 'Lasso']]
LASS.rename(columns={'Lasso':'SalePrice'}, inplace=True)
LASS.to_csv("LASS.csv", index=False)


# In[ ]:


ENET = submission[['Id', 'ElasticNet']]
ENET.rename(columns={'ElasticNet':'SalePrice'}, inplace=True)
ENET.to_csv("ENET.csv", index=False)

