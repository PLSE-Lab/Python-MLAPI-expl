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


# **Frame the problem**
# 
# Based on the competition's overview, we are tasked to predict sales prices and practice feature engineering, Random forest and gradient boosting. And later we will try to improve our results using advance algorithms.
# 
# For now we will load and explore the data.

# In[ ]:


train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv", index_col ='Id')
test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv", index_col ='Id')


# Let's quick check the data.
# Look for NaN, null and missing values.
# 

# In[ ]:



train.head(5)


# In[ ]:


train.info()


# we have 37 numerical columns and 43 categorical columns. there are also columns that have null values.

# In[ ]:


print("List of Columns:\n" , train.columns)
print("Shape of train data:", train.shape)
print("Shape of test data:", test.shape)


# Summary of numerical attributes:

# In[ ]:


train.describe().T


# We can see that there are a lot of zeros in the summary of numerical data. It may mean that it is a non-numeric features or numbered-qualitative features.

# Histogram for each numerical attributes.

# In[ ]:


# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
categorical_cols = [cname for cname in train.columns if
                    train[cname].nunique() < 10 and 
                    train[cname].dtype == "object"]
high_cardinality_cols = [cname for cname in train.columns if
                    train[cname].nunique() > 10 and 
                    train[cname].dtype == "object"]


# In[ ]:


print("List of categorical columns:\n", categorical_cols+high_cardinality_cols)
print("Number of Categorical columns:",len(categorical_cols+high_cardinality_cols))
print("High Cardinality cols:",high_cardinality_cols )


# In[ ]:


# Select numerical columns
numerical_cols = [cname for cname in train.columns if 
                  train[cname].nunique() >60 and train[cname].dtype in ['int64', 'float64']]
numeric_category = [cname for cname in train.columns if 
                  train[cname].nunique() <60 and train[cname].dtype in ['int64', 'float64']]


# In[ ]:


print("List of numerical columns:\n", numerical_cols+numeric_category)
print("Number of numerical columns:",len(numerical_cols+numeric_category))
print("List of convertable feature:",numeric_category, len(numeric_category) )


# Group features by numerical and categorical, so it will be easy to transform and fit.
# Check for cardinality of categorical features. Low cardinality categories will be easy to transform.
# Check for numeric features that has few unique values. We will check if its ok to convert to categorical feature.
# 
# **Visualize data:**

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm, skew
from sklearn.preprocessing import LabelEncoder
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


fig, ax = plt.subplots()
fig.set_size_inches(14,14)
sns.heatmap(train[numerical_cols].corr(),center = 0, annot=True)


# Make a heatmap to see the correlation of numeric features. 
# 
# List brown and blue correlation not including SalePrice:
# 
# **GarageYrBlt and YearBuilt**
# 
# **1stFlrSF and TotalBsmtSF**
# 
# **GrLivArea and 2ndFlrSF**
# 
# **BsmtUnfSF and BsmtFinSF**
# 
# **GarageYrBlt and YearRemodAdd**
# 
# We have option to drop or retain correlated features.

# In[ ]:


numeric_corr = train[numerical_cols].corr()['SalePrice'][:-1]
high_corr = numeric_corr[abs(numeric_corr) > 0.5].sort_values(ascending=False)
print("List of High Correlation of features to SalePrice:\n", high_corr)


# In[ ]:


fig, ax = plt.subplots()
fig.set_size_inches(14,14)
sns.heatmap(train[numeric_category + ['SalePrice']].corr(),center = 0, annot=True)


# This is a heatmap of numeric features that can be converted to categorical features. Most of them are not correlated. We see that TotRmsAbvGrd and BedroomAbvGrd are highly correlated, we can experiment with these later.

# In[ ]:


categ_corr = train[numeric_category+['SalePrice']].corr()['SalePrice'][:-1]
high_catcor = categ_corr[abs(categ_corr) > 0.5].sort_values(ascending=False)
print("List of High Correlated features to SalesPrice:\n",high_catcor)


# In[ ]:


#cols_none = ['PoolQC', "MiscFeature", "Alley", "Fence","FireplaceQu",'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond','BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',"MasVnrType"]
#train[cols_none] = train[cols_none].fillna("None")
#train["LotFrontage"] = train.groupby("Neighborhood")["LotFrontage"].transform(
#    lambda x: x.fillna(x.median()))
#cols_zero = ['GarageYrBlt', 'GarageArea', 'GarageCars','BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', "MasVnrArea",'BsmtFullBath', 'BsmtHalfBath']
#train[cols_zero] = train[cols_zero].fillna(0)
#train['MSZoning'] = train['MSZoning'].fillna(train['MSZoning'].mode()[0])
#train['Electrical'] = train['Electrical'].fillna(train['Electrical'].mode()[0])
#missing = train.isnull().sum()
#missing = missing[missing > 0]
#missing


# Check for missing values. We can see that Fireplace, Fence, MiscFeature, PoolQC has a lot of no values. We can assume that zero values has no fireplace, fence, alley or pool. 

# In[ ]:


train[numerical_cols].hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)


# Looks like a lot of our numerical features are skewed. Check features that can be log transform.

# In[ ]:


train["SalePrice"] = np.log1p(train["SalePrice"])


# In[ ]:


sns.distplot(train['SalePrice'] , fit = norm)


# In[ ]:


f = plt.figure(figsize=(20,20))
for i in range(len(numerical_cols)):
    f.add_subplot(5, 4, i+1)
    sns.scatterplot(train[numerical_cols].iloc[:,i], train['SalePrice'])
plt.tight_layout()
plt.show()


# Scatterplot numerical attributes vs saleprice

# In[ ]:


f = plt.figure(figsize=(20,20))
for i in range(len(numeric_category)):
    f.add_subplot(5, 4, i+1)
    sns.scatterplot(train[numeric_category].iloc[:,i], train['SalePrice'])
plt.tight_layout()
plt.show()


# **Data Cleaning**
# 
# **Fixing Missing Values**
# 
# There are missing values in our dataset. There are houses that has no garage, pool, fireplace etc. we can change the NaN input as "none". And other missing input we can get their mode and median. We can also use SImpleImputer() to fix mssing values.
# 

# In[ ]:


sns.set_style("whitegrid")
missing = train.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace = True)
missing.plot.bar()


# In[ ]:


cols_none = ['PoolQC', "MiscFeature", "Alley", "Fence","FireplaceQu",'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond','BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',"MasVnrType"]
train[cols_none] = train[cols_none].fillna("None")
train["LotFrontage"] = train.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))
cols_zero = ['GarageYrBlt', 'GarageArea', 'GarageCars','BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', "MasVnrArea",'BsmtFullBath', 'BsmtHalfBath']
train[cols_zero] = train[cols_zero].fillna(0)
train['MSZoning'] = train['MSZoning'].fillna(train['MSZoning'].mode()[0])
train['Electrical'] = train['Electrical'].fillna(train['Electrical'].mode()[0])
missing = train.isnull().sum()
missing = missing[missing > 0]
missing


# **Removing Outliers**

# In[ ]:


train = train.drop(train['LotFrontage'][train['LotFrontage']>200].index)


# In[ ]:


train = train.drop(train['BsmtFinSF1'][train['BsmtFinSF1']>2000].index)


# In[ ]:


train = train.drop(train['LotArea'][train['LotArea']>100000].index)


# In[ ]:


train = train.drop(train['BsmtFinSF2'][train['BsmtFinSF2']>1200].index)
train = train.drop(train['WoodDeckSF'][train['WoodDeckSF']>800].index)
train = train.drop(train['OpenPorchSF'][train['OpenPorchSF']>450].index)
train = train.drop(train['EnclosedPorch'][train['EnclosedPorch']>500].index)
train = train.drop(train['ScreenPorch'][train['SalePrice']>500000].index)


# In[ ]:


train = train.drop(train['TotalBsmtSF'][train['TotalBsmtSF']>2700].index)


# In[ ]:


train = train.drop(train['LotArea'][train['SalePrice']>700000].index)


# In[ ]:


train = train.drop(train['YearBuilt'][(train['YearBuilt']<1900) & (train['SalePrice'] > 400000)].index)


# In[ ]:


train = train.drop(train['MasVnrArea'][train['MasVnrArea']>1200].index)


# **Fixing numerical feautres that are truly categorical**

# In[ ]:


#MSSubClass=The building class
train['MSSubClass'] = train['MSSubClass'].apply(str)

#Changing OverallCond into a categorical variable
train['OverallCond'] = train['OverallCond'].astype(str)

#Year and month sold are transformed into categorical features.
train['YrSold'] = train['YrSold'].astype(str)


# In[ ]:


target = train['SalePrice'].values
train.drop(['SalePrice'], axis=1, inplace=True)


# In[ ]:


cat_cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
for c in cat_cols:
    lbl = LabelEncoder() 
    lbl.fit(list(train[c].values)) 
    train[c] = lbl.transform(list(train[c].values))

# shape        
print('Shape: {}'.format(train.shape))


# In[ ]:


#https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard
#Skewed features
numeric_feats = train.dtypes[train.dtypes != "object"].index
# Check the skew of all numerical features
skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(10)


# In[ ]:


#https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard
skewness = skewness[abs(skewness) > 0.75]

print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    #all_data[feat] += 1
    train[feat] = boxcox1p(train[feat], lam)

train[skewed_features] = np.log1p(train[skewed_features])


# **Handling Text and Categorical Attributes**

# In[ ]:



train = pd.get_dummies(train)


# In[ ]:


print(train.shape)


# **Select and train models**

# In[ ]:


from sklearn.linear_model import ElasticNet, Lasso
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb


# In[ ]:


lasso = make_pipeline(RobustScaler(), Lasso())


# In[ ]:


ENet = make_pipeline(RobustScaler(), ElasticNet())


# In[ ]:


KRR = KernelRidge()


# In[ ]:


GBoost = GradientBoostingRegressor()


# In[ ]:


model_xgb = xgb.XGBRegressor()


# In[ ]:


model_lgb = lgb.LGBMRegressor()


# In[ ]:


forest_reg = RandomForestRegressor()


# **Evaluate Models**

# In[ ]:


#Validation function
n_folds = 5

#def rmsle_cv(model):
  #  kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
 #   rmse= np.sqrt(-cross_val_score(model, train.values, target, scoring="neg_mean_squared_error", cv = kf))
#    return(rmse)
def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    scores= np.sqrt(-cross_val_score(model, train.values, target, scoring="neg_mean_squared_error", cv = kf))
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std(),'\n')


# In[ ]:


rmsle_cv(lasso)


# In[ ]:


rmsle_cv(ENet)


# In[ ]:


rmsle_cv(KRR)


# In[ ]:


rmsle_cv(GBoost)


# In[ ]:


rmsle_cv(model_xgb)


# In[ ]:


rmsle_cv(model_lgb)


# In[ ]:


rmsle_cv(forest_reg)


# In[ ]:


X_train, X_valid, y_train, y_valid = train_test_split(train, target, 
                                                                train_size=0.8, test_size=0.2,
                                                                random_state=0)


# **Fine-tune model**

# In[ ]:


from sklearn.model_selection import GridSearchCV

#param_grid = [    {'n_estimators': [200,240,280,320,400], 'max_features': [50, 60, 80, 90, 100]},    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},  ]
#forest_reg = RandomForestRegressor()
#grid_search = GridSearchCV(forest_reg, param_grid, cv=5,scoring='neg_mean_squared_error', return_train_score=True)
#grid_search.fit(X_train, y_train)


# In[ ]:


#grid_search.best_params_


# In[ ]:


def rmsle(y_train, pred):
    return np.sqrt(mean_squared_error(y_train, pred))


# In[ ]:


forest_reg = RandomForestRegressor(max_features = 80,n_estimators = 280, random_state =42)

forest_reg.fit(X_train, y_train)
forest_reg_train_pred = forest_reg.predict(X_train)
forest_pred = forest_reg.predict(X_valid)
print(rmsle(y_train, forest_reg_train_pred))


# In[ ]:


rmsle_cv(forest_reg)


# In[ ]:


#param_grid = [    {'alpha': [0.00005,0.0005,0.005,0.05,0.5], 'max_iter': [1000,2000,3000]},   ]
#lasso = Lasso()
#grid_search = GridSearchCV(lasso, param_grid, cv=5,scoring='neg_mean_squared_error', return_train_score=True)
#grid_search.fit(X_train, y_train)


# In[ ]:


#grid_search.best_params_


# In[ ]:


#https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1, max_iter = 2000))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3,max_iter = 2000))
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=1825,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)


# In[ ]:


print("Lasso")
rmsle_cv(lasso)
print("Elastic Net")
rmsle_cv(ENet)
print("Kernel Ridge Regressor")
rmsle_cv(KRR)
print("Gradient Boosting Regressor")
rmsle_cv(GBoost)
print("Extreme Gradient Boosting")
rmsle_cv(model_xgb)
print("Light GBM")
rmsle_cv(model_lgb)
print("Random Forest Regressor")
rmsle_cv(forest_reg)


# In[ ]:


GBoost.fit(X_train, y_train)
GBoost_train_pred = GBoost.predict(X_train)
GBoost_pred = GBoost.predict(X_valid)
print(rmsle(y_train, GBoost_train_pred))


# In[ ]:


model_xgb.fit(X_train, y_train)
model_xgb_train_pred = model_xgb.predict(X_train)
model_xgb_pred = model_xgb.predict(X_valid)
print(rmsle(y_train, model_xgb_train_pred))


# In[ ]:


print(model_xgb.predict(X_train[:5]))
print(y_train[:5])


# In[ ]:





# In[ ]:


forest_reg.fit(X_train, y_train)
forest_reg_train_pred = forest_reg.predict(X_train)
forest_pred = forest_reg.predict(X_valid)
print(rmsle(y_train, forest_reg_train_pred))

