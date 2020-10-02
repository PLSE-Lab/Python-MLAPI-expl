#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Import libraries.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.feature_extraction import DictVectorizer
import statsmodels.regression.linear_model as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
get_ipython().run_line_magic('matplotlib', 'notebook')


# In[ ]:


#Import dataset.
train = pd.read_csv('../input/home-data-for-ml-course/train.csv', delimiter=',')
test = pd.read_csv('../input/home-data-for-ml-course/test.csv', delimiter=',')

train.set_index('Id', inplace=True)
test.set_index('Id', inplace=True)

print('train set size : {}'.format(train.shape))
print('test set size : {}'.format(test.shape))


# #  1) Exploratory Data Analysis

# Objectives:
# 
# - Create numerical/categorical variables DataFrame.
# - Repartition of values for each numerical/categorical features.
# - Check distplot/countplot to see which feature can be dropped.
# - Check if can transform numerical feature into categorical one (resp)

# ### a) Numerical variables 

# In[ ]:


train.columns[train.dtypes != 'object']


# In[ ]:


numerical = train.select_dtypes(exclude='object').drop('SalePrice', axis=1).copy()
numerical


# In[ ]:


numerical.shape


# In[ ]:


# #Repartition of values for each numerical features.
# for elt in numerical.columns:
#     print("{} -> {}".format(elt, numerical[elt].unique()), end='\n\n')


# Distribution of numerical variable.

# In[ ]:


# Histogram.
fig = plt.figure(figsize=(12,18))

for i in range(len(numerical.columns)):
   fig.add_subplot(9,4,i+1)
   sns.distplot(numerical.iloc[:,i].dropna(), kde=False)
   plt.xlabel(numerical.columns[i])

plt.tight_layout()


# Certains numerical features are in fact categorical variable:
# 
# - MSSubClass

# ### b) Categorical variables

# In[ ]:


train.columns[train.dtypes == 'object']


# In[ ]:


categorical = train.select_dtypes(include='object').copy()
categorical


# In[ ]:


categorical.shape


# In[ ]:


#Repartition of values for each categorical features.
for elt in categorical.columns:
    print("{} -> {}".format(elt, categorical[elt].unique()), end='\n\n')


# Distribution of categorical variable.

# In[ ]:


#Histogram.
fig = plt.figure(figsize=(15,25))

for i in range(len(categorical.columns)):
   fig.add_subplot(9,5,i+1)
   ax = sns.countplot(categorical.iloc[:,i].dropna())
   ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right", fontsize=7)
   plt.xlabel(categorical.columns[i])

plt.tight_layout()


# In[ ]:


train['Utilities'].value_counts()


# Note:
# 
# We can remove 'Utilities' because it is filled only of AllPub, we can remove it.
# 
# Utilities: Type of utilities available
# 		
#        AllPub	All public Utilities (E,G,W,& S)	
#        NoSewr	Electricity, Gas, and Water (Septic Tank)
#        NoSeWa	Electricity and Gas Only
#        ELO	Electricity only	
# 

# ### c) Target variable analysis

# #### i) skewness & kurtosis

# In[ ]:


#plt.title('Distribution of SalePrice')
#sns.distplot(train['SalePrice'])

#fig = plt.figure()
#stats.probplot(train['SalePrice'], plot=plt)

print("Skewness: %f" % train['SalePrice'].skew())
print("Kurtosis: %f" % train['SalePrice'].kurt())


# Note:
# 
# - Perform log transformation on target variable to fix skewness.
# - Remove outliers to fix kurtosis.

# #### ii) Numerical columns and target variable

# Bivariate analysis - scatter plots for target versus numerical attributes (Find outliers)

# In[ ]:


f = plt.figure(figsize=(12,20))

for i in range(len(numerical.columns)):
   f.add_subplot(9, 4, i+1)
   sns.scatterplot(numerical.iloc[:,i], train['SalePrice'])
   
plt.tight_layout()


# Note:
# 
# Based on a first viewing of the scatter plots against SalePrice, there appears to be:
# 
# - A few outliers on the LotFrontage (say, >200) and LotArea (>100000) data.
# - BsmtFinSF1 (>4000) and TotalBsmtSF (>6000)
# - 1stFlrSF (>4000)
# - GrLivArea (>4000 AND SalePrice <300000)
# - LowQualFinSF (>550) 

# ### d) Correlation Matrix

# We need to remove highly correlated features.

# In[ ]:


correlation_mat = train.corr()

f, ax = plt.subplots(figsize=(12,9))
plt.title('Correlation of numerical attributes', size=16)
sns.heatmap(correlation_mat, vmin=0.2, vmax=0.8, square=True, cmap='BuPu')
plt.show()


# Note:
# 
# Those features are highly-correlated (multicollinearity). Remove the one that is less-correlated with target variable, denoted with (x).
#   
# - GarageArea (x) / GarageCars
# - 1stFlrSF / TotalBsmtSF (x)
# - GarageYrBlt (x) / YearBuilt
# - TotRmsAbvGrd (x) / GrLivArea
# 

# In[ ]:


correlation_mat['SalePrice'].sort_values(ascending=False)


# ### e) Missing values check

# In[ ]:


#Numerical
percentage_missing = numerical.isna().sum() / len(train) * 100
percentage_missing.sort_values(ascending=False)


# In[ ]:


#Categorical
percentage_missing = categorical.isna().sum() / len(train) * 100
percentage_missing.sort_values(ascending=False)


# Note:
# 
# - Could it be possible for a missing value 'NaN' to have another meaning ? 
#     - PoolQC -> pool quality. It is described by "Ed/Fa/Ex". 'NaN' can mean here that there is no pool at all.

# In[ ]:


train.PoolQC.value_counts()


# # 2) Data Cleaning & Preprocessing

# ### a) Adressing kurtosis (outliers)

# In[ ]:


#Remove outliers in LotFrontage.
train.drop(train[train['LotFrontage'] > 200].index, inplace=True)

#Remove outliers in LotArea.
train.drop(train[train['LotArea'] > 100000].index, inplace=True)

#Remove outliers in BsmtFinSF1.
train.drop(train[train['BsmtFinSF1'] > 4000].index, inplace=True)

#No need to remove outliers in TotalBsmtSF because we will delete this feature further (high-correlated).

#Remove outliers in 1stFlrSF.
train.drop(train[train['1stFlrSF'] > 4000].index, inplace=True)

#Remove outliers in GrLivArea.
train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index, inplace=True)

#Remove outliers in LowQualFinSF.
train.drop(train[train['LowQualFinSF'] > 550].index, inplace=True)


#fig = plt.figure()
plt.title('Probability plot with kurtosis fixed')
stats.probplot(train['SalePrice'], plot=plt)
print("Kurtosis: %f" % train['SalePrice'].kurt())


# ### b) Adressing skewness (target variable)

# To perform regression, we have to follow some assumptions. Two of them are "Normality of error" (Multivariate Normality) and  "Homoscedasticity" (Constant Error Variance).
# 
# - Normality of error: Error (difference between predicted value and exact value) has to follow a normal distribution.
# 
# - Homoscedasticity: Error has a constant variance.
# 
# When you apply log function on y, you are normalizing it which means that y will have a mean = 0 and a variance = 1.
# ("Normality of error" assumption checked). Since variance = 1 which is a constant, error has a constant variance ("Homoscedasticity" assumption checked).

# In[ ]:


train['SalePrice'] = np.log(train['SalePrice'])


# In[ ]:


fig = plt.figure()
plt.title('Distribution of SalePrice without skewness')
sns.distplot(train['SalePrice'])
print("Skewness: %f" % train['SalePrice'].skew())


# ### c) Concatenate train/test set

# In[ ]:


#Concat train/test set.
all_data = pd.concat([train.drop('SalePrice', axis=1), test], sort=False)


# In[ ]:


all_data.shape


# ### d) Adressing skewness (independant variable)

# No assumptions required us to normalize our independant variables. If we want to do so, we are not "normalizing" but rather making it easier to detect "multicolinearity" (one of the assumption).

# ### e) Dealing with missing values

# In[ ]:


#Numerical missing values.
all_data[numerical.columns].isna().sum().sort_values(ascending=False).head(15)


# In[ ]:


tmp = ['LotFrontage', 'GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1','BsmtFinSF2', 'BsmtUnfSF']

all_data[tmp] = all_data[tmp].fillna(0)
all_data[numerical.columns] = all_data[numerical.columns].fillna(all_data[numerical.columns].mean())


# In[ ]:


#Categorical missing values.
all_data[categorical.columns].isna().sum().sort_values(ascending=False).head(15)


# In[ ]:


tmp = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageCond', 'GarageQual', 'GarageFinish',
       'GarageType', 'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
all_data[tmp] = all_data[tmp].fillna('None')

all_data[categorical.columns] = all_data[categorical.columns].fillna(all_data[categorical.columns].mode().iloc[0, :])


# # 3) Feature selection & Engineering

# ### a) Convert numerical variable to categorical (resp)

# In[ ]:


all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)
all_data['OverallCond'] = all_data['OverallCond'].astype(str)
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)


# ### b) Creating new features

# In[ ]:


all_data['TotalBathroom'] = all_data['FullBath'] + all_data['HalfBath']
all_data.drop(['FullBath', 'HalfBath'],axis=1,inplace=True)


# ### c) Adressing highly-correlated independant features 

# In[ ]:


features_to_drop = ['GarageArea', 'TotalBsmtSF', 'GarageYrBlt', 'TotRmsAbvGrd']
all_data.drop(columns=features_to_drop, inplace=True)


# ### d) Adressing useless features

# In[ ]:


all_data.drop(columns='Utilities', inplace=True)


# ### e) Encoding categorical values

# In[ ]:


all_data = pd.get_dummies(all_data)


# In[ ]:


all_data.shape


# ### f) Split all_data into train/test set 

# In[ ]:


nTrain = train.SalePrice.shape[0]


# In[ ]:


train_transf = all_data[:nTrain]
test_transf = all_data[nTrain:]


# # 4) Build a model

# In[ ]:


#Split dataset into training and validation set.
#X_train, X_val, y_train, y_val = train_test_split(train_transf, train['SalePrice'], random_state=1)


# In[ ]:


from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso, LinearRegression, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


# ### a) Hyperparameter tuning

# In[ ]:


lin_reg = LinearRegression()
lasso = Lasso(alpha =0.0005, random_state=1)
ridge = Ridge(alpha =0.0005, random_state=1)
Enet = ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3)
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
model_xgb = XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
model_lgb = LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

base_models = [lin_reg, lasso, ridge, Enet, GBoost, model_xgb, model_lgb]


# ### b) Stacking 

# In[ ]:


# header_list = ['y_lin_reg', 'y_lasso', 'y_ridge', 'y_Enet', 'y_GBoost', 'y_model_xgb', 'y_model_lgb', 'y_cb']
header_list = ['y_lin_reg', 'y_lasso', 'y_ridge', 'y_Enet', 'y_GBoost', 'y_model_xgb', 'y_model_lgb']
new_train_dataset = pd.DataFrame(columns=header_list)
new_test_dataset = pd.DataFrame()
#Enable us to pick the meta model.
mae_compare = pd.Series(index=header_list)


# In[ ]:


kfold = KFold(n_splits=6, random_state=42)

#For each model.
for model, header in zip(base_models, header_list):
    #Fit 80% of the training set (train_index) and predict on the remaining 20% (oof_index). 
    for train_index, oof_index in kfold.split(train_transf, train['SalePrice']):
        X_train, X_val = train_transf.iloc[train_index, :], train_transf.iloc[oof_index, :]
        y_train, y_val = train.SalePrice.iloc[train_index], train.SalePrice.iloc[oof_index]
    
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        new_train_dataset[header] = y_pred
        
        mae_compare[header] = mean_absolute_error(np.exp(y_val), np.exp(y_pred))
    
    #Create new_test_set at the same time.
    print(header)
    new_test_dataset[header] =  model.predict(test_transf)

# #Add y_val to new_train_dataset.
# #If we don't drop the ID, we will get NaN if ID doesn't match with new_train_dataset index.
new_train_dataset['y_val'] = y_val.reset_index(drop=True)


# In[ ]:


#Pick the meta-model.
mae_compare.sort_values(ascending=True)


# In[ ]:


#Train meta-model on new_train_dataset.
lasso.fit(new_train_dataset.iloc[:, :-1], new_train_dataset.iloc[:, -1])


# In[ ]:


#Apply train meta-model to new_test_dataset.
y_meta_pred = lasso.predict(new_test_dataset)


# In[ ]:


print('{}: {}'.format('train_transf',train_transf.shape))
print('{}: {}'.format('test_transf',test_transf.shape))
print('{}: {}'.format('X_train',X_val.shape))
print('{}: {}'.format('y_train',y_val.shape))
print('{}: {}'.format('X_val',X_val.shape))
print('{}: {}'.format('y_val',y_val.shape))
print('{}: {}'.format('new_train_dataset',new_train_dataset.shape))
print('{}: {}'.format('new_test_dataset',new_test_dataset.shape))
print('{}: {}'.format('y_meta_pred',y_meta_pred.shape))


# In[ ]:


# #Submission
# output = pd.DataFrame({'Id': test.index,
#                        'SalePrice': (np.exp(y_meta_pred))})
# output.to_csv('stacking_submission.csv', index=False)

