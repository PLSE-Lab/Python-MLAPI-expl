#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings("ignore")


# # 1. Load datasets

# In[ ]:


# Load dataset
df_train = pd.read_csv('../input/train.csv')
df_train['dataset'] = 'train'
df_test = pd.read_csv('../input/test.csv')
df_test['dataset'] = 'test'


# In[ ]:


df_train.shape


# Wow !!! There's a lot of  features. From now lets focus on exploratory data first

# # 2. Exploratory data analysis

# ## 2.1 Response

# In[ ]:


# Response (SalePrice)
distplot = sns.distplot(df_train.SalePrice, fit=stats.norm)


# In[ ]:


probplot = stats.probplot(df_train.SalePrice, plot=plt)


# Not really normal distribution, we will fix it with log transformation.<br/>
# Now we have outlier on response. Ideally, we must remove outlier from training dataset.

# In[ ]:


# get standard deviation
response = df_train[['SalePrice']].copy()
response['stddev'] = df_train[['SalePrice']].apply(lambda x: abs(x - x.mean())/x.std())
response[response.stddev > 3].sort_values('SalePrice', ascending=False).min()


# In[ ]:


# SalePrice >= 423000 is outlier
df_train = df_train[df_train.SalePrice < 423000]


# In[ ]:


# Transform response with log function
df_train['SalePrice'] = np.log(df_train['SalePrice'])


# outlier has been remove and log transform applied, lets see distplot again

# In[ ]:


distplot = sns.distplot(df_train.SalePrice, fit=stats.norm)


# yup, its better now

# ## 2.2 Numerical features

# In[ ]:


# select all numeric features, including response
num_features= df_train.select_dtypes(['float64', 'int64']).keys()
len(num_features)


# 38 numeric features. We will select highly correlated features to SalePrice

# In[ ]:


# select correlation > 0.5
num_corr = abs(df_train[num_features].corr()['SalePrice']).sort_values(ascending=False)
num_ok = num_corr[num_corr > 0.5].drop('SalePrice')
num_ok


# Now it's just 10 features. Lets check this plot.

# In[ ]:


for i in range(0, len(num_ok), 5):
    sns.pairplot(data=df_train, x_vars=num_ok[i:i+5].keys(), y_vars='SalePrice', kind='reg')


# This is how its look like features with highly correlated to SalePrice

# In[ ]:


corr_plot = sns.heatmap(df_train[num_ok.keys()].corr(), cmap=plt.cm.Reds, annot=True)


# YearBuilt-GarageYrBlt and TotalBsmtSF-1stFlrSF correlate each other, but leave it be.

# In[ ]:


# check NaN values on numerical falues
df_train[num_ok.keys()].isnull().sum()
train_num_null = df_train[num_ok.keys()].isnull().sum().sort_values(ascending=False)
test_num_null = df_test[num_ok.keys()].isnull().sum().sort_values(ascending=False)
print('null from train dataset:\n{}\n'.format(train_num_null[train_num_null > 0]))
print('null from test dataset:\n{}'.format(test_num_null[test_num_null > 0]))


# There is only one feature that has NaN value on train dataset<br/>
# Issue on test dataset: TotalBsmtSF, GarageArea, GarageCars missing 1 value.<br/>
# We assume NaN equal to 0. We fill it.

# In[ ]:


# fill GarageYrBlt on train dataset
df_train['GarageYrBlt'] = df_train[['GarageYrBlt']].applymap(lambda x: 0 if pd.isnull(x) else x)
# fill GarageYrBlt, TotalBsmtSF, GarageArea, GarageCars on test dataset
for i in test_num_null[test_num_null > 0].keys():
    df_test[i] = df_test[[i]].applymap(lambda x: 0 if pd.isnull(x) else x)


# Now lets see categorical features.<br/>

# ## 2.3 Categorical features

# In[ ]:


cat_features = df_train.select_dtypes(['object'])
len(cat_features.keys())


# We have 44 categorical features.<br/>
# For selecting categorical features i ever hear about features importance from preliminary Random Forest model,<br/>
# but all of features must first convert to numerical.<br/>
# I also hear about ANOVA test or dimensionality reduction but i haven't ever try it yet.<br/>
# So i will selecting based on boxplot and countplot.

# In[ ]:


# after see all plots, i will choose this features
cat_ok = ['MSZoning', 'LotShape', 'LotConfig', 'Neighborhood', 'Condition1', 'BldgType', 'HouseStyle', 'MasVnrType', 'ExterQual', 'Foundation', 'BsmtQual', 'BsmtCond','HeatingQC', 'CentralAir', 'KitchenQual', 'GarageType', 'GarageFinish', 'SaleType', 'SaleCondition']


# In[ ]:


# this is how its look like
plt.rcParams['figure.max_open_warning'] = len(cat_ok)
for i in cat_ok:
    fig, ax = plt.subplots(1, 2, sharey=True, figsize=(16, 3))
    sns.boxplot(x='SalePrice', y=i, data=df_train, ax=ax[0])
    sns.countplot(y=i, data=df_train, ax=ax[1])


# In[ ]:


# check NaN values on categorical features
train_cat_null = df_train[cat_ok].isnull().sum().sort_values(ascending=False)
test_cat_null = df_test[cat_ok].isnull().sum().sort_values(ascending=False)
print('null from train dataset:\n{}\n'.format(train_cat_null[train_cat_null > 0]))
print('null from test dataset:\n{}'.format(test_cat_null[test_cat_null > 0]))


# 8 features contain NaN value on test dataset.<br/>
# from data_description.txt we can understand that:<br/>
# for GarageFinish and GarageType NaN means No Garage<br/>
# for BsmtCond and BsmtQual NaN means No Basement

# In[ ]:


print('GarageFinish:', df_train['GarageFinish'].unique())
print('GarageType:', df_train['GarageType'].unique())
print('BsmtCond:', df_train['BsmtCond'].unique())
print('BsmtQual:', df_train['BsmtQual'].unique())
print('MasVnrType:', df_train['MasVnrType'].unique())


# In[ ]:


# Replace NaN value to 'None'
for i in ['GarageFinish', 'GarageType', 'BsmtCond', 'BsmtQual', 'MasVnrType']:
    df_train[i] = df_train[[i]].applymap(lambda x: 'None' if pd.isnull(x) else x)
    df_test[i] = df_test[[i]].applymap(lambda x: 'None' if pd.isnull(x) else x)


# In[ ]:


print('MSZoning:', df_train.groupby(['MSZoning']).size())
print('\nKitchenQual:', df_train.groupby(['KitchenQual']).size())
print('\nSaleType:', df_train.groupby(['SaleType']).size())


# for NaN value on MSZoning, KitchenQual, SaleType, we can simply replace with most frequent value (mode).

# In[ ]:


df_test['MSZoning'] = df_test['MSZoning'].fillna(df_train['MSZoning'].mode()[0])
df_test['KitchenQual'] = df_test['KitchenQual'].fillna(df_train['KitchenQual'].mode()[0])
df_test['SaleType'] = df_test['SaleType'].fillna(df_train['SaleType'].mode()[0])


# # 3. Preparing data

# Now we must transform predictor (Standarize).

# In[ ]:


# concat train and test dataset
df_test['SalePrice'] = np.nan
df = pd.concat([df_train, df_test], sort=True)


# In[ ]:


# mean and stddev from numeric train dataset
num_mean = df_train[num_ok.index].mean()
num_std = df_train[num_ok.index].std()

# Standarize numeric features
df_num = (df[num_ok.keys()] - num_mean) / num_std


# In[ ]:


# Create dummies for categorical features
df_dummies = pd.get_dummies(df[cat_ok])


# In[ ]:


# dataset column + SalePrice + numeric features + categorical features
df = pd.concat([df.dataset, df.SalePrice, df_num, df_dummies], axis=1)
df.head()


# We finish preprocessing data. Now time to train model.

# # 4. Build model

# In[ ]:


# for splitting train and validate dataset
from sklearn.model_selection import train_test_split

# machine learning regressor
from sklearn.linear_model import LinearRegression, Lasso, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor

# for evaluate model using RootMeanSquaredError
from sklearn.metrics import mean_squared_error


# In[ ]:


X = df[df.dataset == 'train'].drop(['dataset', 'SalePrice'], axis=1)
y = df[df.dataset == 'train'][['SalePrice']]
X_test = df[df.dataset == 'test'].drop(['dataset', 'SalePrice'], axis=1)


# In[ ]:


# X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
X_train, y_train = X, y


# In[ ]:


model = ElasticNetCV(alphas=[0.0001, 0.0005, 0.001, 0.01, 0.1, 1, 10], l1_ratio=[.01, .1, .5, .9, .99], max_iter=5000)
model.fit(X_train, y_train)
# y_pred = model.predict(X_valid)
# mean_squared_error(y_valid, y_pred)


# In[ ]:


X_test.isnull().sum().sum()


# In[ ]:


# Create submission file
y_test_pred = np.exp(model.predict(X_test))
result = df_test[['Id']].copy()
result['SalePrice'] = y_test_pred
result.set_index('Id').to_csv('submission.csv')

