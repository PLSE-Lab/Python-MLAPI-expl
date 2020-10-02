#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)
pd.set_option('display.max_columns', 100)

import os
train_data = pd.read_csv("../input/train.csv", encoding = "ISO-8859-1")
test_data = pd.read_csv("../input/test.csv", encoding = "ISO-8859-1")
train_data.head(n=20)


# In[15]:


train_data.describe()


# In[16]:


def filter_data(dataset):    
    #     dataset = dataset[dataset['TotalBsmtSF'] > 0]
    #     dataset = dataset[dataset['GarageArea'] > 0]
    #     dataset = dataset[dataset['GrLivArea'] > 0]
    #     dataset = dataset[dataset['LotArea'] < 30000]
    return dataset
    
train_data = filter_data(train_data)


# In[17]:


x_vars = ['PoolArea', 'GarageArea', 'YearBuilt', 'OverallQual', 'BedroomAbvGr', 'GrLivArea', 'TotRmsAbvGrd', 'TotalBsmtSF', 'MasVnrArea']
sns.pairplot(data=train_data, x_vars=x_vars, y_vars=['SalePrice'], size=3, kind='reg')
plt.show()


# In[18]:


from math import floor
# from sklearn.preprocessing import MinMaxScaler

y_train = train_data['SalePrice']
del train_data['SalePrice']

def transform(dataset):    
    dataset = pd.concat([dataset, pd.get_dummies(dataset['OverallQual'], prefix='Quality')], axis=1)
    dataset = pd.concat([dataset, pd.get_dummies(dataset['TotRmsAbvGrd'], prefix='NbRooms')], axis=1)
    #dataset = pd.concat([dataset, pd.get_dummies(dataset['YearBuilt'], prefix='YearBuilt')], axis=1)
    dataset['TotalBsmtSF'].fillna(int(floor(dataset['TotalBsmtSF'].mean())), inplace=True)
    dataset['GarageArea'].fillna(int(floor(dataset['GarageArea'].mean())), inplace=True)
    dataset['MasVnrArea'].fillna(int(floor(dataset['MasVnrArea'].mean())), inplace=True)
    del dataset['OverallQual']
    del dataset['TotRmsAbvGrd']
    #del dataset['YearBuilt']
    return dataset

X_train = filter_data(train_data)
X_train = transform(train_data)
X_test = transform(test_data)

predictor_cols = [col for col in X_train 
                  if col.startswith('Quality') 
                      or col.startswith('NbRooms') 
                      or col == 'GrLivArea' 
                      or col == 'TotalBsmtSF' 
                      or col == 'GarageArea'
                 ]

for col_name in predictor_cols:
    if col_name not in X_test.columns:
        X_test[col_name] = 0
        
# cols_to_scale = ['GrLivArea', 'TotalBsmtSF', 'GarageArea', 'MasVnrArea']
# scaler = MinMaxScaler()
# X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
# X_test[cols_to_scale] = scaler.fit_transform(X_test[cols_to_scale])
        
X_train[predictor_cols].head()


# In[32]:


from sklearn import linear_model

clf = linear_model.Lasso(alpha=0.1, max_iter=600000)
clf.fit(X_train[predictor_cols], y_train)

y_pred = clf.predict(X_test[predictor_cols])

print(clf.intercept_)
print(y_pred)
my_submission = pd.DataFrame({'Id': X_test.Id, 'SalePrice': y_pred})
my_submission.to_csv('submission.csv', index=False)


# **Improvements**
# 
# example : https://www.kaggle.com/erikbruin/house-prices-lasso-model-0-119-and-detailed-eda/notebook
