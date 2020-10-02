#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import os
print(os.listdir("../input"))
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_log_error
from sklearn.impute import SimpleImputer
my_imputer=SimpleImputer()
import gc
# Any results you write to the current directory are saved as output.


# In[ ]:


##all the models
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


# In[ ]:


trainfile="../input/train.csv"
testfile="../input/test.csv"
data=pd.read_csv(trainfile)
test=pd.read_csv(testfile)
print(data.head())


# In[ ]:


#scatter_matrix(data,alpha=.5,figsize=(50,50))


# for improper columns in test and train

# In[ ]:


year_features=['YearBuilt','YearRemodAdd','GarageYrBlt','YrSold'] # these are ordinal values
categorical_features=[col for col in data.columns if data[col].unique().shape[0]<50]
for col in list(set(categorical_features)-set(year_features)):
    data[col]=data[col].astype('category')
    test[col]=test[col].astype('category')
datamod=data.copy()
y=datamod.SalePrice
datamod.drop('SalePrice',axis=1,inplace=True)
testmod=test.copy()
all_data=pd.concat((datamod,testmod),sort=False).reset_index(drop=True)
for col in list(set(categorical_features)-set(year_features)):
    all_data[col]=all_data[col].astype('category')


# one hot encoding

# for col in list(set(categorical_features)-set(year_features)): ## hot_encoding non category and non year features
#     datamod=pd.get_dummies(columns=[col],data=datamod,sparse=True,dummy_na=True,)
# datamod=datamod.drop('SalePrice',axis=1).copy()
# for col in list(set(categorical_features)-set(year_features)): ## hot_encoding non category and non year features
#     testmod=pd.get_dummies(columns=[col],data=testmod,sparse=True,dummy_na=True,)
# for nullcol in list(set(datamod.columns)-set(testmod.columns)):
#     testmod[nullcol]=0
# for nullcol in list(set(testmod.columns)-set(datamod.columns)):
#     datamod[nullcol]=0
# test_mod=testmod.copy()
# data_mod=datamod.copy()

# label encoding

# testmod and datamod were confusing so deleted them from memory to free and also avoid confusing the model

# In[ ]:


for col in list(set(categorical_features)-set(year_features)):
    all_data["{}_encoded".format(col)]=all_data[col].cat.codes
    all_data.drop([col],inplace=True,axis=1)
del datamod,testmod
gc.collect()


# In[ ]:


data_mod=all_data.loc[0:y.shape[0]-1]
test_mod=all_data[y.shape[0]:]


# Lets try with normal linear regressor
# using plain linear regression, accuracy will be bad as too many variables seperated from a single feature

# In[ ]:


def msle_score(y_val,y_predict):
    msle=mean_squared_log_error(y_val,abs(y_predict))
    mae=mean_absolute_error(y_val,abs(y_predict))
    print("MSLE:",msle**(1/2),"MAE:",mae)
def dtree_test(X_train,X_val,y_train,y_val,max_leaf):
    decision_tree_model=DecisionTreeRegressor(max_leaf_nodes=max_leaf,random_state=42)
    decision_tree_model.fit(X_train,y_train)
    y_predict=decision_tree_model.predict(X_val)
    msle=mean_squared_log_error(y_val,abs(y_predict))
    mae=mean_absolute_error(y_val,y_predict)
    print("LeafCount:",max_leaf)
    msle_score(y_val,y_predict)
def rforest_test(X_train,X_val,y_train,y_val,max_leaf,min_leaf):
    random_forest_model=RandomForestRegressor(max_leaf_nodes=max_leaf,random_state=42,min_samples_leaf=5)
    random_forest_model.fit(X_train,y_train)
    y_predict=random_forest_model.predict(X_val)
    print("LeafCount:",max_leaf)
    msle_score(y_val,y_predict)


# Linear model gave score of .1836 when submitted with labelencoder whoo better than 1M error

# In[ ]:


linear_model=LinearRegression()
#data_mod=my_imputer.fit_transform(datamod) ##_mod are imputed values of original
data_mod=my_imputer.fit_transform(data_mod) ##used when using label encoder
X_train,X_val,y_train,y_val=train_test_split(data_mod,y,random_state=42)
linear_model.fit(X_train,y_train)
y_predict=linear_model.predict(X_val)
msle_score(y_val,y_predict)


# In[ ]:


data_mod=my_imputer.fit_transform(data_mod)
linear_model=LinearRegression()
linear_model.fit(data_mod,y)
test_mod=my_imputer.fit_transform(test_mod)
y_test_predict=linear_model.predict(test_mod)
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': y_test_predict})
# you could use any filename. We choose submission here
my_submission.to_csv('submission_linr.csv', index=False)


# using plain decision tree

# In[ ]:


decision_tree_model=DecisionTreeRegressor(random_state=42)
data_mod=my_imputer.fit_transform(data_mod)
X_train,X_val,y_train,y_val=train_test_split(data_mod,y,random_state=42)
decision_tree_model.fit(X_train,y_train)
y_predict=decision_tree_model.predict(X_val)
msle_score(y_val,y_predict)


# In[ ]:


leaf=[5*2**i for i in range(10)]
for i in leaf:
    dtree_test(X_train,X_val,y_train,y_val,i) ##lowest mea around max_leaf_count=80


# optimal max_tree length around 80

# trying random forest classifier
# df_mod is giving better accuracy on data 24K->19K in random forest when using one hot encoder but overfits 

# labelencoding has increasd(decreased) MAE to 17K which is using lesser features 

# In[ ]:


random_forest_model=RandomForestRegressor(random_state=42,n_estimators=100,min_samples_leaf=5)
data_mod=my_imputer.fit_transform(data_mod)
X_train,X_val,y_train,y_val=train_test_split(data_mod,y,random_state=42,)
random_forest_model.fit(X_train,y_train)
y_predict=random_forest_model.predict(X_val)
msle_score(y_val,y_predict)


# In[ ]:


data_mod=my_imputer.fit_transform(data_mod)
random_forest_model=RandomForestRegressor(random_state=42,n_estimators=100,min_samples_leaf=5)
random_forest_model.fit(data_mod,y)
test_mod=my_imputer.fit_transform(test_mod)
y_test_predict=random_forest_model.predict(test_mod)
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': y_test_predict})
my_submission.to_csv('submission_rf.csv', index=False)


# In[ ]:


xgb_model=XGBRegressor(random_state=42,learning_rate=.005,n_estimators=10000)
data_mod=my_imputer.fit_transform(data_mod)
X_train,X_val,y_train,y_val=train_test_split(data_mod,y,random_state=42)
xgb_model.fit(X_train,y_train,verbose=False)
y_predict=xgb_model.predict(X_val)
msle_score(y_val,y_predict)


# In[ ]:


data_mod=my_imputer.fit_transform(data_mod)
xgb_model=XGBRegressor(random_state=42,learning_rate=.005,n_estimators=10000)
xgb_model.fit(data_mod,y,verbose=False)
test_mod=my_imputer.fit_transform(test_mod)
y_test_predict=xgb_model.predict(test_mod)
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': y_test_predict})
my_submission.to_csv('submission_xgb.csv', index=False)


# #lets try finding values which are not in train but in test 
# for col in categorical_features:
#     print(col)
#     print("Values unique to test, not present in train are:")
#     print(list(set(test[col].unique())-set(data[col].unique())))
#     print("Vice-versa :")
#     print(list(set(data[col].unique())-set(test[col].unique())))
#     print("\n")

# **
# After seeing data description Qual,Cond is overall quality ranging from Very Excellent to Very Poor<br>
# maybe try to break it to "poor","avg","good", so we reduce 20 features from OverallCond/Qual to 6(3 each)<br>
# MSSubclass identifies type of dewlling Categorical feature<br>
# ExternQual/Cond is 5 each i.e 10 can be reduced to 6(3 each)<br>
# BsmtCond/Qual/Exposure same as above<br>
# Kitchen/FireplaceQual same to above<br>
# GarageQual/PoolQC same as above<br>**
# 

# In[ ]:




