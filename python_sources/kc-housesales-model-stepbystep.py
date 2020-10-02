#!/usr/bin/env python
# coding: utf-8

# # KC_Housesales_Model

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Importing required packages

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# Getting the dataset

# In[ ]:


data=pd.read_csv("/kaggle/input/kc-housesales-data/kc_house_data.csv")


# Viewing Data and To see if if null values if any

# In[ ]:


data.head(5)


# here target variable=price
# 
# and
# 
# other are feature variables

# In[ ]:


data.describe(include="all")


# 2.159700e+04 == 21,597.00
# 
# 5.402966e+05 ==  540,296.6

# In[ ]:


data.isna().sum()


# No null values 
# 
# Hurrah!!!

# finding number of uniques

# In[ ]:


r=data.columns
for i in r:
    print("'",i,"'has these many uniques",data[i].nunique())


# we can drop id column as they are so unique

# In[ ]:


data=data.drop(["id"],axis=1)


# let us separate features and target variable

# In[ ]:


X=data.drop(["price"],axis=1)
y=data["price"]


# # EDA

# In[ ]:


ax = sns.scatterplot(x="price", y="date", data=data)


# there are few ouliers otherwise uniformly distributed

# In[ ]:


X=X.drop(["date"],axis=1)


# In[ ]:


X.dtypes


# In[ ]:


ax = sns.heatmap(data.corr())


# all features seems to be correlated

# # linear regression

# In[ ]:


from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[ ]:


xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.35,random_state=0)


# In[ ]:


lr=LinearRegression(fit_intercept=True)
model=lr.fit(xtrain,ytrain)
prediction=lr.predict(xtest)
print("Train_Accuracy")
print(lr.score(xtrain,ytrain))
print("Test_Accuracy")
print(lr.score(xtest,ytest))


# # RandomForestRegressor

# In[ ]:


from sklearn.ensemble import RandomForestRegressor


# In[ ]:


regressor = RandomForestRegressor(n_estimators=100,max_features='auto',max_depth=100,min_samples_leaf=4,min_samples_split=10,random_state=0)
model=regressor.fit(xtrain, ytrain)
y_pred = regressor.predict(xtest)
print("Train_Accuracy")
print(regressor.score(xtrain,ytrain))
print("Test_Accuracy")
print(regressor.score(xtest,ytest))


# Still learning feature engineering and feature selection which can increase the efficiency more.
# 
# do give a suggestion.
# 
# Exploring Learning Improving....
