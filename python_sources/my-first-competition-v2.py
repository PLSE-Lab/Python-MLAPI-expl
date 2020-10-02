#!/usr/bin/env python
# coding: utf-8

# **Getting started with Kaggle**

# This is my first trial writing a notebook in Kaggle.

# In[ ]:


# Loading some useful libraries

import numpy as np 
import pandas as pd 
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


#Importing train and test data

train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')


# Visualization of the top five rows of the train data set:

# In[ ]:


train.head()


# More information on the train data set, with the minimum, maximum and mean values of each column:

# In[ ]:


train.describe()


# In[ ]:


#Name of each of the colums in the train data set

train.columns


# In[ ]:


#Define the y variable (price)

Y_train = train.SalePrice


# I will split the train data set in two datala set for model building and validation before prediction the value for the test data set.

# In[ ]:


Y_train.head()


# In[ ]:


train.describe()


# In[ ]:


train.info()


# I will start considering only some quantitative features with no missing values in the train data set:

# In[ ]:


features = ['MSSubClass', 'LotArea', 
        'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',  '1stFlrSF', 'YrSold', 'OpenPorchSF' , 'MiscVal', 'BedroomAbvGr', 'TotRmsAbvGrd', 
            'TotalBsmtSF', 'GrLivArea']# 'EnclosedPorch']# ,  '3SsnPorch', 'ScreenPorch'  ]


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train = train[features]

train_X, val_X, train_y, val_y = train_test_split(X_train, Y_train, random_state = 0)


# In[ ]:


X_train.head()


# In[ ]:


train_X.info()


# In[ ]:


#from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Define model. Specify a number for random_state to ensure same results each run
model = XGBRegressor(n_estimators=300, learning_rate=0.15)

# Fit model
model.fit(train_X, train_y)


# In[ ]:


Y_train_prediction = model.predict(train_X)


# In[ ]:


from sklearn.metrics import mean_absolute_error

mae1 = mean_absolute_error(train_y, Y_train_prediction)

Y_val_prediction = model.predict(val_X)

mae2 = mean_absolute_error(val_y, Y_val_prediction)

print(mae1)
print(mae2)


# ![](http://)Lowest value so far 25.927
# but validation mae is 17805

# In[ ]:


test.info()


# In[ ]:


X_test = test[features]

Y_test = model.predict(X_test)

print(Y_test)


# In[ ]:


my_submission = test.Id


# In[ ]:


my_submission.head()


# In[ ]:


my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': Y_test})


# In[ ]:


my_submission.to_csv('submission.csv', index=False)

print(my_submission.head())


# In[ ]:




