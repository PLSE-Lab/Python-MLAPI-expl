#!/usr/bin/env python
# coding: utf-8

# In[27]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestRegressor

import os
print(os.listdir("../input"))


# Explaratory Data Analysis: Taking in the data and exploring the insides so that we can learn what model to apply to the data to best predict new housing prices.

# In[28]:


test_set = pd.read_csv("../input/test.csv")
train_set = pd.read_csv("../input/train.csv")


# Let's take a look at the head of the data.

# In[29]:


train_set.head()


# Let's get the shape of the data to tell rows and columns. 

# In[30]:


print(test_set.shape)
print(train_set.shape)


# Let's take a look at the problem at hand here: 
# 
# What are we trying to predict? Housing Prices
# 
# What do we think are possible variables that could explain this? LotArea, OverallQual, OverallCond, etc. are the most generalizeable of the variables, so I think it makes sense to explore these ones first and add other variables to make the model more precise if necessary.

# In[31]:


train_outcome_variables = train_set["SalePrice"]

predictor_variables = ["LotArea","OverallQual","OverallCond","BedroomAbvGr"]
train_predictor_variables = train_set[predictor_variables]

model = RandomForestRegressor()
model.fit(train_predictor_variables, train_outcome_variables)


# In[32]:


test_predictor_values = test_set[predictor_variables]
test_prices = model.predict(test_predictor_values)
print(test_prices)


# In[34]:


my_submission = pd.DataFrame({'Id':test_set['Id'], 'SalePrice':test_prices})

my_submission.to_csv("submission.csv", index=False)

