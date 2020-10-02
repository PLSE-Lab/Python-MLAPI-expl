#!/usr/bin/env python
# coding: utf-8

# #### General information
# 
# This is simple machine learning tutorial for absolute begineers. 
# 
# 
# This project demonstrates a simple example of [housing price prediction using multiple variables](https://www.youtube.com/watch?v=J_LnPL3Qg70&list=PLeo1K3hjS3uvCeTYTeyfe0-rN5r8zn9rw&index=3). Using this project, we are going to predict the housing price. 
# 
# It does not cover advanced ML steps. However, new ML concepts and steps will be added (step-by-step) as we progress in the tutorial series.

# In[20]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from sklearn import linear_model

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[13]:


# Read data 
data = pd.read_csv("../input/homeprices.csv")

# the data has total four attributes. 
# area, bedrooms and age are features and price is target.
# We will use this data to train our model and predict the housing price, given area, bedrooms and age of a house
data.head()


# In[14]:


# Data preprocessing/data cleaning
# Before we train our model, we need to look the missing values (NaN)
median_bedrooms = data['bedrooms'].median()
median_bedrooms


# In[19]:


# Data pre-processing/data cleaning
data.bedrooms = data.bedrooms.fillna(median_bedrooms)
data


# In[23]:


# Train the model Linear Regression - Multiple variable 
reg = linear_model.LinearRegression()
reg.fit(data[['area', 'bedrooms', 'age']], data.price)


# In[25]:


# Predict the housing price of 3000 sq ft.,3 bedrooms, 40 years old
reg.predict([[3000,3,40]])


# In[ ]:




