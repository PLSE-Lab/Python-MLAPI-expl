#!/usr/bin/env python
# coding: utf-8

# #### General information
# 
# This is simple machine learning tutorial for absolute begineers. 
# 
# 
# This project demonstrates a simple example of [housing price prediction using Linear Regression](https://www.youtube.com/watch?v=8jazNUpO3lQ&list=PLeo1K3hjS3uvCeTYTeyfe0-rN5r8zn9rw&index=2). It does not cover advanced ML steps. However, new ML concepts and steps will be added as we progress in the tutorial series.

# In[14]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import matplotlib.pyplot as plt
from sklearn import linear_model

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[15]:


# Reading data from CSV

df = pd.read_csv("../input/homeprices.csv")

# For simplicity reason, we have consider two columns. 
# area - It indicates an area of a house.
# price - It indicates the price of a house.

df.head()


# In[16]:


# We are now plotting the data, just to get insight.
# After looking the the plotted graph, it seems that
# it is suitable for linear regression
get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel("area (square fit)")
plt.ylabel("price (US$)")
plt.scatter(df.area, df.price, color='red', marker='+')


# In[17]:


# use Linear Regression model for housing price prediction. 
reg = linear_model.LinearRegression()

# train your linear regression model
reg.fit(df[['area']], df.price)


# In[18]:


# Once we train our model, let's use the trained model to predict the price
# predict function will return the predicted price of a house with 3300 sq ft
reg.predict([[3300]])


# In[19]:


# So far, we have predicted  house price of a one house, given the area of a house.
# Now, we are going to give a list of area.
# Read the list
#area_data = pd.read_csv("../input/arealist/areas.csv")
#area_data.head()


# In[20]:


#reg.predict(area_data)

