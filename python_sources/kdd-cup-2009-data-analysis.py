#!/usr/bin/env python
# coding: utf-8

# **KDD Cup 2009 Customer Relationship Prediction**
# 
# > KDD Cup is the annual Data Mining and Knowledge Discovery competition organized by ACM Special Interest Group on Knowledge Discovery and Data Mining, the leading professional organization of data miners. 
# > 
# > In the The KDD Cup 2009, a large marketing databases from the French Telecom company Orange had been offered to work on to predict the propensity of customers to 
# > * switch provider (churn), 
# > * buy new products or services (appetency), 
# > * make the sale more profitable (up-selling).
# >

# **CUSTOMER CHURN ANALYSIS**
# >Customer churn is when an existing customer, user, player, subscriber or any kind of return client stops doing business or ends the relationship with a company.
# >
# >The first part of the task is to analyze the data to predict in advance, which customers are likely to churn. 
# >
# > (Appetency and Upselling analysis are also very similar)
# > 
# > The steps are:
# > 1. Acquiring a the dataset which I only used the small version of the data.
# > 2. Clean the dataset and get it into a format convenient for analysis.
# > 3. Summarizing and visualizing important characteristics and statistical properties of the dataset.
# > 4. Appling Machine Learning. Building models to predict the propensity of customers to churn, evaluate the results of models, and eventually, choose the best suitable predictive model to predict customers who may churn.
# > 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns 
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


trainData = pd.read_table('../input/orange_small_train.data').replace('\\', '/')


# In[ ]:


trainData.head(10)


# In[ ]:


trainData.tail(10)


# In[ ]:


trainData.info()


# In[ ]:


churn = pd.read_table('../input/orange_small_train_churn.txt').replace('\\', '/')
churn.head(10)


# In[ ]:


trainData.describe()


# In[ ]:


trainData.corr()


# In[ ]:


trainData.dtypes


# >
# >
# **Converting all the integer columns to float.**
# >
# **object, non-integer, non-float columns to categorical**
# >
# >

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




