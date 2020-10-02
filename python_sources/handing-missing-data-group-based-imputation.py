#!/usr/bin/env python
# coding: utf-8

# ### Group-based Imputation
# 
# Reference: https://www.kdnuggets.com/2017/09/python-data-preparation-case-files-group-based-imputation.html
# 
# To learn about GroupBy (Video): https://www.youtube.com/watch?v=Wb2Tp35dZ-I

# In[2]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[5]:


# reading data from dataset
df = pd.read_csv("../input/mock_bank_data_original_PART1.csv")
df.head(10)


# In[7]:


#  look at the current state of missing values in our dataset
df.isnull().sum()

#  checking account balance ("cheq_balance")   and savings account balances ("savings_balance") 
#  have 23 and 91 missing values, respectively.


# In[24]:


# Let's compute the mean of cheq_balance by state

df.groupby(['state']).mean()['cheq_balance']


# In[25]:


df.groupby(['state']).mean()['savings_balance']


# - It should be clear to see why such group-based imputation is a valid approach to a problem such as this. The mean savings account balance difference between California (\$9174.56) and New York (\$10443.61), for example, is nearly \$1270. Taking an overall mean of \$9603.64 to fill in missing values would not provide the most accurate picture.

# In[ ]:


# Let's go ahead and fill in these missing values by using the Pandas 'groupby' and 'transform' functionality, along with a lambda function. 
# We then round the result in the line of code beneath.


# In[12]:


# Replace cheq_balance NaN with mean cheq_balance of same state
df['cheq_balance'] = df.groupby('state').cheq_balance.transform(lambda x: x.fillna(x.mean()))
df.cheq_balance = df.cheq_balance.round(2)
df.cheq_balance.head()


# In[14]:


# Replace savings_balance NaN with mean savings_balance of same state
df['savings_balance'] = df.groupby('state').savings_balance.transform(lambda x: x.fillna(x.mean()))
df.savings_balance = df.savings_balance.round(2)
df.savings_balance.head()


# In[16]:


# Checking the results 
df.isnull().sum()


# In[ ]:




