#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
print(os.listdir('../input'))



# Any results you write to the current directory are saved as output.


# If we encounter with missing data, what we can do;
# * leave as is
# * drop them with dropna()
# * fill missing values with fillna()
# * fill missing value with test statistics like mean.
# 
# Assert statement: check that you can turn on or turn off when you are done with your testing of the program.

# In[ ]:


data = pd.read_csv('../input/world-bank-wdi-212-health-systems/2.12_Health_systems.csv')


# In[ ]:


data.info()


# In[ ]:


data.head(10)


# In[ ]:


#Lets check External_health_exp_pct_2016
data['External_health_exp_pct_2016'].value_counts(dropna=False)


# In[ ]:


#Lets drop NaN values
data1 = data
data1['External_health_exp_pct_2016'].dropna(inplace=True)
#Dropna() is drop the NaN values in the data and then, save to data1. inplace = true ,we assignin to new variable.
data1.head(10)


# In[ ]:


#We want to check with assert statements for this code block.
assert data['External_health_exp_pct_2016'].notnull().all()#Return nothing because we drop non values.


# In[ ]:


data['External_health_exp_pct_2016'].fillna('empty', inplace=True)
assert data['External_health_exp_pct_2016'].notnull().all()#Return nothing because we drop  non values.


# In[ ]:


assert data.columns[0] == 'Country_Region'#return nothing because this statement is true.

