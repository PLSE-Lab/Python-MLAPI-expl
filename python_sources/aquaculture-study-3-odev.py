#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data1 = pd.read_csv('../input/Asia_Quantity.csv')
data2 = pd.read_csv('../input/Europe_Quantity.csv')


# In[ ]:


data1.columns


# In[ ]:


# FILTERING DATA
# Data1 column names have upper characters and spaces. I want to clean the data to be able to use it more easily.
data1.columns = [each.lower() for each in data1.columns]
data1.columns = [each.split()[0]+"_"+each.split()[1] if (len(each.split())>1) else each for each in data1.columns]
data1.columns


# In[ ]:


# To delete each column starting with 's_':
for col in data1.columns:
    if 's_'in col:
        data1.drop([col], axis=1, inplace=True)
data1.head()


# In[ ]:


# EXPLORATORY DATA ANALYSIS
# I want to see the each country in data1
data1.land_area.value_counts()


# In[ ]:


# I want to get Turkey and Cyprus from data1 (asia) to add data2 (europe)
# So I filter data1
filter1 = data1.land_area == 'Turkey'
filter2 = data1.land_area == 'Cyprus'
ndata1 = data1[filter1 | filter2]
ndata1


# Now lets clean data2 for upper char and spaces:

# In[ ]:


data2.columns


# In[ ]:


data2.columns = [each.lower() for each in data2.columns]
data2.columns = [each.split()[0]+'_'+each.split()[1] if (len(each.split())>1) else each for each in data2.columns]
data2.columns


# In[ ]:


for col in data2.columns:
    if 's_' in col:
        data2.drop([col], axis=1, inplace=True)
data2.head()


# In[ ]:


# CONCATENATING DATA
# Now we can concatenate data2 (Europe) with ndata1 (Turkey and Cyprus)
data_eu = pd.concat([data2, ndata1], axis=0, ignore_index = True)
data_eu.land_area.value_counts()


# In[ ]:


data_eu.info()


# In[ ]:


# To see number of rows and columns
data_eu.shape


# In[ ]:


# filter data_eu for exporting crustaceans
data_cr = data_eu[data_eu.commodity == 'Crustaceans']
data_cr = data_cr[data_cr.trade_flow == 'Export']


# In[ ]:


# EXPLORATORY DATA ANALYSIS
data_cr.describe()


# In[ ]:


# VISUAL EXPLORATORY DATA ANALYSIS
data_cr.boxplot(column='2015')


# In[ ]:


data_cr.head()


# In[ ]:


# TIDY DATA
# Melting example
melted = pd.melt(frame=data_cr, id_vars=['land_area','trade_flow'], value_vars=['2015'])
melted


# In[ ]:


# Pivoting Data (reverse of melting)
melted.pivot(index='land_area', columns='variable', values='value')


# In[ ]:


# DATA TYPES
data_cr.dtypes


# In[ ]:




