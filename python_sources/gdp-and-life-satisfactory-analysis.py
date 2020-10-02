#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# Today I want to do some data analysis.  I downloaded the dataset 'life.csv' from https://stats.oecd.org/index.aspx?DataSetCode=BLI, it is the "better life index" dataset.
# The "gdp.xls" is downloaded from IMF website.  It contains GDP per capita for 190 countries.

# In[ ]:


life = pd.read_csv('../input/life.csv')
gdp = pd.read_csv('../input/gdp.xls', encoding='latin1', delimiter='\t')


# In[ ]:


print(life.columns); print(life.index)
print(gdp.columns); print(gdp.index)


# Let's see how many countries there are in the life dataset and how many there are in the gdp dataset.

# In[ ]:


print(np.unique(life.Country)) # or pd.unique(life.Country)
print("Total number of Countries in dataset life: %d" %len(pd.unique(life.Country)))
#print("Total number of Countries in dataset: {}".format(len(pd.unique(life.Country))))


# In[ ]:


print(np.unique(gdp.Country)) # or pd.unique(life.Country)
print("Total number of Countries in dataset gdp: %d" %len(pd.unique(gdp.Country)))
#print("Total number of Countries in dataset gdp: {}".format(len(pd.unique(gdp.Country))))


# Obviously there are more countries' data available in the gdp dataset, so we need a join operation.  But join is considered an expensive operation, it is better when dataset is small. 
# 
# Take a look at life dataset in column INEQUALITY

# In[ ]:


pd.unique(life.INEQUALITY)


# It has five categories, "TOT" means total, "MN" means man, "WMN" means woman, "HGH" and "LW" 
# stand for high and low respectively. Here we choose only "TOT" category.

# In[ ]:


life = life[life['INEQUALITY'] == 'TOT']


# Next we want to reshape the life dataset, we care only about the life satisfactory index in column 'Value', under different 'INDICATOR'.
# 

# In[ ]:


life = life.pivot(index='Country', columns='Indicator', values='Value')


# In[ ]:


life.head()


# In[ ]:


print(life.shape)
print(life.columns)


# Next we can join the two dataset

# In[ ]:


data = pd.merge(gdp, life, how='inner', on='Country')


# In[ ]:


data.shape


# In[ ]:


data.head()


# The GDP data is in column '2015', better rename it

# In[ ]:


data.rename(columns={'2015':'GDP'}, inplace=True)


# In[ ]:


data.head()


# Finally I have the data reshaped into what we want. Now I need to plot some data

# In[ ]:


sample = data[['GDP', 'Life satisfaction']].iloc[list(np.random.randint(0,len(data),5))]


# In[ ]:


#sample.plot(kind='scatter', x='GDP', y='Life satisfaction')
sample = np.float(sample)


# In[ ]:




