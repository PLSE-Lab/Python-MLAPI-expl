#!/usr/bin/env python
# coding: utf-8

# This data set consists of:
# *Band names 
# Fans 
# Formed 
# Origin
# Spilt 
# Style*
# 

# In[12]:


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


# In[13]:


# taking it into a pandas data frame
bands = pd.read_csv("../input/metal_bands_2017.csv", low_memory= False,encoding='latin-1')
print(pd.DataFrame.describe(bands))
    


# Test for NA, NULL values in the data set. If yes, remove them.

# In[14]:


bands = bands.dropna(axis= 0, how= 'any')


# Since  I am trying to identify the number of bands in each country. Lets group the band name and origin into a different column

# In[15]:


group1 = bands[['band_name','origin']] #create  a new data frame using band_name and origin
group1 = group1.groupby(['origin']).count() #since we need the total number of countries which have the bands and so on
# lets rename the columns 
group1 = group1.rename(columns={'band_name':'number_of_bands'}) #re name the column
group1 = group1.sort_values('number_of_bands', ascending= False) #sorting the values 
group2 = group1.head(20) #taking the top 20 values in order to plot in the graph
print(group2)


# **Bar  Graph and Histogram**
# 
# 
# 

# In[50]:


bands_bar = group2.plot(kind='barh', fontsize= 10, color= ['purple'])
bands_hist = group2.plot(kind= 'hist', fontsize=10, color = ['y'])


# Create a histogram showing the number of fans versus the band names 
# 

# In[ ]:





# 
