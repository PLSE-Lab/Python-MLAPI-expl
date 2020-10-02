#!/usr/bin/env python
# coding: utf-8

# Hello.
# I have changed the dataset because of the old dataset(World Happiness Report) had no value 'NaN'.
# 
# This homerwork kernel is about these issues :
# * Diagnose data for cleaning
# * Explotanory data analysis 
# * Visual explotanory data analysis
# * Tidy Data
# * Pivoting Data
# * Concatenating Dataframes
# * DataTypes
# * Missing Data and Test with Asserts
# 

# In[ ]:


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


# In[ ]:


data = pd.read_csv('../input/anime.csv')
data.info()


# Lets check frequency value of the 'Type ' column.

# In[ ]:


print(data['type'].value_counts(dropna=True))


# ### EXPLORATORY DATA ANALYSIS
# 
# 
# <br>outliers: the value that is considerably higher or lower from rest of the data
# * Lets say value at 75% is Q3 and value at 25% is Q1. 
# * Outlier are smaller than Q1 - 1.5(Q3-Q1) and bigger than Q3 + 1.5(Q3-Q1). (Q3-Q1) = IQR
# <br>We will use describe() method. Describe method includes:
# * count: number of entries
# * mean: average of entries
# * std: standart deviation
# * min: minimum entry
# * 25%: first quantile
# * 50%: median or second quantile
# * 75%: third quantile
# * max: maximum entry
# 
# <br> What is quantile?
# 
# * 1,4,5,6,8,9,11,12,13,14,15,16,17
# * The median is the number that is in **middle** of the sequence. In this case it would be 11.
# 
# * The lower quartile is the median in between the smallest number and the median i.e. in between 1 and 11, which is 6.
# * The upper quartile, you find the median between the median and the largest number i.e. between 11 and 17, which will be 14 according to the question above.

# In[ ]:


data.describe()


# ### VISUAL EXPLORATORY DATA ANALYSIS
# * Box plots: visualize basic statistics like outliers, min/max or quantiles

# In[ ]:


data.boxplot(column = 'rating', by='type', figsize = (13,13))


# ### Melting & Pivoting Dataset
# 
# When we would see especially the result of relation between different columns, we using melting function.
# Pivoting function is reverse of melting.****

# In[ ]:


#id_vars is what we don't want to wish to melt
#value_vars is what we want to wish to melt
data_new = data.head()
melted = pd.melt(frame=data_new, id_vars = 'name', value_vars =['genre', 'type'])
melted


# In[ ]:


melted.pivot(index = 'name', columns = 'variable', values = 'value')


# ### Concatenating Dataframes
# We will concatenate two dataframes. We will do it from two different ways : 
# * Vertical concatenate
# * Horizontal concatenate

# In[ ]:


data1 = data.head()
data2 = data.tail()
conc_data_row = pd.concat([data1,data2], axis = 0, ignore_index=True)#axis=0 is meaning, vertical concat.
conc_data_row


# In[ ]:


data_conc_cols = pd.concat([data1,data2], axis = 1 , ignore_index=True)#axis = 1 is meaning, horizontal cocnat.
data_conc_cols


# ### Data Types 
# 
# There are 5 basic data types: object(string),booleab,  integer, float and categorical.
# <br> We can make conversion data types like from str to categorical or from int to float
# <br> Why is category important: 
# * make dataframe smaller in memory 
# * can be utilized for anlaysis especially for sklear(we will learn later)

# In[ ]:


data.dtypes


# In[ ]:


data['type'] = data['type'].astype('category')
data['anime_id'] = data['anime_id'].astype('float')
data.dtypes


# <a id="24"></a> <br>
# ### MISSING DATA and TESTING WITH ASSERT
# If we encounter with missing data, what we can do:
# * leave as is
# * drop them with dropna()
# * fill missing value with fillna()
# * fill missing values with test statistics like mean
# <br>Assert statement: check that you can turn on or turn off when you are done with your testing of the program

# In[ ]:


data.info()
#there are 12294 object in out dataframe.
#but as we can see, there are 12064 rating value at dataframe


# In[ ]:


data['rating'].value_counts(dropna=False)
#there are 230 NaN value.


# In[ ]:


dataNew = data
dataNew['rating'].dropna(inplace =True)


# ### How can we use ' Assert ' ?
# Assert statement works like ' if statement. If assert gets boolean 1, returns nothing. But if assert gets boolean 0, returns error.

# In[ ]:


assert 1 == 1 # returns nothing.


# In[ ]:


assert 1 == 2 # returns error.


# In[ ]:


assert dataNew['rating'].dropna().all()#returns nothing because of we dropped all NaN values already.

