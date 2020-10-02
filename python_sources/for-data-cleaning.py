#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv("../input/Pokemon.csv")


# **DIAGNOSE DATA for CLEANING**
# We need to diagnose and clean data before exploring.
# Unclean data:
# * Column name inconsistency like upper-lower case letter or space between words
# * missing data
# * different language
# 
# We will use head, tail, columns, shape and info methods to diagnose data
# 
# 

# In[ ]:


data.head()


# In[ ]:


data.tail()


# In[ ]:


data.columns


# In[ ]:


data.shape


# In[ ]:


data.info()


# EXPLOTARY DATA ANALYSIS (EDA)
# 
# value_counts() : Frequency counts
# outliers : the value that is considerably higher or lower from rest of the data
# 
# * Lets say value at 75% is Q3 and values at 25% is Q1
# * Outlier are smaller than Q1 - 1.5(Q3-Q1) and bigger than Q3 + 1.5(Q3-Q1).
# (Q3-Q1) =IQR
# We will use describe() method. Describe method includes:
# * count : number of entries
# * mean : average of entries
# * std : standart deviation
# * min : minumun entry
# 
# Our series:  1,2,3,4,5,6,7,8,9
# 
# * Q1 : 25% : first quartile (The lower quartile is the median in between the smallest number and the median => 3)
# * 50% : median or second quartile (middle of series = 5)
# * Q3 : 75%: third quartile (The higher quartile is the median in between the highest number and the median => 7)
# * max : maximum entry
# 

# In[ ]:


# For example lets look frequnecy of pokemon types
print(data["Type 1"].value_counts(dropna = False)) # if there are nan values that also be counted


# In[ ]:


# For example max HP is 255 or min defense is 5
data.describe() # ignore null entries


# **VISUAL EXPLORATORY DATA ANALYSIS**
# 
#  Box plots : visualize basic statistics like outliers, min/max or quartiles

# In[ ]:


# circles mean outlies value

data.boxplot(column='Attack', by = 'Legendary')
plt.show()


# **TIDY DATA**
# 
# We tidy data with melt(). Describing melt is confusing. Therefore lets make example to understand it
# 

# In[ ]:


# Firstly I create new data from pokemons data to explain melt nore easily
data_new = data.head()
data_new


# In[ ]:


# lets melt
# id_vars = what we do not wish to melt
# value_vars = what we watn to melt
melted = pd.melt(frame=data_new, id_vars = 'Name', value_vars = ['Attack','Defense'])
melted


# In[ ]:


# PIVOTING DATA
# Reverse of melting

# Index is name
# I want to make that columns are variable
# Finally values in columns are value

melted.pivot(index = 'Name', columns = 'variable', values = 'value')


# **CONCATENATING DATA**
# We can concatenate two dataframe

# In[ ]:


# Firstly lets create 2 data frame

data1 = data.head()
data2 = data.tail()
conc_data_row = pd.concat([data1,data2], axis = 0, ignore_index = True) # axis = 0 : adds dataframes in row
conc_data_row


# In[ ]:


data3 = data['Attack'].head()
data4 = data['Defense'].head()
conc_data_col = pd.concat([data3,data4], axis = 1) # axis = 1 : adds dataframes in column
conc_data_col


# **DATA TYPES**
# 
# There are 5 basic data types: object(string), boolean, integer, float and categorical.
# We can make conversion data types like from str to categorical or from int to float
# Why is category important?
# *  Make dataframe smaller in memory
# * can be utilized for analysis especially for sklearn

# In[ ]:


data.dtypes


# In[ ]:


# Lets convert object(str) to categorical and it to float
data['Type 1']= data['Type 1'].astype('category')
data['Speed'] = data['Speed'].astype('float')
data.dtypes


# **MISSING DATA and TESTING WITH ASSERT**
# If we encounter with missing data, what we can do:
# 
# * leave as is
# * drop them with dropna()
# * fill missing value with fillna()
# * fill missing values with test statistics like mean
# Assert statement : check that you can turn on or turn off when you are done with your testing of the program

# In[ ]:


data.info()
# Type 2 has 414 non-null object so it has 386 null object


# In[ ]:


data['Type 2'].value_counts(dropna = False)


# In[ ]:


data1 = data
data1['Type 2'].dropna(inplace = True)


# In[ ]:


assert data1['Type 2'].notnull().all # returns nothing because we drop nan values


# In[ ]:


data['Type 2'].fillna('empty',inplace = True)


# In[ ]:


assert data['Type 2'].notnull().all() # returns nothing because we don't have nan values

