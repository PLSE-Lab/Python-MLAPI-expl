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


# 3.CLEANING DATA
# 
# 
# DIAGNOSE DATA for CLEANING
# 
# We need to diagnose and clean data before exploring.
# Unclean data:
# 
#    - Column name inconsistency like upper-lower case letter or space between words
#     -missing data
#     -different language
# 
# 
# We will use head, tail, columns, shape and info methods to diagnose data
# 

# In[ ]:


data = pd.read_csv("../input/anime.csv")
data.head() # head show first 5 row


# In[ ]:


data.tail() # tail show last 5 row


# In[ ]:


# columns gives column names of features
data.columns


# In[ ]:


# shape gives number of rows and columns in a tuble
data.shape


# In[ ]:


# info gives data type like dataframe, number of sample or row, number of feature or column, feature types and memory usage
data.info()


# EXPLORATORY DATA ANALYSIS
# 
# value_counts(): Frequency counts
# outliers: the value that is considerably higher or lower from rest of the data
# 
#    - Lets say value at 75% is Q3 and value at 25% is Q1.
#     -Outlier are smaller than Q1 - 1.5(Q3-Q1) and bigger than Q3 + 1.5(Q3-Q1). (Q3-Q1) = IQR
#     -We will use describe() method. Describe method includes:
#     -count: number of entries
#     -mean: average of entries
#     -std: standart deviation
#     -min: minimum entry
#     -25%: first quantile
#     -50%: median or second quantile
#     -75%: third quantile
#     -max: maximum entry
# 
# 
# What is quantile?
# 
# *     1,4,5,6,8,9,11,12,13,14,15,16,17
# * 
# *     The median is the number that is in middle of the sequence. In this case it would be 11.
# * 
# *     The lower quartile is the median in between the smallest number and the median i.e. in between 1 and 11, which is 6.
# *     The upper quartile, you find the median between the median and the largest number i.e. between 11 and 17, which will be 14 according to the question above.
# 

# In[ ]:


# For example lets look frequency of anime genre
data.genre.value_counts(dropna=False) # if there are nan values that also be counted


# In[ ]:


data.describe() #ignore null entries


# **CONCATENATING DATA**
# 
# We can concatenate two dataframe

# In[ ]:


# This codes skipping
filter1 = data.type == "Movie"
filter2 = data.type == "TV"
data1 = data[filter1]
data2 = data [filter2]
ver_concatdata =pd.concat([data1,data2],axis=0,ignore_index=True) # axis = 0 vertial concatenating
ver_concatdata


# In[ ]:


hor_concatdata = pd.concat([data1,data2],axis = 1) # axis = 1 horizontal concatenating
hor_concatdata


# VISUAL EXPLORATORY DATA ANALYSIS
# 
#     Box plots: visualize basic statistics like outliers, min/max or quantiles

# In[ ]:


# For example: compare attack of pokemons that are legendary  or not
# Black line at top is max
# Blue line at top is 75%
# Red line is median (50%)
# Blue line at bottom is 25%
# Black line at bottom is min
# There are no outliers
cdata.boxplot(column='members',by ="type",figsize = (24,12))


# 
# **TIDY DATA**
# 
# We tidy data with melt(). Describing melt is confusing. Therefore lets make example to understand it.
# 

# In[ ]:


# Firstly I create new data from anime data to explain melt nore easily.
data_new = data.head(10)   # I only take 10 rows into new data
data_new


# In[ ]:


# lets melt
# id_vars = what we do not wish to melt
# value_vars = what we want to melt
melted = pd.melt(frame=data_new,id_vars = 'name', value_vars= ['type','episodes'])
melted


# PIVOTING DATA
# 
# Reverse of melting.

# In[ ]:


melted.pivot(index = "name",columns="variable",values="value")


# DATA TYPES
# 
# There are 5 basic data types: object(string),booleab, integer, float and categorical.
# We can make conversion data types like from str to categorical or from int to float
# Why is category important:
# 
# *     make dataframe smaller in memory
# *     can be utilized for anlaysis especially for sklear(we will learn later)
# 

# In[ ]:


data.dtypes


# In[ ]:




# lets convert object(str) to categorical and int to float.
data.type = data.type.astype('category')
data.members = data.members.astype("float")


# In[ ]:


# As you can see Type 1 is converted from object to categorical
# And Speed ,s converted from int to float
data.dtypes


# MISSING DATA and TESTING WITH ASSERT
# 
# If we encounter with missing data, what we can do:
# * 
# *     leave as is
# *     drop them with dropna()
# *     fill missing value with fillna()
# *     fill missing values with test statistics like mean
# *     Assert statement: check that you can turn on or turn off when you are done with your testing of the program
# 

# In[ ]:




# Lets look at does pokemon data have nan value
# As you can see there are 800 entries. However Type 2 has 414 non-null object so it has 386 null object.
data.info()


# In[ ]:




# Lets chech Type 2
data.genre.value_counts(dropna =False)
# As you can see, there are 62 NAN value


# In[ ]:


# Lets drop nan values
datax=data   # also we will use data to fill missing value so I assign it to data1 variable
datax.genre.dropna(inplace = True)  # inplace = True means we do not assign it to new variable. Changes automatically assigned to data
# So does it work ?


# In[ ]:


#  Lets check with assert statement
# Assert statement:
assert 1==1 # return nothing because it is true


# In[ ]:


assert  datax.genre.notnull().all() # returns nothing because we drop nan values


# In[ ]:




datax.genre.fillna('empty',inplace = True)


# In[ ]:


data.type.value_counts(dropna=False)


# In[ ]:


#data.type.fillna("empty",inplace = True)# give error. Because category feature is not working this metod
data.type = data.type.astype("object")# well we convert category to object
data.type.fillna('empty',inplace = True)# 


# In[ ]:


data.type.value_counts(dropna=False)


# In[ ]:


assert  data.type.notnull().all()# returns nothing because we do not have nan values


# 
# In this part, you learn:
# 
# *     Diagnose data for cleaning
# *     Exploratory data analysis
# *     Visual exploratory data analysis
# *     Tidy data
# *     Pivoting data
# *     Concatenating data
# *     Data types
# *     Missing data and testing with assert
