#!/usr/bin/env python
# coding: utf-8

# **Data Science Tutorial For Everyone #3**
# 
# **Cleaning Data:**
# 
# * Diagnose data for cleaning
# * Exploratory data analysis
# * Visual exploratory data analysis
# * Tidy data
# * Pivoting data
# * Concatenating data
# * Data types
# * Missing data and testing with assert
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **DIAGNOSE DATA for CLEANING**
# 
# We need to diagnose and clean data before exploring. 
# Unclean data:
# 
# * Column name inconsistency like upper-lower case letter or space between words
# * missing data
# * different language
# 
# We will use head, tail, columns, shape and info methods to diagnose data

# In[ ]:


data = pd.read_csv('../input/pokemon.csv')
data.head()  # head shows first 5 rows


# In[ ]:


# tail shows last 5 rows
data.tail()


# In[ ]:


# columns gives column names of features
data.columns


# In[ ]:


# shape gives number of rows and columns in a tuble
data.shape


# In[ ]:


# info gives data type like dataframe, number of sample or row, number of feature or column, feature types and memory usage
data.info()


# **EXPLORATORY DATA ANALYSIS**
# 
# value_counts(): Frequency counts 
# outliers: the value that is considerably higher or lower from rest of the data
# 
# * Lets say value at 75% is Q3 and value at 25% is Q1.
# * Outlier are smaller than Q1 - 1.5(Q3-Q1) and bigger than Q3 + 1.5(Q3-Q1). (Q3-Q1) = IQR 
# * We will use describe() method. Describe method includes:
# * count: number of entries
# * mean: average of entries
# * std: standart deviation
# * min: minimum entry
# * 25%: first quantile
# * 50%: median or second quantile
# * 75%: third quantile
# * max: maximum entry
# 
# What is quantile?
# 
# * 1,4,5,6,8,9,11,12,13,14,15,16,17
# * The median is the number that is in middle of the sequence. In this case it would be 11.
# * The lower quartile is the median in between the smallest number and the median i.e. in between 1 and 11, which is 6.
# * The upper quartile, you find the median between the median and the largest number i.e. between 11 and 17, which will be 14 according to the question above.

# In[ ]:


# For example lets look frequency of pokemom types
print(data['Type 1'].value_counts(dropna =False))  # if there are nan values that also be counted
# As it can be seen below there are 112 water pokemon or 70 grass pokemon


# In[ ]:


# For example max HP is 255 or min defense is 5
data.describe() #ignore null entries


# **VISUAL EXPLORATORY DATA ANALYSIS**
# 
# * Box plots: visualize basic statistics like outliers, min/max or quantiles

# In[ ]:


# For example: compare attack of pokemons that are legendary  or not
# Black line at top is max
# Blue line at top is 75%
# Red line is median (50%)
# Blue line at bottom is 25%
# Black line at bottom is min
# There are no outliers
data.boxplot(column='Attack',by = 'Legendary')


# **TIDY DATA**
# 
# We tidy data with melt(). Describing melt is confusing. Therefore lets make example to understand it.

# In[ ]:


# Firstly I create new data from pokemons data to explain melt nore easily.
data_new = data.head()    # I only take 5 rows into new data
data_new


# In[ ]:


# lets melt
# id_vars = what we do not wish to melt
# value_vars = what we want to melt
melted = pd.melt(frame=data_new,id_vars = 'Name', value_vars= ['Attack','Defense'])
melted


# **PIVOTING DATA**
# 
# Reverse of melting.

# In[ ]:


# Index is name
# I want to make that columns are variable
# Finally values in columns are value
melted.pivot(index = 'Name', columns = 'variable',values='value')


# **CONCATENATING DATA**
# 
# We can concatenate two dataframe

# In[ ]:


# Firstly lets create 2 data frame
data1 = data.head()
data2= data.tail()
conc_data_row = pd.concat([data1,data2],axis =0,ignore_index =True) # axis = 0 : adds dataframes in row
conc_data_row


# In[ ]:


data1 = data['Attack'].head()
data2= data['Defense'].head()
conc_data_col = pd.concat([data1,data2],axis =1) # axis = 1 : adds dataframes in row
conc_data_col


# **DATA TYPES**
# 
# There are 5 basic data types: object(string),booleab, integer, float and categorical. 
# We can make conversion data types like from str to categorical or from int to float 
# Why is category important:
# 
# * make dataframe smaller in memory
# * can be utilized for anlaysis especially for sklear(we will learn later)
# 

# In[ ]:


data.dtypes


# In[ ]:


# lets convert object(str) to categorical and int to float.
data['Type 1'] = data['Type 1'].astype('category')
data['Speed'] = data['Speed'].astype('float')


# In[ ]:


# As you can see Type 1 is converted from object to categorical
# And Speed ,s converted from int to float
data.dtypes


# **MISSING DATA and TESTING WITH ASSERT**
# 
# If we encounter with missing data, what we can do:
# 
# * leave as is
# * drop them with dropna()
# * fill missing value with fillna()
# * fill missing values with test statistics like mean 
# * Assert statement: check that you can turn on or turn off when you are done with your testing of the program

# In[ ]:


# Lets look at does pokemon data have nan value
# As you can see there are 800 entries. However Type 2 has 414 non-null object so it has 386 null object.
data.info()


# In[ ]:


# Lets chech Type 2
# dropna = True : We drop non values
# We want to see nan values  
data["Type 2"].value_counts(dropna =False)
# As you can see, there are 386 NAN value


# In[ ]:


# Lets drop nan values
data1=data   # also we will use data to fill missing value so I assign it to data1 variable
data1["Type 2"].dropna(inplace = True)  # inplace = True means we do not assign it to new variable. Changes automatically assigned to data
# So does it work ?


# In[ ]:


#  Lets check with assert statement
# Assert statement:
assert 1==1 # return nothing because it is true


# In[ ]:


# In order to run all code, we need to make this line comment
# assert 1==2 # return error because it is false


# In[ ]:


assert  data['Type 2'].notnull().all() # returns nothing because we drop nan values
#  In 33, we dropped nan values and we are asking "Are there nan values?" here.


# In[ ]:


data["Type 2"].fillna('empty',inplace = True) 
# if there are nan value, we are filling with "empty"


# In[ ]:


assert  data['Type 2'].notnull().all() # returns nothing because we do not have nan values


# In[ ]:


# # With assert statement we can check a lot of thing. For example
# assert data.columns[1] == 'Name'


# **In this part, you learn:**
# 
# * Diagnose data for cleaning
# * Exploratory data analysis
# * Visual exploratory data analysis
# * Tidy data
# * Pivoting data
# * Concatenating data
# * Data types
# * Missing data and testing with assert
# 
