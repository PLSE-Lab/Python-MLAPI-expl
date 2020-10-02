#!/usr/bin/env python
# coding: utf-8

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


# ![](https://i0.wp.com/everydaywanderer.com/wp-content/uploads/2017/11/panda-206297_1280.jpg?fit=1280%2C851&ssl=1)

# **Introduction:**
# 
# Hello everyone, here I'm gonna walk you through basic data preprocessing and Exploratory Data Analysis(EDA) on Google PlayStore Dataset. 
# 
# Lets have some fun.
# 

# **Data Exploration: ** 
# 

# In[ ]:


#Reading the CSV file ( googleplaystore dataset ) into a DataFrame.

df1 = pd.read_csv("../input/googleplaystore.csv")


# In[ ]:


#Representing the dimensionality of the DataFrame.

#Tuple represents the number of records and fields in our dataset.

df1.shape


# In[ ]:


#Returns the column names.

df1.columns


# In[ ]:


#Show first 5 records of the dataset.

#Specifying n values returns first n rows.

df1.head()


# In[ ]:


#Show last 5 records of the dataset.

#Specifying n values returns last n rows.

df1.tail()


# In[ ]:


#Returns summary of a DataFrame including the index dtype and column dtypes, non-null values and memory usage.

df1.info()


# In[ ]:


#Returns unique values across a series.

df1.Category.unique()


# In[ ]:


#Returns the maximum of the values in the object.

df1.Rating.max()


# In[ ]:


#Returns the maximum of the values in the object.

df1.Rating.min()


# In[ ]:


#Returns data type of each field in the dataset.

df1.dtypes


# In[ ]:


#Return the sum of the values.

df1.sum()


# In[ ]:


#Groups series of columns using a mapper

df2 = df1.groupby(['Category']).count()

print(df2)

