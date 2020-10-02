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
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# **This notebook is a tutorial to exploratory data analysis and looks to find the relationship between mpg and different attributes**
# 
# It is structured as follows: 
# 1. Importing data
# 2. Previewing data
# 3. Basic info of MTcars dataset
# 4. Initial analysis
# 5. Using conditional selection to trim down rows
# 6. Summary statistics
# 7. Using Split/Apply/Combine to determine relationships between mpg and cyl/hp

# Importing Data

# In[ ]:


df = pd.read_csv("/kaggle/input/mtcars/mtcars.csv")


# Previewing Data

# In[ ]:


df.head(5) # Top 5 data


# In[ ]:


df.tail(5) # Bottom 5 data


# **Basic Info of MTCars dataset**

# Rows and column number of dataset

# In[ ]:


df.shape


# Column names

# In[ ]:


df.columns


# **Initial analysis of data to find insights**

# Testing out loc and index resetting by making model name into index and selecting Mazda RX4

# In[ ]:


df.set_index("model").loc["Mazda RX4"]


# Selecting rows 1,3,5,7 using iloc

# In[ ]:


df.iloc[[1,3,5,7]]


# Counting how many cars have X number of cylinders

# In[ ]:


df.cyl.value_counts()


# Determining which car has the highest mpg

# In[ ]:


df.loc[df.mpg == df.mpg.max()]


# Sorting cars based on mpg numbers (high to low)

# In[ ]:


df.sort_values("mpg", ascending = False)


# **Conditional Selection**

# Using boolean indexing to select all cars with 6 or more cylinders

# In[ ]:


df.loc[df.cyl >= 6]


# Using a query to find all cars with mpg less than or equal to 12

# In[ ]:


df.query("mpg <= 12")


# **Summary statistics**

# Determining mean mpg for all cars

# In[ ]:


df.mpg.mean()


# Determining median number of cylinders

# In[ ]:


df.cyl.median()


# Determining standard deviation of mpg for cars with median number of cylinders

# In[ ]:


df.loc[df.cyl == df.cyl.median()].mpg.std()


# **Split/Apply/Combine** to determine relationships between mpg and other factors

# Comparing mean mpg and cylinder count

# In[ ]:


df.groupby("cyl").mean().mpg.plot.bar()


# Looking at the mean averages of mpg, we can see that cars with more cylinders tend to have lower mpgs.

# Compare horsepower and mpg

# In[ ]:


df[["mpg","hp"]].plot.scatter(x = "hp", y = "mpg")


# From the graph above, we can see an inverse relationship between hp and mpg.
