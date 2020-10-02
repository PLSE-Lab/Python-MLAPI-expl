#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# Generally real world dataset will require data preprocessing before data modeling or creating machine learning models. In Python, Pandas is most popular tool for data preprocessing. But there are other package which is useful in data preprocessing like : dplython, numpy and many more packages. In this kernel, we are going to discuss two Python packages for data preprocessing.
# 
# - **Dplython**
# - **plydata**

# # Dplyr
# 
# Every R programming user might have used package tidyverse. The package dplyr is an important member of package tidyverse. This package is used for data preprocessing. Using package dplyr, user can perform data filtering, column selection, data mutation and many tasks related to data preprocessing.

# # Dplython
# 
# Package dplython is dplyr for Python users. It provide infinite functionality for data preprocessing. Dplython datastructure **DplyFrame**, which has been written on Pandas **DataFrame**. With **DplyFrame**, every method defined on Pandas DataFrame, works. Following are some methods defined in Dplython package. 
# 
# - select : Select columns by their name.
# - arrange : Use to sort data.
# - dfilter : Data filtering can be done using dfilter.
# - mutate : Add new variables using function on existing columns.
# - summarize : Function to use summarizing data.

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


# ## Installing Dplython using pip

# In[ ]:


get_ipython().system('pip install dplython')


# In[ ]:


get_ipython().system('pip install plydata')


# ### We are going to discuss following in this tutorial using packages dplython and plydata
# 
# - Data column selection
# - Data Filtering
# - Data Sorting
# - Data Summarization
# 
# We are going to explore dataset diamonds. diamonds dataset is inbuilt in package dplython.

# ### Importing required functionality from dplython package

# In[ ]:


import dplython as dpl
from dplython import (DplyFrame, 
                      X, 
                      diamonds, 
                      dfilter,
                      select,
                      sift, 
                      sample_n,
                      sample_frac, 
                      head, 
                      arrange,
                      mutate,
                      nrow,
                      group_by,
                      summarize, 
                      DelayFunction) 


# You are going to be amazed that on dataset **diamonds**, head() function of R can be apply in python environment in R style as in following code cell. This function is not R function here, but provided by dplython package.

# In[ ]:


head(diamonds,3)


# #### What is datatype of diamonds ? 
# 
# Python provide a function type(), to get the class of an object. 

# In[ ]:


type(diamonds)


# #### How many data row is there in diamonds dataset?
# 
# Package dplython, consists of a function nrow(). Function nrow() will return total number of rows in a **DplyFrame**

# In[ ]:


nrow(diamonds)


# ### Pandas methods on DplyFrame objects.
# 
# Since DplyR has been written on Pandas DataFrame, hence, all the function of DataFrame are available with DplyFrame too. In following code cell, let us see that how Pandas Dataframe are working with DplyFrame object.

# ### Data Aggregation on DplyFrame using Pandas DataFrame methods
# 
# #### Calculate the price of diamonds where data is grouped on cut.

# In[ ]:


priceCutWise = diamonds.groupby("cut")["price"].mean()
priceCutWise


# ### Selecting some column 
# 
# ### Select column carat	cut	color	clarity	depth	table	price

# In[ ]:


diamondCol = select(diamonds,X.carat, X.cut,X.color,X.clarity,
                                X.depth,X.table,X.price)
diamondCol.head(5)


# Best part of using package **dplython** is that, the pipe operator **>>** can be used. What is pipe operator? Pipe operator can pipeline output of one operation to another operator in pipeline.  

# In[ ]:


diamondCol = diamonds >> select(X.carat, X.cut,X.color,X.clarity,
                                X.depth,X.table,X.price)
diamondCol.head(5)


# In[ ]:


diamonds.describe()


# ### Filtering data from diamonds DplyFrame
# #### Get the data where caret > 0.5 and price > 950

# In[ ]:


filteredData = dfilter(diamonds, X.carat > 0.5, X.price > 950)
filteredData.head()


# In[ ]:


# Filtering the data using pipe operators 
filteredData = diamonds >> dfilter( X.carat > 0.5, X.price > 950)
filteredData.head()


# ### Data sorting on diamonds DplyFrame
# #### Sort the diamond data increasing order on carat and cut

# In[ ]:


sortedData = arrange(diamonds, X.carat, X.cut)
sortedData.head()


# ### Summarizing the data
# 
# #### Get the mean value of price grouped on cut and color of diamond.
# 
# dplython provides group_by() function to group the data. Function group_by() makes method summarize() very powerful. You might be thinking that, usefulness of method summarize(). Method summarize() is used to summarize the data and provide summarized value of data like mean, median, variance, sum etc.

# In[ ]:


groupd = group_by(diamonds,X.cut,X.color)
summaryVal = summarize(groupd,meanval=X.price.mean())
summaryVal


# ### Filtering the data using shift function. 
# #### Get the data where caret > 0.5 and price > 950

# In[ ]:


filteredData = sift(diamonds, X.carat > 0.5, X.price > 950)
filteredData.head()


# # plydata package 
# 
# This is a library for data manupulation in python. It is also based on R programming dplyr package. In plydata, **>>** is used as pipe operator same as dplython. Using plydata package we are going to explore following operations on diamonds data. 
# 
# - Data column selection
# - Data Filtering
# - Data Sorting
# - Data Summarization
# 
# There are many functionality in plydata package. But we are going to use the following functions. 
# 
# - select
# - query
# - arrange
# - group_by
# - summarize

# In[ ]:


import plydata as pldt


# ### Selecting some column 
# 
# ### Select column carat	cut	color	clarity	depth	table	price

# In[ ]:


diamondCol = pldt.select(diamonds,"carat", "cut","color","clarity",
                                "depth","table","price")
diamondCol.head(5)


# ### Using pipe operator

# In[ ]:


diamondCol = diamonds >> pldt.select("carat", "cut","color","clarity",
                                "depth","table","price")
diamondCol.head(5)


# ### Filtering data from diamonds DplyFrame
# #### Get the data where caret > 0.5 and price > 950

# In[ ]:


filteredData = pldt.query(diamonds, "carat > 0.5 & price > 950")
filteredData.head()


# In[ ]:


# Filtering the data using pipe operators 
filteredData = diamonds >> pldt.query("carat > 0.5 & price > 950")
filteredData.head()


# ### Data sorting on diamonds DplyFrame
# #### Sort the diamond data increasing order on carat and cut
# 

# In[ ]:



sortedData = pldt.arrange(diamonds, "carat", "cut")
sortedData.head()


# ### Using pipe operator

# In[ ]:


sortedData = diamonds >> pldt.arrange( "carat", "cut")
sortedData.head()


# ### Summarizing the data
# 
# #### Get the mean value of price grouped on cut and color of diamond.
# 
# plydata provides group_by() function to group the data. Function group_by() makes method summarize() very powerful. You might be thinking that, usefulness of method summarize(). Method summarize() is used to summarize the data and provide summarized value of data like mean, median, variance, sum etc.

# In[ ]:


groupd = pldt.group_by(diamonds,"cut","color")
summaryVal = pldt.summarize(groupd,meanval="np.mean(price)")
summaryVal


# ### Data summarization using pipe

# In[ ]:


summaryVal = diamonds >> pldt.group_by("cut","color") >> pldt.summarize(meanval="np.mean(price)")
summaryVal


# # If you have enjoyed this kernel then kindly upvote it.

# ### Github link for dplython
# 
# https://github.com/dodger487/dplython
# 
# ### Github link for plydata
# 
# https://github.com/has2k1/plydata
