#!/usr/bin/env python
# coding: utf-8

# ![](https://datatable.readthedocs.io/en/latest/_static/py_datatable_logo.png)

# 

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


# # Introduction to DataTable

# Python package **datatable** was inspired from its counterpart R package data.table. It was developped with the aim to analyse BigData efficiently. Following is about  datatable package.(**Taken from datatable github page**)
# 
# The set of features that we want to implement with datatable is at least the following:
# 
# - Column-oriented data storage.
# 
# - Native-C implementation for all datatypes, including strings. Packages such as pandas and numpy already do that for numeric columns, but not for strings.
# 
# - Support for date-time and categorical types. Object type is also supported, but promotion into object discouraged.
# 
# - All types should support null values, with as little overhead as possible.
# 
# - Data should be stored on disk in the same format as in memory. This will allow us to memory-map data on disk and work on out-of-memory datasets transparently.
# 
# - Work with memory-mapped datasets to avoid loading into memory more data than necessary for each particular operation.
# 
# - Fast data reading from CSV and other formats.
# 
# - Multi-threaded data processing: time-consuming operations should attempt to utilize all cores for maximum efficiency.
# 
# - Efficient algorithms for sorting/grouping/joining.
# 
# - Expressive query syntax (similar to data.table).
# 
#  - LLVM-based lazy computation for complex queries (code generated, compiled and executed on-the-fly).
# 
# - LLVM-based user-defined functions.
# 
# - Minimal amount of data copying, copy-on-write semantics for shared data.
# 
# - Use "rowindex" views in filtering/sorting/grouping/joining operators to avoid unnecessary data copying.
# 
# - Interoperability with pandas / numpy / pure python: the users should have the ability to convert to another data-processing framework with ease.
# 
# - Restrictions: Python 3.5+, 64-bit systems only.

# # About data used in this kernel

# - Data has been taken from 
# https://www.kaggle.com/grikomsn/amazon-cell-phones-reviews
# 
# - Data is by Griko Nibras. Thanks to him for uploading this beutiful data set.

# # Installing Datatable

# Installation
# On MacOS systems installing datatable is as easy as
# 
# pip install datatable
# On Linux you can install a binary distribution as
# 
# # If you have Python 3.5
# pip install https://s3.amazonaws.com/h2o-release/datatable/stable/datatable-0.9.0/datatable-0.9.0-cp35-cp35m-linux_x86_64.whl
# 
# # If you have Python 3.6
# * pip install https://s3.amazonaws.com/h2o-release/datatable/stable/datatable-0.9.0/datatable-0.9.0-cp36-cp36m-linux_

# In[ ]:


get_ipython().system('pip install datatable')


# In[ ]:


import datatable as dt


# In[ ]:


path = "/kaggle/input/amazon-cell-phones-reviews/"


# In[ ]:


help(dt.fread)


# In[ ]:


reviews = dt.fread(path+"20190928-reviews.csv")
reviews.head(2)


# ### Note  :
# 
# - In output of head, color coding can be observed.
# - **Red** : String Data 
# - **Green** : Integer 
# - **Blue** : FLoating data 

# In[ ]:


type(reviews)


# ### It can be observed that, the method fread return datatable Frame object.

# In[ ]:


items = dt.fread(path+"20190928-items.csv")
items.head(2)


# # Some important Attributes of Frame class
# 
# - **names** : Return the column names as tuple
# - **nrows** : Number of rows 
# - **ncols** : Number of columns 
# - **ndims** : Number of dimentions
# - **stypes** : Storage type

# In[ ]:


reviews.names


# In[ ]:


reviews.nrows


# In[ ]:


reviews.ncols


# In[ ]:


reviews.ndims


# In[ ]:


reviews.stypes


# # Data indexing in DataTable
# 
# Data indexing can be done using [] bracket

# In[ ]:


reviews[0,0]


# In[ ]:


# The column label can be provided also 
reviews[0,"asin"]


# # Data slicing 

# In[ ]:


reviews[0:10,0:4]


# In[ ]:


reviews[0:10,'asin':'verified']


# # Data filtering 

# ### Filter out the data where rating is 3 

# In[ ]:


from datatable import *


# In[ ]:


rating3 = reviews[f.rating == 3,:]
rating3.head(5)


# # Data aggregation 

# ### Average rating conditioned on **verified** columns

# In[ ]:


avgRating = reviews[:,dt.mean(f.rating),dt.by(f.verified)]
avgRating


# ## Average rating by verified and customer (Conditioned on multiple columns)

# In[ ]:


avgRating1 = reviews[:,dt.mean(f.rating),dt.by(f.verified,f.asin)]
avgRating1


# ## Multiple aggregation on  rating by verified and customer (Conditioned on multiple columns)

# In[ ]:


avgRating2 = reviews[:,[dt.mean(f.rating),dt.sum(f.rating)],dt.by(f.verified,f.asin)]
avgRating2


# # Data joining in datatable 

# ### By default it perform left outer join. In future perhaps other type of join will be implemented 

# In[ ]:


items.key="asin"


# In[ ]:


data12 =  reviews[:,:,dt.join(items)]


# In[ ]:


data12.head(3)


# # Data Sorting

# In[ ]:


sortedData = reviews[:,:,dt.sort(f.rating)]
sortedData.head(3)


# # Sorting by rating in decreasing order

# In[ ]:


sortedData = reviews[:,:,dt.sort(-f.rating)]
sortedData.head(3)


# # Sorting by many columns

# In[ ]:


sortedData = reviews[:,:,dt.sort(-f.rating),dt.sort(f.verified)]
sortedData.head(3)


# # Hope you have liked it. Kindly upvote if you like to motivate writers :)

# # References 
# 
# -  https://github.com/h2oai/datatable
# - https://pypi.org/project/datatable/ 
# - https://datatable.readthedocs.io/en/latest/index.html

# In[ ]:




