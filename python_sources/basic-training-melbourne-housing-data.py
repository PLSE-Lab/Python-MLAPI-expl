#!/usr/bin/env python
# coding: utf-8

# # 1. Introduction
# 
# This notebook is an exercise for practice of very basic concepts regarding data processing, analysis, visualization etc.
# We are using the dataset with home prices in Melbourne, Australia, that is stored in file *melb_data.csv*.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # 2. Exploring data

# First of all, we need to import Pandas library to allow us to create a dataframe from the .csv file.

# In[ ]:


import pandas as pd
# The path below is based on the output in the previous code cell.
path = "../input/melbourne-housing-snapshot/melb_data.csv"
df = pd.read_csv(path)


# In order to verify that the dataframe creation was alright, we can check some attributes (properties) of the object df: shape, size, ndim and columns.
# * Shape: Number of lines and number of columns;
# * Size: Number of lines multiplied by number of columns;
# * ndim: Number of dimensions (axis);
# * columns: Number of columns.

# In[ ]:


print(df.shape)
print(df.size)
print(df.ndim)
print(len(df.columns))


# The 'shape' attribute informs that our dataframe has 21 columns and 13580 rows.
# The 'size' attribute value comes from multiplying the number of rows by the number of columns (285180 = 13580x21).
# The 'ndim' shows the number of dimensions of the dataframe: two dimensions, in the present case.
# And the 'columns' is used here just to confirm the number of columns. We can use it whenever we want to remember this number because it is important to consider which features we are going to use in our prediction model. The names of the column labels are the features' names. So, one must take into account in the first place what is the maximum number of features they can use, and this number is equal the number of columns the dataframe has.

# The attribute 'columns' itself shows the **names** of the columns, or the 'features', as is usual to refer to them in machine learning's jargon (lingo).

# In[ ]:


df.columns


# Another very used method for displaying basic information about the dataframe is the 'head' method.
# It shows the five first rows of the entire dataframe.

# In[ ]:


df.head()


# The following method ("describe") is perhaps one of the first to be used to perform some descriptive statistics.
# 
# Without any argument, "describe" shows statistics only for numeric features.

# In[ ]:


df.describe()


# Note that, in spite of having 21 columns in the dataframe, 'describe' without any argument just deals with 13 of them, and so statistical computations are performed only over numeric data.

# Using the argument *include="all"* in the 'describe' method, a statistical summary is provided for every feature (column).

# In[ ]:


df.describe(include="all")


# Note that, for example, the "Suburb" feature does not take part in the 'df.describe()' output, but it does in the 'df.describe(include="all")' output.
