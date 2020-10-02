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


# Here we read a .csv (comma seperate values) file into a pandas DataFrame called `data`.
# 
# And display the first 5 rows using the `.head()` method.
# 

# In[ ]:


data = pd.read_csv("../input/heart.csv")
data.head()


# **Q1** Find out the number of rows and columns in the  DataFrame using the `.shape` method.

# In[ ]:


# Your code here


# The answer should be (303, 14)

# **Q2** Display column names of the DataFrame

# In[ ]:


# Your code here


# We use the `.info()` method to understand what datatypes each column in the DataFrame is as well as if there Null/Missing values in the dataset.

# In[ ]:


print(data.info())


# **Q3** Write a line of code that tells you how many male and female patients are present

# In[ ]:


# Your code here


# 207 males and 96 females

# **Q4** Display the 5 maximum `thalach` using `sort_values()`

# In[ ]:


# Your code here


# **Q5** What is the average `trestbps` of a patient with a Heart Disease (i.e, with a `target == 1`)?
# 

# In[ ]:


# Your code here


# **Q6** What is the max cholestorol where `fbs == 1` and `cp == 0`?

# In[ ]:


# Your code here


# In[ ]:


df1, df2 = data[:100], data[100:]
print("df1.shape:", df1.shape)
print("df2.shape:", df2.shape)


# **Q7** concatenate the two DataFrames `df1` and `df2`

# In[ ]:


# Your code here


# **Q8** Display histogram for the column `age`

# In[ ]:


# Your code here


# In[ ]:


import seaborn as sns


# **Q9** Draw a countplot using the `seaborn as sns` module.

# In[ ]:


# Your code here


# **Q10** Display the correlation matrix for the DataFrame `data`

# In[ ]:


# Your code here

