#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('../input/athlete_events.csv')


# In[ ]:


# Head shows first 5 rows
data.head()


# In[ ]:


# Head shows last 5 rows
data.tail()


# In[ ]:


# Columns gives column names of features
data.columns


# In[ ]:


# Shape gives number of rows and columns in a tuble
data.shape


# In[ ]:


# Info gives data type like dataframe, number of sample or row, number of feature or column, feature types and memory usage
data.info()


# In[ ]:


# For example lets look frequency of Team types
print(data["Team"].value_counts(dropna = False))


# In[ ]:


# Describe show the statistics features
data.describe()


# In[ ]:


data.boxplot(column = 'Height', by = 'Age') 


# In[ ]:


# Show the data types
data.dtypes


# In[ ]:


# Information about the data
data.info()


# In[ ]:


# Lets check Weight for NaN values
data["Weight"].value_counts(dropna = False)


# In[ ]:


# Lets drop nan values
data_1 = data
data_1["Weight"].dropna(inplace = True)


# In[ ]:



assert data_1["Weight"].notnull().all()


# In[ ]:


data["Weight"].fillna('empty',inplace = True)


# In[ ]:


assert data["Weight"].notnull().all()


# In[ ]:


data["Weight"].value_counts(dropna = False)


# In[ ]:




