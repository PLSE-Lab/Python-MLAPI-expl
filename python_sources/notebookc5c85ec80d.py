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

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# full dataframe
df = pd.read_json(open("../input/train.json", "r"))


# In[ ]:


# short data view
df.head()


# In[ ]:


# data shape
df.shape


# In[ ]:


# column names
print(df.columns)


# In[ ]:


# unique values per feature
print(len(df['bathrooms'].unique()))
print(len(df['bedrooms'].unique()))
print(len(df['building_id'].unique()))
print(len(df['interest_level'].unique()))
print(len(df['manager_id'].unique()))
print(len(df['price'].unique()))
print(len(df['street_address'].unique()))


# In[ ]:


# distribution of 'price' feature
import matplotlib.pyplot as plt
plt.hist(df['price'], bins='auto')  # plt.hist passes it's arguments to np.histogram
plt.title("Histogram with 'auto' bins")
plt.show()


# In[ ]:




