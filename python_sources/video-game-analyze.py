#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ### This kernel is still developing . . .

# In[ ]:


df = pd.read_csv('../input/vgsales.csv')
df.info()


# - We read the data with using `read_csv` and defined a data frame name in `df`.
# - Also, we got the general information about the data with `info()` function.
# - There are some missing-values in our data set.
# - There are 6 float, 1 integer and 4 string features in our data set.

# In[ ]:


df.columns


# - We got the columns name.

# In[ ]:


df.head(10) # The first 10 lines in our data set to look at quickly.


# In[ ]:


df.tail(10) # The last 10 lines are shown


# In[ ]:


df.dtypes  # Shows the types of columns.


# In[ ]:


df.describe()  # We'll get some of mathematical and statical results with this function.


# In[ ]:


df.corr()  # Correlation between the features.


# - There is a high correlation between the North America & Europe and global sales. (+%90)

# In[ ]:


# HeatMap

f, ax = plt.subplots(figsize=(18, 18))
sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt='.2f', ax=ax)


# - White areas show that correlation between the features is high.

# In[ ]:


# Scatter plot
# x = year y = Japan Sales

df.plot(kind='scatter', x='Year', y='JP_Sales', alpha=0.5, color='blue', grid=True)
plt.xlabel("Year")
plt.ylabel("Japan Sales")
plt.title("Year - Japan Sales Plot")
plt.show()


# In[ ]:


# Histogram plot

df.Year.plot(kind='hist', bins=100, figsize=(12, 12))
plt.show()


# In[ ]:




