#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import os
import pandas as pd
import sys
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set(color_codes=True)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('../input/flavors_of_cacao.csv')


# In[ ]:


df.head()


# In[ ]:


df.isnull().sum(axis=0)


# In[ ]:


df.dtypes


# # Visualization

# In[ ]:


sns.distplot(df['REF'])


# In[ ]:


sns.distplot(df['Review\nDate'])


# In[ ]:


sns.distplot(df['Rating'])


# In[ ]:


#scatter plot 
var = 'REF'
data = pd.concat([df['Rating'], df[var]], axis=1)
data.plot.scatter(x=var, y='Rating');


# In[ ]:


#scatter plot 
var = 'Review\nDate'
data = pd.concat([df['Rating'], df[var]], axis=1)
data.plot.scatter(x=var, y='Rating');


# In[ ]:


df['Company\nLocation'].value_counts()


# In[ ]:


df['Company\nLocation'].value_counts().head(10).plot.bar()


# Seems like U.S.A. consumes chocolate far more than any other country of the world.

# In[ ]:


df['Cocoa\nPercent'].value_counts()


# In[ ]:


df['Cocoa\nPercent'].value_counts().head(10).plot.bar()


# Most of the chocolates have 70% cocoa. Let's see which chocolate has highest cocoa.

# In[ ]:


df[df['Cocoa\nPercent'] == '100%']


# In[ ]:


df['Rating'].value_counts().sort_index().plot.bar()


# More ratings means more better chocolate. Most of the chocolates got 3.5 rating but very few chocolate got full ratings. Let's see them.

# In[ ]:


df[df['Rating'] == 5.0]


# **Chuao** and **Toscano Black** from *Italy* got full ratings. 

# In[ ]:




