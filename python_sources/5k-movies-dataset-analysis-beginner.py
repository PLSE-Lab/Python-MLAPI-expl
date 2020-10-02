#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


movies=pd.read_csv("../input/tmdb_5000_movies.csv")
credits=pd.read_csv("../input/tmdb_5000_credits.csv")


# Let's get summary information about data

# In[ ]:


movies.info()
movies.describe()


# In[ ]:


movies.head()


# In[ ]:


movies.columns


# In[ ]:


movies.corr()


# In[ ]:


f,ax=plt.subplots(figsize=(10,10))
sns.heatmap(movies.corr(),annot=True, linewidths=.5, ax=ax)
plt.show()


# In[ ]:


movies.budget.plot(label="Budget",grid=True,figsize=(12,9),alpha=0.5, color="r",linestyle="-.")
movies.revenue.plot(label="Revenue", alpha=0.5, color="g",linestyle=":")
plt.title("Line Plot - Budget and Revenue")
plt.legend()
plt.xlabel("id")
plt.ylabel("Price")
plt.show()


# In[ ]:


movies.plot(kind="scatter",x="vote_average", y="runtime", alpha=.5,figsize=(12,9),label="Scatter Plot - Runtime and Vote Average")
plt.show()


# In[ ]:


movies.vote_average.plot(kind="hist",bins=30 , figsize=(12,9))
plt.xlabel("Average Vote")
plt.show()


# In[ ]:


high=movies["revenue"]>1200000000
movies[high]


# In[ ]:


movies[(movies["revenue"]>1200000000)&(movies["budget"]<200000000)]


# In[ ]:


for index, value in movies[["budget"]][0:6].iterrows():
    print(index," ", value)


# to be continued...
