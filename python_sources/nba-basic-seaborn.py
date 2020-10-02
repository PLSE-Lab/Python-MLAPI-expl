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


# **import matplotlib and seaborn modules**

# In[ ]:


from matplotlib import pyplot as plt
import seaborn as sns


# In[ ]:


df = pd.read_csv('../input/NBA_player_of_the_week.csv')
df.head() #shows first six rows....


# In[ ]:


df.info


# In[ ]:


df.describe()


# In[ ]:


print('df count is',df.count().sum())
print('draft year count is',df['Draft Year'].count())


# In[ ]:


df.isnull().sum()


# **barplot**

# In[ ]:


sns.barplot(x='Age',y='Real_value',data=df,)
plt.xticks(Rotation=90)


# **violin plot**

# In[ ]:


sns.violinplot(data=df['Age'])


# **scatter plot**

# In[ ]:


sns.set_style("whitegrid")
sns.scatterplot(x='Age',y='Draft Year',data=df)


# **count plot**

# In[ ]:


sns.countplot(x='Age',data=df)


# > thanks
# [https://github.com/ikunalvashisht]
