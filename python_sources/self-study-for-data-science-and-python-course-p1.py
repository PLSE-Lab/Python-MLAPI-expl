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
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# <h2>**1.Introduction**</h2>
# 
# For self-study homework I've chosen Black Friday dataset. 

# In[ ]:


data = pd.read_csv("../input/BlackFriday.csv")


# This data set has 537,577 entries and 12 features. 

# In[ ]:


data.info()


# In[ ]:


data.sort_values("Purchase").head()


# In[ ]:


data.sort_values("Purchase").tail()


# In this dataset there are 5,891 unique Users. 

# In[ ]:


len(data.User_ID.unique())


# In[ ]:


data.Age.unique()


# In[ ]:


data.describe(include=['object'])


# In[ ]:


data.describe()


# In[ ]:


f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.2f',ax=ax)
plt.show()


# In[ ]:


with sns.axes_style(style=None):
    sns.violinplot("Age", "Purchase", hue="Gender", data=data,
                   split=True, inner="quartile",
                   palette=["lightblue", "lightpink"]);


# In[ ]:


data["Gender_Age"] = data.Gender + " / " + data.Age


# In[ ]:


data.head()


# In[ ]:


sns.boxplot(x=data["Gender_Age"], y=data["Purchase"])
sns.set(rc={'figure.figsize':(15,5)})
plt.show()


# In[ ]:


grid = sns.FacetGrid(data, row="Gender", col="Age", margin_titles=True)
grid.map(plt.hist, "Purchase", color="pink", density=True);
plt.show()


# In[ ]:


with sns.axes_style(style='ticks'):
    g = sns.factorplot("Age", "Purchase", "Gender", data=data, height=5, aspect=2)
    g.set_axis_labels("Age", "Purchase");

