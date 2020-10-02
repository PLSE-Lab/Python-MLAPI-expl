#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv("/kaggle/input/videogamesales/vgsales.csv")


# In[ ]:


data.info()


# We can see from here just the year and publisher columns have null values.

# In[ ]:


data.head()


# In[ ]:


data.tail()


# There are total 16600 video games.
# Firstly, let's fill the null values.

# In[ ]:


data.fillna("empty",inplace=True)


# In[ ]:


for each in data.columns:
    assert data["{}".format(each)].notnull().all 


# Let's have a statistic analysis now.

# In[ ]:


data.describe()


# In[ ]:


f,ax = plt.subplots(figsize=(20,15))
data.boxplot(column="Global_Sales",by = "Genre",ax=ax)


# We can understand from the boxplot that shooter and role-playing video games are the most popular genres. Also sports genre has an extreme outlier.

# In[ ]:


label = data.Platform.unique()
datasets = [data[data.Platform == each] for each in label]
count_per_platform = [sum(datasets[each]["Global_Sales"]) for each in range(len(label))]

f,ax = plt.subplots(figsize=(15,10))
plt.bar(label,count_per_platform)
plt.show()


# This is the distribution of global video game sales for each platform.

# In[ ]:




