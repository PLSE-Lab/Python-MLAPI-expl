#!/usr/bin/env python
# coding: utf-8

# **COLUMNS of Google Play Store Apps**
# 
# *These are the details of the applications on Google Play. There are 13 features describe a given app.. 
# *
# * **App** Application name
# * **Category** Category the app belongs to
# * **Rating** Overall user rating of the app (as when scraped)
# * **Reviews** Number of user reviews for the app (as when scraped)
# * **Size** Size of the app (as when scraped)
# * **Installs** Number of user downloads/installs for the app (as when scraped)
# * **Type** Paid or Free
# * **Price** Price of the app (as when scraped)
# * **Content** Rating Age group the app is targeted at - Children / Mature 21+ / Adult
# * **Genres** An app can belong to multiple genres (apart from its main category). For eg, a musical family game will belong to Music, Game, Family genres.
# * **Last Updated** Date when the app was last updated on Play Store (as when scraped)
# * **Current Ver** Current version of the app available on Play Store (as when scraped)
# * **Android Ver** Min required Android version (as when scraped)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import seaborn as sns  # visualization tool
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


play = pd.read_csv('../input/googleplaystore.csv')


# In[ ]:


play.info()


# Let's learn the memory usage of dataset that we are using **googleplaystore.csv**

# In[ ]:


play.memory_usage(deep=True)


# In[ ]:


# Sum of memory usage
play.memory_usage(deep=True).sum()


# In[ ]:


sorted(play.Genres.unique())


# In[ ]:


play.head()


# In[ ]:


play.tail()


# In[ ]:


play.sample(10)


# In[ ]:


play.shape


# In[ ]:


# Checking how many null datas in dataset
play.isnull().sum()


# In[ ]:


play.columns


# Let's change the column name

# In[ ]:


play.columns = play.columns.str.replace(" ", "_")
play.head()


# In[ ]:


play.dtypes


# Let's change the display option

# In[ ]:


pd.set_option('display.max_colwidth', 1000)


# Now look at the **App Column**.
# 
# The **(...)** is gone

# In[ ]:


play.head()


# In[ ]:


play[play["Type"] == "Free"][['App' , 'Category' , 'Reviews' , 'Size', 'Installs', 'Genres']].set_index("Category").head()


# In[ ]:


play.Category.value_counts()


# In[ ]:


play.Category.unique()


# In[ ]:


# Analyzing the Category
play.Category.value_counts().plot(kind='bar')
plt.title("Category counts")
plt.xlabel("Categories")
plt.ylabel("Count")


# In[ ]:


play.Type.value_counts()


# In[ ]:


# Analyzing the Type
play.Type.value_counts().plot(kind='bar')
plt.title("Type counts")
plt.xlabel("Types")
plt.ylabel("Count")


# In[ ]:


# Analyzing the Type as Horizontal
play.Type.value_counts().plot(kind='barh')
plt.title("Type counts")
plt.xlabel("Count")
plt.ylabel("Types")


# In[ ]:


# Analyzing the Type
play.Type.value_counts().plot(kind='pie')
plt.title("Pie chart of Types")


# In[ ]:


# Analyzing the Type
play.Type.value_counts().plot(kind='box')
plt.title("Pie chart of Types")
plt.show()


# In[ ]:


play.Installs.value_counts()


# In[ ]:


# Analyzing the Installed Apps
play.Installs.value_counts().plot(kind='bar')
plt.title("Installed App counts")
plt.xlabel("How many installed apps")
plt.ylabel("Count")

