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
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# **Reading Data**

# In[ ]:


food = pd.read_csv("/kaggle/input/starbucks-nutrition-with-sugar-and-etc/starbacs_food.csv")
drink = pd.read_csv("/kaggle/input/starbucks-nutrition-with-sugar-and-etc/starbucks_drink.csv")


# ****Data Headers and Informations**

# In[ ]:


food.info()


# In[ ]:


food.head(5)


# In[ ]:


drink.info()


# In[ ]:


drink.head(5)


# There is no NULL field, that's why we don't need to apply data cleaning process.

# In[ ]:


plt.figure(figsize = (13,13))
plt.rcParams["font.size"] =22
food["Category"].value_counts().plot(kind="bar")


# In[ ]:


plt.figure(figsize = (13,13))
plt.rcParams["font.size"] =22
food["Category"].value_counts().plot(kind="pie")
print("Total: ", food["Category"].count())
food["Category"].value_counts()


# In[ ]:


plt.figure(figsize = (13,13))
plt.rcParams["font.size"] = 15
sns.heatmap(food.corr(), annot=True)


# In[ ]:


plt.figure(figsize = (13,13))
plt.rcParams["font.size"] =22
drink["Category"].value_counts().plot(kind="bar")


# In[ ]:


plt.figure(figsize = (13,13))
plt.rcParams["font.size"] =22
drink["Category"].value_counts()[:-2].plot(kind="pie")
print("Total: ", drink["Category"].count())
drink["Category"].value_counts()


# In[ ]:


plt.figure(figsize = (13,13))
plt.rcParams["font.size"] =15
sns.heatmap(drink.corr(), annot=True)

