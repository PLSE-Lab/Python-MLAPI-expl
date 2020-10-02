#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


master_data = pd.read_csv("../input/master.csv")


# In[3]:


master_data.head()


# In[4]:


master_data.shape


# In[5]:


master_data.describe()


# [](http://)

# In[6]:


master_data["country"].unique()


# In[7]:


jp_data = master_data.loc[master_data["country"]=="Japan"]


# In[8]:


jp_data.reset_index(drop=True,inplace=True)
jp_data.head()


# In[9]:


jp_data[" gdp_for_year ($) "] = jp_data[" gdp_for_year ($) "].str.replace(',', '')
jp_data = jp_data.rename(columns={" gdp_for_year ($) ": "gdp_for_year"})
jp_data['gdp_for_year'] = jp_data['gdp_for_year'].astype(np.int64)


# In[10]:


jp_data.groupby(["year"])


# In[11]:


gdp_data = jp_data.groupby(["year"]).gdp_for_year.mean()


# In[12]:


years = jp_data["year"].unique()
suicide_data = jp_data.groupby(["sex", "year"]).suicides_no.sum()


# In[13]:


plt.stackplot(years, [suicide_data["male"], suicide_data["female"]])


# In[22]:


fig, ax1 = plt.subplots()
ax1.stackplot(years, [suicide_data["male"], suicide_data["female"]], labels=['Male','Female'])
ax1.set_ylabel('Suicide Num', color='black')
ax2 = ax1.twinx()
ax2.plot(years, gdp_data, 'black')
ax1.legend(loc='upper left')
ax2.set_ylabel('GDP', color='black')
ax2.tick_params('y', colors='black')


# In[ ]:




