#!/usr/bin/env python
# coding: utf-8

# # Day1 

# In[17]:


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


# This data is from https://www.kaggle.com/dorbicycle/world-foodfeed-production
# 
# Sometimes we need specify a encode used in the file.

# In[18]:


data = pd.read_csv("../input/FAO.csv", encoding='latin1') 


# To see the how many records and how many fields in the data, using shape.  

# In[19]:


data.shape


# To check the fields:

# In[20]:


data.columns


# See how the records look like:

# In[21]:


data.head()


# See the summarization of statistics:

# In[22]:


data.describe()


# In[16]:


data["Item"].value_counts()


# # Day 2

# In[24]:


import matplotlib.pyplot as plt


# In[25]:


data_to_plot = data["Y2004"]


# In[26]:


data_to_plot.head()


# Find how to use the function.

# In[27]:


help(plt.hist)


# The following plot results error.
# The reason we get this error is because the data missing values. 
# So let's investigate the situation of missing values in next session.

# In[29]:


plt.hist(data_to_plot)


# ## Missing Values

# In[30]:


data.isnull().any(axis=0)


# We see Y2013 doesn't miss values.
# So try to plot using Y2013

# In[36]:


data_to_plot = data["Y2013"]


# In[37]:


plt.hist(data_to_plot)


# Only one bar!
# This may be the data mainly distribute near zero, but a few of them take a huge value.
# So we may just see the situation near zero by specify a range of the plot.

# In[46]:


plt.hist(data_to_plot, range=[0, 100])
plt.title("Range 0-100")


# In[43]:


plt.hist(data_to_plot, range=[10000, 20000])


# ## More on missing values

# If you want to know how many values are missing in each field. Do the following.

# In[44]:


data.isnull().sum()


# In[45]:


help(data.isnull)


# In[ ]:




