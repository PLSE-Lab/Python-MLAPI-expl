#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


dataset = pd.read_csv('../input/master.csv')


# In[3]:


dataset.head()


# In[4]:


dataset.hist(grid=True,figsize=(15,20))


# # What we know so far
# We can see from the histogram above, that the most suicides were commited in the year 2003/2002. What we also can see in this histogram is that it is very strange to see such a enormous growth of suicide from 1985 to the 200's. What also amazes me, is the fact that in the year 1990, the amount of suicides went down and within in 5 years it increased again. 
# 
# # GDP per capita
# We can see that the higher the GDP per capita, the lower the amount of suicide. If we look at the Human Development Index, we can see that there are more suicides at 0.73 HDI. So why could this be so?
# 
# ## What now?
# Now let's clean the data and go on with our investigation.

# In[5]:


#get all the countries in the dataset

dataset['country'].value_counts()


# # Countries
# We can see that certain countries do not occur that much. This gives me some question marks. Why does some countries have a higher suicide rate than other countries?
