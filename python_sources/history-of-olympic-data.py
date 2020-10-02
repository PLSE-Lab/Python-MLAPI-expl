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


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


events = pd.read_csv('../input/athlete_events.csv')
events.shape


# In[ ]:


events.head()


# In[ ]:


region = pd.read_csv('../input/noc_regions.csv')
region.shape


# In[ ]:


region.head()


# In[ ]:


events.isnull().sum()


# In[ ]:


events.Sex.value_counts()


# In[ ]:


events.Year.value_counts()


# In[ ]:


plt.figure(figsize=(20,5))
sns.pointplot('Year',y = events.ID.index.unique(),hue = 'Sex',data=events,dodge= True)
plt.show()


# In[ ]:


plt.figure(figsize=(20,5))
events.Year.value_counts().plot(kind = 'bar')


# more in pipe line, if you like it please upvote for me.
# 
# Thank you : )
