#!/usr/bin/env python
# coding: utf-8

# In[46]:


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


# In[47]:


times_df = pd.read_csv("../input/timesData.csv")


# In[48]:


times_df.columns


# In[49]:


col = ['world_rank', 'teaching', 'international',
       'research', 'citations', 'income', 'total_score', 'num_students',
       'student_staff_ratio', 'international_students',
       'year']
times_df = times_df[col]
len(times_df)


# In[50]:


times_df.head()


# In[63]:


times_df = times_df[~times_df['income'].isin(['-'])]

times_df.head(7)


# In[64]:


len(times_df)


# In[75]:


times_df = times_df.dropna()
len(times_df)


# In[76]:


import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(1,1, figsize=(10,5))
corr = times_df.corr()
sns.heatmap(corr, ax=ax)
ax.set_title("correlation")

plt.show()


# In[ ]:




