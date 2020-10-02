#!/usr/bin/env python
# coding: utf-8

# YOUTUBE DATA ANALYSIS

# In[ ]:


import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('../input/data.csv')


# In[ ]:


df.head()


# In[ ]:


df[df['Video Uploads']=='--']


# In[ ]:


import numpy as np
t = df.replace('--', np.nan)


# In[ ]:


t['Grade'] = t['Grade'].astype('category')
t['Grade']=t['Grade'].cat.codes


# In[ ]:


t=t.drop(columns="Channel name")


# In[ ]:


t['Video Uploads']=t['Video Uploads'].fillna(0)


# In[ ]:


t['Video Uploads']= t['Video Uploads'].astype('uint64')


# In[ ]:


t['Subscribers']=[x.replace('--','0') for x in t['Subscribers']]


# In[ ]:


t['Subscribers']=t['Subscribers'].astype('uint64')


# In[ ]:


for i in range(1,5001):
    t.iloc[i-1,0]=i


# In[ ]:


t.info()


# In[ ]:


t.hist(figsize=(10,10))


# In[ ]:


t['Grade'].value_counts().plot(kind='bar')


# In[ ]:


t['Grade'].value_counts().plot(kind='pie',figsize=(15,15))


# In[ ]:


t['Video views'].plot(kind='box')


# In[ ]:


pd.plotting.scatter_matrix(t,figsize=(10,10))


# In[ ]:


import seaborn as sns
sns.heatmap(t.corr())


# There is corelation between subscribers and video views.
# Rank and Grade are dependent.
# The rank of the channel increases if the video views,subscribers,video uploads increases.
