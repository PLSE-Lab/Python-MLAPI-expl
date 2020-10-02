#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#I show distinct plots where we can see the relationship between features and how it works the Rank made by 
#socialblade

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.

import os
df = pd.read_csv("../input/data.csv")


# In[ ]:


df.head()


# In[ ]:


df['Subscribers'] = pd.to_numeric(df['Subscribers'], errors='coerce')
df['Video Uploads'] = pd.to_numeric(df['Video Uploads'], errors='coerce')


# In[ ]:


df.info()


# In[ ]:


sns.jointplot(data=df, x='Subscribers', y='Video views')


# Let's do a top 100 channels analysis!!!
# 

# In[ ]:


top_100df = df[0:100]


# In[ ]:


sns.jointplot(data=df, y='Subscribers', x='Video Uploads')


# In[ ]:


sns.jointplot(data=df, x='Video Uploads', y='Video views')


# There is not a proportional relationship between Video Uploads either Video views or Subscribers!!!

# In[ ]:


sns.lmplot(data=top_100df,x='Subscribers', y='Video views', hue='Grade', palette='Set1')
plt.title('Top 100 Rank Channels')


# They did a good evaluation of the channels, the grade shows a good distinction between the most seen channels!!!

# In[ ]:


df.corr()


# In[ ]:


sns.heatmap(df.corr(), cmap='viridis')
plt.tight_layout()


# In[ ]:




