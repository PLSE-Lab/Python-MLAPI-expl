#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


df_usa=pd.read_csv('../input/USvideos.csv')


# In[6]:


df_usa.head()


# In[7]:


df_usa['trending_date'] = pd.to_datetime(df_usa['trending_date'], format='%y.%d.%m')
df_usa['publish_time'] = pd.to_datetime(df_usa['publish_time'], format='%Y-%m-%dT%H:%M:%S.%fZ')


# In[8]:


df_usa.insert(4, 'publish_date', df_usa['publish_time'].dt.date)
df_usa['publish_time'] = df_usa['publish_time'].dt.time
df_usa['publish_date']=pd.to_datetime(df_usa['publish_date'])


# In[9]:


df_usa.head()


# In[10]:


import seaborn as sns
plt.figure(figsize=(12,10))


# In[11]:


plt.figure(figsize=(12,10))
sns.heatmap(df_usa.corr())


# In[12]:


df_usa.columns


# In[13]:


sns.heatmap(df_usa[[u'views',
       u'likes', u'dislikes', u'comment_count']].corr())


# In[14]:


df_usa.head()


# In[15]:


df_usa_stat=df_usa[[u'video_id',u'views',
       u'likes', u'dislikes', u'comment_count']]


# In[16]:


df_usa.columns


# In[17]:


df_usa_stat.groupby('video_id').sum().sort_values(by='views',ascending=False)[:10].plot(kind='line')


# In[18]:


df_usa_stat.groupby('video_id')['views'].sum().plot(kind='line')


# In[19]:


df_usa_stat[]


# In[20]:


df_usa_stat.columns


# In[ ]:



# Scatter Plot
plt.scatter(df_usa_stat['views'], df_usa_stat['likes'],
            alpha=0.4, edgecolors='r')

plt.xlabel('views')
plt.ylabel('likes')
plt.title('US youtube views - likes',y=1.05)


# Joint Plot

jp = sns.jointplot(x='views', y='likes', data=df_usa_stat,
                   kind='reg', space=0, size=5, ratio=4)


# In[ ]:



# Scatter Plot
plt.scatter(df_usa_stat['comment_count'], df_usa_stat['dislikes'],
            alpha=0.4, edgecolors='r')

plt.xlabel('comment_count')
plt.ylabel('dislikes')
plt.title('US youtube comment_count - dislikes',y=1.05)


# Joint Plot
jp = sns.jointplot(x='comment_count', y='dislikes', data=df_usa_stat,
                   kind='reg', space=0, size=5, ratio=4)


# In[ ]:





# In[ ]:





# In[ ]:




