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


# In[3]:


import matplotlib.pyplot as plt


# In[4]:


import seaborn as sns
import cufflinks as cf
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
cf.go_offline()


# In[12]:


df = pd.read_csv("../input/911.csv")


# In[13]:


df.info()


# In[14]:


df['zip'].value_counts().head(5)


# In[15]:


df['twp'].value_counts().head()


# In[16]:


df['title'].nunique()


# In[17]:


df['Reason'] = df['title'].apply(lambda x: x.split(':')[0])
df['Reason'].head()


# In[18]:


df['Reason'].value_counts()


# In[19]:


plt.style.use('ggplot')


# In[20]:


df['Reason'].value_counts().iplot(kind='bar')


# In[21]:


type(df['timeStamp'].iloc[0])


# In[22]:


df['timeStamp'] = pd.to_datetime(df['timeStamp'])


# In[23]:


df['Hour'] = df['timeStamp'].apply(lambda x: x.hour)
df['Month'] = df['timeStamp'].apply(lambda x: x.month)
df['Day of Week'] = df['timeStamp'].apply(lambda x: x.dayofweek)


# In[24]:


dmap= {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}


# In[25]:


df['Day of Week']= df['Day of Week'].map(dmap)


# In[26]:


plt.figure(figsize=(12,8))
sns.countplot(x='Day of Week',data=df,hue='Reason')
plt.legend(loc=[0,1])
plt.title('Day wise count plot for different reasons')


# In[27]:


plt.figure(figsize=(12,8))
sns.countplot(x='Month',data=df,hue='Reason')
plt.legend(loc=[0,1])


# In[28]:


byMonth = df.groupby('Month').count()


# In[29]:


byMonth['Mon'] = (['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
byMon=byMonth.set_index('Mon')
byMon


# In[30]:


byMon['twp'].iplot(title='Variation according to Month', xTitle='Month',yTitle='Count',colors='red',width=2)


# In[31]:


byMonth.reset_index(inplace=True)
sns.lmplot(y='twp',x='Month',data=byMonth)


# In[32]:


df['Date'] = df['timeStamp'].apply(lambda x : x.date())


# In[35]:


plt.figure(figsize=(12,8))
df.groupby('Date').count()['twp'].iplot()


# In[36]:


df[df['Reason']=='EMS'].groupby('Date').count()['twp'].iplot(title='EMS')


# In[37]:


df[df['Reason']=='Traffic'].groupby('Date').count()['twp'].iplot(title='Traffic')


# In[38]:


df[df['Reason']=='Fire'].groupby('Date').count()['twp'].iplot(title='Fire')


# In[39]:


new=df.groupby(by=['Day of Week','Hour']).count()['Reason'].unstack()
new


# In[40]:


new.iplot(kind='heatmap',xTitle='Days of Week',yTitle="Hour")


# In[41]:


sns.clustermap(new,cmap='viridis')


# In[ ]:




