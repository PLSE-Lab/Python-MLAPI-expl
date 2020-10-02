#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Import Libraries

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


Import Data Set


# In[ ]:


df=pd.read_csv('../input/montcoalert/911.csv')


# In[ ]:


df.info()


# In[ ]:


df.head()


# In[ ]:


df.tail()


# In[ ]:


df['zip'].value_counts().head(5)


# In[ ]:


df['twp'].value_counts().head(5)


# In[ ]:


df['title'].nunique()


# In[ ]:


x = df['title'].iloc[0]


# In[ ]:


x.split(':')[0]


# In[ ]:


df['Reason']= df['title'].apply(lambda title: title.split(':')[0])


# In[ ]:


df['Reason']


# In[ ]:


df['Reason'].value_counts()


# In[ ]:


df['Reason'].value_counts().head(1)


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.countplot(x= 'Reason', data=df, palette='viridis')


# In[ ]:


df['timeStamp'].iloc[0]


# In[ ]:


df['timeStamp']= pd.to_datetime(df['timeStamp'])


# In[ ]:


type(df['timeStamp'].iloc[0])


# In[ ]:


time=df['timeStamp'].iloc[0]
time.hour


# In[ ]:


time.dayofweek


# In[ ]:


df['Hour']=df['timeStamp'].apply(lambda time: time.hour)


# In[ ]:


df['Hour']


# In[ ]:


df['Month']=df['timeStamp'].apply(lambda time: time.month)
df['Month']


# In[ ]:


df['Day of Week']=df['timeStamp'].apply(lambda time: time.dayofweek)
df['Day of Week']


# In[ ]:


df.head()


# In[ ]:


dmap= {0:'Mon',1:'Tue', 2:'Wed', 3:'Thu', 4:'Fri', 5:'Sat',6:'Sun'}


# In[ ]:


df['Day of the Week']= df['Day of Week'].map(dmap)


# In[ ]:


sns.countplot(x='Day of Week', data=df,hue='Reason', palette='viridis')


# In[ ]:


sns.countplot(x='Month', data=df,hue='Reason', palette='viridis')


# In[ ]:


byMonth = df.groupby('Month').count()
byMonth.head(12)


# In[ ]:


byMonth['lat'].plot()


# In[ ]:


sns.countplot(x='Month',data=df,palette='viridis')


# In[ ]:


sns.lmplot(x='Month',y='twp',data=byMonth.reset_index())


# In[ ]:


t=df['timeStamp'].iloc[0]
df['Date']=df['timeStamp'].apply(lambda t:t.date())
df.head()


# In[ ]:


t.date()
df.groupby('Date').count().head()
df.groupby('Date').count()['lat'].plot()
plt.tight_layout()


# In[ ]:


df[df['Reason']=='Traffic'].groupby('Date').count()['lat'].plot()
plt.title('Traffic')
plt.tight_layout()


# In[ ]:


df[df['Reason']=='EMS'].groupby('Date').count()['lat'].plot()
plt.title('EMS')
plt.tight_layout()


# In[ ]:


df[df['Reason']=='Fire'].groupby('Date').count()['lat'].plot()
plt.title('Fire')
plt.tight_layout()


# In[ ]:


Creating Heat Map


# In[ ]:


dayHour=df.groupby(by=['Day of Week','Hour']).count()['Reason'].unstack()
sns.heatmap(dayHour,cmap='viridis')


# In[ ]:


creating cluster map


# In[ ]:


sns.clustermap(dayHour,cmap='viridis')


# In[ ]:


dayMonth=df.groupby(by=['Day of Week','Month']).count()['Reason'].unstack()
sns.heatmap(dayMonth,cmap='viridis')


# In[ ]:


sns.clustermap(dayMonth,cmap='coolwarm')

