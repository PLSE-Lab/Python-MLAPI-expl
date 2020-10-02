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


import numpy as np
import pandas as pd


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('../input/911.csv')


# In[ ]:


df.info()


# In[ ]:


df.head(3)


# In[ ]:


# Top 5 zip codes for 911 Calls
df['zip'].value_counts().head(5)


# In[ ]:


# Top 5 townships (twp) for 911 calls
df['twp'].value_counts().head(5)


# In[ ]:


# Unique titles
df['title'].nunique()


# In[ ]:


# Lets create Column for EMS, Fire, and Traffic from Title Column
df['Reason'] = df['title'].apply(lambda title: title.split(':')[0])


# In[ ]:


df['Reason'].value_counts()


# In[ ]:


sns.countplot(x='Reason',data=df,palette='viridis')


# In[ ]:


type(df['timeStamp'].iloc[0])
#timestamp is a String


# In[ ]:


df['timeStamp'] = pd.to_datetime(df['timeStamp'])


# In[ ]:


df['Hour'] = df['timeStamp'].apply(lambda time: time.hour)
df['Month'] = df['timeStamp'].apply(lambda time: time.month)
df['Day of Week'] = df['timeStamp'].apply(lambda time: time.dayofweek)


# In[ ]:


# Since day of the week is an integer
dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
df['Day of Week'] = df['Day of Week'].map(dmap)


# In[ ]:


sns.countplot(x='Day of Week',data=df,hue='Reason',palette='viridis')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# In[ ]:


sns.countplot(x='Month',data=df,hue='Reason',palette='viridis')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# In[ ]:


# Notice that 9, 10 and 11 Months are missing..
byMonth = df.groupby('Month').count()
byMonth.head()


# In[ ]:


byMonth['twp'].plot()


# In[ ]:


sns.lmplot(x='Month',y='twp',data=byMonth.reset_index())


# In[ ]:


# Creating Data Column
df['Date']=df['timeStamp'].apply(lambda t: t.date())


# In[ ]:


df.groupby('Date').count()['twp'].plot()
plt.tight_layout()


# In[ ]:


# Creating the same for Traffic, Fire and EMS
df[df['Reason']=='Traffic'].groupby('Date').count()['twp'].plot()
plt.title('Traffic')
plt.tight_layout()


# In[ ]:


df[df['Reason']=='Fire'].groupby('Date').count()['twp'].plot()
plt.title('Fire')
plt.tight_layout()


# In[ ]:


df[df['Reason']=='EMS'].groupby('Date').count()['twp'].plot()
plt.title('EMS')
plt.tight_layout()


# In[ ]:


dayHour = df.groupby(by=['Day of Week','Hour']).count()['Reason'].unstack()
dayHour.head()


# In[ ]:


# Creating some Heatmap
plt.figure(figsize=(12,6))
sns.heatmap(dayHour,cmap='viridis')


# In[ ]:


sns.clustermap(dayHour,cmap='viridis')


# In[ ]:


dayMonth = df.groupby(by=['Day of Week','Month']).count()['Reason'].unstack()
dayMonth.head()


# In[ ]:


plt.figure(figsize=(12,6))
sns.heatmap(dayMonth,cmap='viridis')


# In[ ]:


sns.clustermap(dayMonth,cmap='viridis')

