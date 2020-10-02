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


import pandas as pd
import numpy as np


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv("../input/911.csv") 


# In[ ]:


df.info()


# In[ ]:


df.head()


# In[ ]:


#top 5 zip codes

df['zip'].value_counts().head(5)


# In[ ]:


#top 5 townships 
df['twp'].value_counts().head(5)


# In[ ]:


#unique values in 'title' column



'''
or df['title'].nunique
'''
len(df['title'].unique())


# In[ ]:


#feature engineering

x = df['title'].iloc[0]


# In[ ]:


x.split(':')


# In[ ]:


x.split(':')[0]


# In[ ]:


#for entire title column

df['Reason'] = df['title'].apply(lambda title:title.split(':')[0])


# In[ ]:


#most common reason for the 911 call
df['Reason'].value_counts()


# In[ ]:


#count plot for 'Reason'
sns.countplot(x='Reason',data=df)


# In[ ]:


#datatype of timestamp column

df.info('all')


# In[ ]:


#datatype of timestamp column
type(df['timeStamp'].iloc[0])


# In[ ]:


#changing str to datetime object
df['timeStamp'] = pd.to_datetime(df['timeStamp'])


# In[ ]:


time = df['timeStamp'].iloc[0]


# In[ ]:


time


# In[ ]:


time.minute


# In[ ]:


#making 3 different columns from timeStamp-> hour, month, day of week

df['hour'] = df['timeStamp'].apply(lambda timeStamp:timeStamp.hour)


# In[ ]:


df['hour']


# In[ ]:


df['month'] = df['timeStamp'].apply(lambda timeStamp:timeStamp.month)


# In[ ]:


df['month'].value_counts()


# In[ ]:


df['dayofweek'] = df['timeStamp'].apply(lambda timeStamp: timeStamp.dayofweek)


# In[ ]:


df['dayofweek']


# In[ ]:


df.head(5)


# In[ ]:


#map actual string names to day of week

dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}

df['dayofweek'] = df['dayofweek'].map(dmap)


# In[ ]:


df['dayofweek']


# In[ ]:


#count plot for 'day of week' and hue based on 'Reason' col

sns.countplot(x='dayofweek',data=df, hue='Reason')

#to relocate the legend
plt.legend(bbox_to_anchor=(1.05, 1),loc=2, borderaxespad=0)


# In[ ]:


#count plot for 'month' and hue based on 'Reason' col
sns.countplot(x='month',data=df, hue='Reason')

#to relocate the legend
plt.legend(bbox_to_anchor=(1.05, 1),loc=2, borderaxespad=0)


# In[ ]:


#dealing with missing months using groupby

bymonth = df.groupby('month').count()


# In[ ]:


bymonth.head()


# In[ ]:


#plotting line plot

bymonth['lat'].plot()


# In[ ]:


#linear fit model for No. of calls per month 

sns.lmplot(x='month', y='twp', data=bymonth.reset_index())


# In[ ]:


#new col containing date

df['date'] = df['timeStamp'].apply(lambda timeStamp: timeStamp.date())


# In[ ]:


df['date'].value_counts().head()


# In[ ]:


#groupby date

bydate = df.groupby('date').count()


# In[ ]:


bydate


# In[ ]:


#plot of count by date

bydate['lat'].plot()
plt.tight_layout()


# In[ ]:


#3 separate plots on the basis of reasons

df.groupby('date').count()['lat'].plot()
plt.tight_layout()


# In[ ]:


df[df['Reason']=='EMS'].groupby('date').count()['lat'].plot()
plt.title('EMS')
plt.tight_layout()


# In[ ]:


df[df['Reason']=='Fire'].groupby('date').count()['lat'].plot()
plt.title('Fire')
plt.tight_layout()


# In[ ]:


df[df['Reason']=='Traffic'].groupby('date').count()['lat'].plot()
plt.title('Traffic')
plt.tight_layout()


# In[ ]:


#creating heatmaps

#grouping by 2 cols -> than using unstack to make one as col and one as row 

dayhour = df.groupby(by=['dayofweek','hour']).count()['Reason'].unstack()


# In[ ]:


#heatmap
plt.figure(figsize=(12,6))
sns.heatmap(dayhour)


# In[ ]:


#month and day of week

daymonth = df.groupby(by=['month','dayofweek']).count()['Reason'].unstack()


# In[ ]:


plt.figure(figsize=(12,6))
sns.heatmap(daymonth)


# In[ ]:


sns.clustermap(dayhour, cmap='coolwarm')


# In[ ]:


sns.clustermap(daymonth, cmap='coolwarm')


# In[ ]:




