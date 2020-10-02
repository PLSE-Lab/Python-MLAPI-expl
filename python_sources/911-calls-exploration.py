#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
sns.set(style="white", color_codes=True)
import matplotlib.pyplot as plt


# In[ ]:


calls = pd.read_csv('../input/montcoalert/911.csv')


# In[ ]:


calls.head(3)


# In[ ]:


# top zip codes for 911 calls
calls['zip'].value_counts().head()


# In[ ]:


# top 5 townships for 911 calls 
calls['twp'].value_counts().head()


# In[ ]:


# unique titles
calls['title'].nunique()


# In[ ]:


# new colum "reason"
calls['reason'] = calls['title'].apply(lambda title : title.split(':')[0])


# In[ ]:


calls.reason.head()


# In[ ]:


#most common reason
calls['reason'].value_counts().head()


# In[ ]:


# plot 
sns.countplot(x='reason',data = calls)


# In[ ]:


calls.info()


# In[ ]:


calls['timeStamp'] = pd.to_datetime(calls['timeStamp'])
calls['timeStamp'][0].time()


# In[ ]:


calls['Hour'] = calls['timeStamp'].apply(lambda time : time.hour)
calls['Month'] = calls['timeStamp'].apply(lambda time : time.month)
calls['DayofWeek'] = calls['timeStamp'].apply(lambda time : time.dayofweek)
calls.head()


# In[ ]:


dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
calls['DayofWeek'] = calls['DayofWeek'].map(dmap)
calls.head(1)


# In[ ]:


sns.countplot(x='DayofWeek', data = calls,hue = 'reason')


# In[ ]:


sns.countplot(x='Month',data=calls,hue='reason')


# In[ ]:


byMonth = calls.groupby('Month').count()
byMonth.head()


# In[ ]:


sns.lmplot(x='Month',y='twp',data=byMonth.reset_index())


# In[ ]:


calls['Date'] = calls['timeStamp'].apply(lambda t : t.date())


# In[ ]:


calls.head()


# In[ ]:


calls.groupby('Date').count()['lat'].plot()
plt.tight_layout


# In[ ]:


calls[calls['reason'] == 'EMS'].groupby('Date').count()['lat'].plot()
plt.title('EMS')
plt.tight_layout


# In[ ]:


calls[calls['reason'] == 'Fire'].groupby('Date').count()['lat'].plot()
plt.title('Fire')
plt.tight_layout


# In[ ]:


dayHour = calls.groupby(by=['DayofWeek','Hour']).count()['reason'].unstack()
dayHour.head()


# In[ ]:


plt.figure(figsize=(12,6))
sns.heatmap(dayHour,cmap='viridis')


# In[ ]:


sns.clustermap(dayHour,cmap='viridis')


# In[ ]:




