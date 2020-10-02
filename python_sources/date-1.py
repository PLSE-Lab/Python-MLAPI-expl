#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd 
from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))
import datetime

train1 = pd.read_csv("../input/train.csv", parse_dates=['date_time'], nrows=1000)


# In[ ]:


train1 = pd.read_csv("../input/train.csv", parse_dates=['date_time'], nrows=10)


# In[ ]:


train1['srch_ci']


# In[ ]:


#df['year'] = df['ArrivalDate'].dt.year
#df['year'] = pd.DatetimeIndex(df['ArrivalDate']).year
train1['month'] = pd.DatetimeIndex(train1['date_time']).month
train1['year'] = train1['date_time'].dt.year
train1['hour'] = train1['date_time'].dt.hour


# In[ ]:


train1[['year','hour','month']]


# In[ ]:


train1.ix[(train1['hour'] >= 5) & (train1['hour'] <= 10), 'hour'] = 2


# In[ ]:


train1['hour']


# In[ ]:


train1.ix[train1.hour >= 5,'year'] = 1


# In[ ]:


train1['year']


# In[ ]:


t = pd.tslib.Timestamp.now()


# In[ ]:


t


# In[ ]:


t.to_datetime()


# In[ ]:


t.hour


# In[ ]:


t.date()


# In[ ]:


t.dayofweek


# In[ ]:


t.day


# In[ ]:


t.year


# In[ ]:


t.time()


# In[ ]:


t.month


# In[ ]:


a=[]


# In[ ]:


a.append(int(t.month))


# In[ ]:


a.append(int(t.year))


# In[ ]:


a


# In[ ]:


len(a)

