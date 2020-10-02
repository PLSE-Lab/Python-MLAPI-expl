#!/usr/bin/env python
# coding: utf-8

# Sorry, not much to see here.  This is my script to spot check the data uploaded.

# In[ ]:


import pandas as pd
import numpy as np
import datetime


import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)


dateparse = lambda x: datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S')

# Read data 
d=pd.read_csv("../input/911.csv",
    header=0,names=['lat', 'lng','desc','zip','title','timeStamp','twp','addr','e'],
    dtype={'lat':str,'lng':str,'desc':str,'zip':str,
                  'title':str,'timeStamp':str,'twp':str,'addr':str,'e':int}, 
     parse_dates=['timeStamp'],date_parser=dateparse)


# Set index
d.index = pd.DatetimeIndex(d.timeStamp)
d=d[(d.timeStamp >= "2016-01-01 00:00:00")]


# In[ ]:


d['timeStamp'].max(),d['timeStamp'].min()


# In[ ]:


d['day']=d['timeStamp'].dt.day
d['month']=d['timeStamp'].dt.month
d['date']=d['timeStamp'].dt.date
d['hr']=d['timeStamp'].dt.hour


# Take a look at the day count.  See if any days have missing values.
# 

# In[ ]:


min(d['date'].value_counts())


# In[ ]:


t=d[(d.timeStamp >= "2017-03-08 16:00:00") & (d.timeStamp <= "2017-03-08 23:00:00")]
t['hr'].value_counts()


# In[ ]:


t=d[(d.timeStamp >= "2017-03-01 16:00:00") & (d.timeStamp <= "2017-03-08 23:00:00")]
t=t[(t['hr']==21)]

t.groupby(['hr','date']).size()


# Check lat and lng values, which should be in the following range:
# ('41.1671565', '30.3335960', '-95.5955947', '-74.9930755')

# In[ ]:


d['lat'].max(),d['lat'].min(),d['lng'].max(),d['lng'].min()


# ## Calls of Interest ##
# 
# These are not problem with the data, but may be of interest to researchers.

# In[ ]:


d[(d['title']=='EMS: ACTIVE SHOOTER')]


# In[ ]:


#Fire: TRAIN CRASH 
d[(d['title']=='Fire: TRAIN CRASH')]


# In[ ]:


# EMS: PLANE CRASH    
d[(d['title']=='EMS: PLANE CRASH')]

