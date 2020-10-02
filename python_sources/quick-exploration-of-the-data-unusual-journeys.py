#!/usr/bin/env python
# coding: utf-8

# ## A quick look at the number of journeys and average journey time by day and hour
# 
# 
# This is my first pulic posting so lets keep it simple. My python isn't top notch but I
# get by.

# In[2]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt

get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[3]:


train = pd.read_csv('../input/train.csv')
train.shape


# In[4]:


days = ['Mon', 'Tue', 'Wed', 'Thr', 'Fri', 'Sat', 'Sun']


# ## add in some useful identifiers
# 
# amendment - just realised that my days were one to long and saturday was included as a working day.

# In[5]:


train['day'] = pd.to_datetime(train['pickup_datetime']).dt.weekday
train['date'] = pd.to_datetime(train['pickup_datetime']).dt.date
train['hour'] = pd.to_datetime(train['pickup_datetime']).dt.hour
train['rushhour'] = np.where(train['hour'].isin([7,8,16,17]) & train['day'].isin([0,1,2,3,4,]), 'yes', 'no')
train['days'] = train['day'].apply(lambda x: days[x])


# In[8]:


##is there a dfference between rushhour and non rush hour journey mean

pd.pivot_table(train[['rushhour','day','trip_duration']], 
               index='day', columns='rushhour', aggfunc='mean').plot.bar()


# In[ ]:





# In[9]:


## number of journeys per day
pd.pivot_table(train[['id','days']], columns='days', aggfunc="count")


# In[10]:


## number of journeys per day/hour
pd.pivot_table(train[['hour','days','id']], index='hour', columns='days', aggfunc="count").plot.line(figsize=(10,8))


# In[11]:


#mean journey time by hour and day
pd.pivot_table(train[['hour','day','trip_duration']], index='hour', columns='day', aggfunc="mean").plot.line(figsize=(10,8))


# In[ ]:





# ## late night on a saturday has some really long journey times - check these out
# 
# and wht is going on on a thursday morning =whats with the splike in trip duration at about 4am? and a fritag at 3am
# 
# bizarrley the peak time is journeys between 3pm and 4pm weekdays??

# In[12]:


#lets just look at trips on a saturday

dayview = train.loc[train['days'] == 'Sat']
pd.pivot_table(dayview[['date','hour','trip_duration']],
               index='hour', columns='date', aggfunc='mean').plot.line(figsize=(10,8), legend=False)


# ### One Saturday looks very odd lets take a closer look at that date

# In[13]:


pd.pivot_table(dayview[['date','hour','trip_duration']].loc[dayview['hour']==22],
               index='hour', columns='date', aggfunc='mean').sum()


# ### Looks like 13th Feb was an odd Saturday - could have been really busy with valentines perhaps, lets have a look at all the journeys longer than the average

# In[14]:


dayview.loc[(dayview['date'] == dt.date(2016, 2, 13)) & (dayview['trip_duration'] >12000)].describe()


# In[15]:


3526282/3600/24


# ## There is a trip in there that last 40 days!!!!
# 
# Someone either has way to much cash and the driver needs zero sleep or there is crappy data - I'm  going with the crap data theory
# 
#     Lets have a look for all trips longer than 5 hours - these are not the norm,  probably anything longer than an hour should be considered odd for a taxi - or they just use taxis differently in NY.
#     
#  I have been on shorter holidays than some of those trips.

# In[16]:


train.loc[train['trip_duration'] >= 18000].describe()


# ### how would the days look if we take out these weird long journeys 

# In[17]:


#mean journey time by hour and day , journeys less than 5 hours
pd.pivot_table(train[['hour','days','trip_duration']].loc[train['trip_duration'] <= 18000],
                                                         index='hour',
                                                         columns='days',
                                                         aggfunc="mean").plot.line(figsize=(10,8))


# ### That to me looks more representative of what i would expect of taxi journeys in a major western city, longest journeys at the weekends in the afternoon not late on a saturday evening whenr roads should be quiet.
# 
# Monday days now looks a bit off in the afternoon though - more bad data? - lets isolate the day and find out.

# In[23]:


mondayview = train.loc[(train['days'] == 'Mon')]# & (train['trip_duration'] < 18000)]
pd.pivot_table(mondayview[['date','hour','trip_duration']],
               index='hour', columns='date', aggfunc='mean').plot.line(figsize=(10,8), legend=False)


# ### Some crazy morning journeys on a monday and some really low journey times around 3pm lets  look at journey less than the mean for 3pm

# In[24]:


mondayview.loc[(mondayview['trip_duration'] < mondayview.loc[mondayview['hour'] ==
                                                    15]['trip_duration'].mean()) &
            (mondayview['hour'] == 15)].sort_values('trip_duration')


# ### It looks like there are a whole bunch of trips with durations shorter than the time it takes to get seated in a taxi!
# lets have a look at trips less than 15s as this seems ludicrous - it can take that long to pull into traffic.

# In[25]:


train.loc[train['trip_duration'] <= 15].describe()


# ## A new frame without the really long and stupidly short journeys
# how does our daily mean taxi journey look then

# In[26]:


trained = train.loc[(train['trip_duration'] > 15) & (train['trip_duration'] < 18000) ].copy()


# In[27]:


#mean journey time by hour and day
pd.pivot_table(trained[['hour','days','trip_duration']],
                index='hour',
                columns='days',
                aggfunc="mean").plot.line(figsize=(10,8))


# ### Monday stil looks odd - maybe its the other days that are weird, maybe its all right, maybe.....

# In[28]:


##do the rush hours look any fi=different with these stripped out

pd.pivot_table(train[['rushhour','day','trip_duration']], 
               index='day', columns='rushhour', aggfunc='mean').plot.bar()

