#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
from builtins import list
import matplotlib
matplotlib.style.use('ggplot')

import datetime

get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# Any results you write to the current directory are saved as output.


# In[ ]:


uber_df=pd.read_csv("../input/My Uber Drives - 2016.csv")


# I need to understand the dataset which i need to work on it

# In[ ]:


uber_df.head()


# In[ ]:


uber_df.tail()


# In[ ]:


# Remove uncessary data
uber_df = uber_df[:-1]


# In[ ]:


# fix data types of data columns

def convert_time(column_name):
    y=[]
    for x in uber_df[column_name]:
        y.append(datetime.datetime.strptime(x, "%m/%d/%Y %H:%M"))

    uber_df[column_name] = y


# In[ ]:


column_date=uber_df[['START_DATE*','END_DATE*']] 
for x in column_date:
    convert_time(x)


# In[ ]:


# check that all data is fixed and ready to work on it
uber_df.info()


# In[ ]:


# plot number of trip at each category
x = uber_df['CATEGORY*'].value_counts().plot(kind='bar')


# As we notice that the most trips made in business category with huge difference beteewn it and personal category.

# In[ ]:


#extract month from start date
count = 0
month=[]
while count < len(uber_df):
    month.append(uber_df['START_DATE*'][count].month)
    count = count+1
uber_df['Month'] = month


# In[ ]:


# calculate duration of each trip in minutes
minutes=[]
uber_df['Duration_Minutes'] = uber_df['END_DATE*'] - uber_df['START_DATE*']
uber_df['Duration_Minutes']
for x in uber_df['Duration_Minutes']:
    minutes.append(x.seconds / 60)

uber_df['Duration_Minutes'] = minutes


# In[ ]:


# plot number of trips at each month
x = uber_df['Month'].value_counts()
x.plot(kind='bar',figsize=(10,5),color='orange')
plt.xlabel('Month')
plt.ylabel('Frequency')
plt.title('Number of trips per Month')


# In[ ]:


# I need to see how many trip made at each clock and as you see the clock which has the higest number of trips is 3:00PM
hours = uber_df['START_DATE*'].dt.hour.value_counts()
hours.plot(kind='bar',color='orange',figsize=(10,5))
plt.xlabel('Hours')
plt.ylabel('Frequency')
plt.title('Number of trips per hour')


# In[ ]:


# see how many trips made by each purpose
purpose_time = uber_df['PURPOSE*'].value_counts()
purpose_time.plot(kind='bar',figsize=(10,5),color='green') 


# we need to know the speed of each drive to accomplish each trip, we need to calculate trip in hours at the first and save it into [duraion_hours] and then apply speed law speed = distance / time

# In[ ]:





# In[ ]:


# aveverage of each trip according to purpose
purpose = uber_df.groupby('PURPOSE*').mean()
purpose.plot(kind = 'bar',figsize=(15,5))


# In[ ]:


# calculate trip speed for each driver
uber_df['Duration_hours'] = uber_df['Duration_Minutes'] / 60
uber_df['Speed_KM'] = uber_df['MILES*'] / uber_df['Duration_hours']
uber_df['Speed_KM']


# In[ ]:




