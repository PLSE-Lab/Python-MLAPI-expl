#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


pd.set_option('display.max_columns',55)


# In[ ]:


pd.set_option('display.max_rows',1500)


# In[ ]:


data = pd.read_csv('/kaggle/input/pump-sensor-data/sensor.csv')


# In[ ]:


data.info()


# There are many null objects in many features, sensor_15 is completely null

# In[ ]:


del data['Unnamed: 0']


# In[ ]:


data.head(15)


# In[ ]:


data.index = data['timestamp']


# In[ ]:


data.index = pd.to_datetime(data.index)


# In[ ]:


del data['timestamp']


# In[ ]:


data.describe()


# We can see that different sensors have different scales

# In[ ]:


corr  = data.corr()


# In[ ]:


fig, ax = plt.subplots(figsize=(8,8))  
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
    
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)


# Based on correlation, there are three sets of clusters
# Basically what it means is some sensors of 0 to 14 have high correlations with each other
# Sensors 14 to 36 have high correlation with each other
# and then Sensors 38 to 41 have correlations with each other
# but there is not significant correlation between these clusters of Sensors

# In[ ]:


data['machine_status'].unique()


# There are 3 unique label values, namely: NORMAL, BROKEN and RECOVERING

# In[ ]:


data[(data['machine_status'] == 'BROKEN')]


# In[ ]:


data[(data['machine_status'] == 'RECOVERING')]


# When it is broken, the next minute onwards, the status is considered as recovering until it becomes normal again 

# In[ ]:


data[(data['machine_status'] == 'RECOVERING')].info()


# When the machine is in recovering status, the following have more NaN values: **sensor00, sensor06, sensor07, sensor08, sensor09, sensor_51** has some NaN values but the number is lot less than than the other sensors

# In[ ]:


columns = ['sensor_00','sensor_06','sensor_07','sensor_08','sensor_09','sensor_51']


# In[ ]:


for column in columns:
    print('{0} Original'.format(column))
    display(data[(data['machine_status'] == 'NORMAL')][column].describe())
    print('{0} In Recovery'.format(column))
    display(data[(data['machine_status'] == 'RECOVERING')][column].describe())


# There is a significant change in standard deviations and means of these sensors during the normal state and not normal state

# In[ ]:


import matplotlib.dates as mdates


# In[ ]:


fig, ax = plt.subplots(figsize=(15,5))
plt.xticks(rotation=70)
ax.plot(data.loc['2018-04-12 21:00:00':'2018-04-12 22:15:00', 'sensor_00'],marker='o', linestyle='-')
plt.grid(True) 
ax.set_ylabel('Reading Unit')
ax.set_title('First Broken: Sensor_00 Reading')
# Set x-axis major ticks to weekly interval, on Mondays
ax.xaxis.set_major_locator(mdates.MinuteLocator())
# Format x-tick labels as 3-letter month name and day number
ax.xaxis.set_major_formatter(mdates.DateFormatter('%HH:%MM'));


# In[ ]:


fig, ax = plt.subplots(figsize=(15,5))
plt.xticks(rotation=70)
ax.plot(data.loc['2018-04-12 21:00:00':'2018-04-12 22:15:00', 'sensor_06'],marker='o', linestyle='-')
plt.grid(True) 
ax.set_ylabel('Reading Unit')
ax.set_title('First Broken: Sensor_06 Reading')
# Set x-axis major ticks to weekly interval, on Mondays
ax.xaxis.set_major_locator(mdates.MinuteLocator())
# Format x-tick labels as 3-letter month name and day number
ax.xaxis.set_major_formatter(mdates.DateFormatter('%HH:%MM'));


# In[ ]:


fig, ax = plt.subplots(figsize=(15,5))
plt.xticks(rotation=70)
ax.plot(data.loc['2018-04-12 21:00:00':'2018-04-12 22:15:00', 'sensor_07'],marker='o', linestyle='-')
plt.grid(True) 
ax.set_ylabel('Reading Unit')
ax.set_title('First Broken: Sensor_07 Reading')
# Set x-axis major ticks to weekly interval, on Mondays
ax.xaxis.set_major_locator(mdates.MinuteLocator())
# Format x-tick labels as 3-letter month name and day number
ax.xaxis.set_major_formatter(mdates.DateFormatter('%HH:%MM'));


# In[ ]:


fig, ax = plt.subplots(figsize=(15,5))
plt.xticks(rotation=70)
ax.plot(data.loc['2018-04-12 21:00:00':'2018-04-12 22:15:00', 'sensor_08'],marker='o', linestyle='-')
plt.grid(True) 
ax.set_ylabel('Reading Unit')
ax.set_title('First Broken: Sensor_08 Reading')
# Set x-axis major ticks to weekly interval, on Mondays
ax.xaxis.set_major_locator(mdates.MinuteLocator())
# Format x-tick labels as 3-letter month name and day number
ax.xaxis.set_major_formatter(mdates.DateFormatter('%HH:%MM'));


# In[ ]:


fig, ax = plt.subplots(figsize=(15,5))
plt.xticks(rotation=70)
ax.plot(data.loc['2018-04-12 21:00:00':'2018-04-12 22:15:00', 'sensor_09'],marker='o', linestyle='-')
plt.grid(True) 
ax.set_ylabel('Reading Unit')
ax.set_title('First Broken: Sensor_09 Reading')
# Set x-axis major ticks to weekly interval, on Mondays
ax.xaxis.set_major_locator(mdates.MinuteLocator())
# Format x-tick labels as 3-letter month name and day number
ax.xaxis.set_major_formatter(mdates.DateFormatter('%HH:%MM'));


# In[ ]:


fig, ax = plt.subplots(figsize=(15,5))
plt.xticks(rotation=70)
ax.plot(data.loc['2018-04-12 21:00:00':'2018-04-12 22:15:00', 'sensor_51'],marker='o', linestyle='-')
plt.grid(True) 
ax.set_ylabel('Reading Unit')
ax.set_title('First Broken: Sensor_51 Reading')
# Set x-axis major ticks to weekly interval, on Mondays
ax.xaxis.set_major_locator(mdates.MinuteLocator())
# Format x-tick labels as 3-letter month name and day number
ax.xaxis.set_major_formatter(mdates.DateFormatter('%HH:%MM'));


# The sensors which have many null values in  'recovering' class show a sudden change when a failure occurs, which means that they may be sensors which are  heavily dependent on the system

# In[ ]:


#Getting the list of columns
cols = list(data.columns)


# In[ ]:


#Removing sensor 15 as it is completely null
cols.remove('sensor_15')


# In[ ]:


for i in cols:
    fig, ax = plt.subplots(figsize=(18,5))
    plt.xticks(rotation=90)
    ax.plot(data.loc['2018-04-12 12:00:00':'2018-04-14 12:00:00', i],marker='o', linestyle='-')
    plt.grid(True) 
    ax.set_ylabel('Reading Unit')
    ax.set_title('First Broken: {0} Reading'.format(i))
    # Set x-axis major ticks to weekly interval, on Mondays
    ax.xaxis.set_major_locator(mdates.HourLocator())
    # Format x-tick labels as 3-letter month name and day number
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%HH:%MM'))


# > **First Broken:** To get back to normal state, it took around 16 hours

# In[ ]:


for i in cols:
    fig, ax = plt.subplots(figsize=(18,5))
    plt.xticks(rotation=90)
    ax.plot(data.loc['2018-04-17 12:00:00':'2018-04-19 12:00:00', i],marker='o', linestyle='-')
    plt.grid(True) 
    ax.set_ylabel('Reading Unit')
    ax.set_title('Second Broken: {0} Reading'.format(i))
    # Set x-axis major ticks to weekly interval, on Mondays
    ax.xaxis.set_major_locator(mdates.HourLocator())
    # Format x-tick labels as 3-letter month name and day number
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%HH:%MM'))


# > **Second Broken:** Even after plotting for more than 24 hours, it has still not recovered

# In[ ]:


for i in cols:
    fig, ax = plt.subplots(figsize=(18,5))
    plt.xticks(rotation=90)
    ax.plot(data.loc['2018-05-18 20:00:00':'2018-05-20 20:00:00', i],marker='o', linestyle='-')
    plt.grid(True) 
    ax.set_ylabel('Reading Unit')
    ax.set_title('Third Broken: {0} Reading'.format(i))
    # Set x-axis major ticks to weekly interval, on Mondays
    ax.xaxis.set_major_locator(mdates.HourLocator())
    # Format x-tick labels as 3-letter month name and day number
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%HH:%MM'))


# > **Third Broken:** To get back to normal state, it took around 22 hours

# In[ ]:


for i in cols:
    fig, ax = plt.subplots(figsize=(18,5))
    plt.xticks(rotation=90)
    ax.plot(data.loc['2018-05-24 12:00:00':'2018-05-26 12:00:00', i],marker='o', linestyle='-')
    plt.grid(True) 
    ax.set_ylabel('Reading Unit')
    ax.set_title('Fourth Broken: {0} Reading'.format(i))
    # Set x-axis major ticks to weekly interval, on Mondays
    ax.xaxis.set_major_locator(mdates.HourLocator())
    # Format x-tick labels as 3-letter month name and day number
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%HH:%MM'))


# > **Fourth Broken:** To get back to normal state, it took around 10 hours

# In[ ]:


for i in cols:
    if i == 'sensor_51':
        continue
    fig, ax = plt.subplots(figsize=(18,5))
    plt.xticks(rotation=90)
    ax.plot(data.loc['2018-06-28 12:00:00':'2018-06-30 12:00:00', i],marker='o', linestyle='-')
    plt.grid(True) 
    ax.set_ylabel('Reading Unit')
    ax.set_title('Fifth Broken: {0} Reading'.format(i))
    # Set x-axis major ticks to weekly interval, on Mondays
    ax.xaxis.set_major_locator(mdates.HourLocator())
    # Format x-tick labels as 3-letter month name and day number
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%HH:%MM'))


# Skipping sensor_51 as it is completely null for this broken state

# > **Fifth Broken:** Even after plotting for more than 24 hours, it has still not recovered

# In[ ]:


for i in cols:
    fig, ax = plt.subplots(figsize=(18,5))
    plt.xticks(rotation=90)
    ax.plot(data.loc['2018-07-07 12:00:00':'2018-07-09 12:00:00', i],marker='o', linestyle='-')
    plt.grid(True) 
    ax.set_ylabel('Reading Unit')
    ax.set_title('Sixth Broken: {0} Reading'.format(i))
    # Set x-axis major ticks to weekly interval, on Mondays
    ax.xaxis.set_major_locator(mdates.HourLocator())
    # Format x-tick labels as 3-letter month name and day number
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%HH:%MM'))


# > **Sixth Broken**: To get back to normal state, it took around 1 hour

# In[ ]:


for i in cols:
    if i == 'sensor_50':
        continue
    fig, ax = plt.subplots(figsize=(18,5))
    plt.xticks(rotation=90)
    ax.plot(data.loc['2018-07-24 12:00:00':'2018-07-26 12:00:00', i],marker='o', linestyle='-')
    plt.grid(True) 
    ax.set_ylabel('Reading Unit')
    ax.set_title('Seventh Broken: {0} Reading'.format(i))
    # Set x-axis major ticks to weekly interval, on Mondays
    ax.xaxis.set_major_locator(mdates.HourLocator())
    # Format x-tick labels as 3-letter month name and day number
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%HH:%MM'))


# Skipping sensor_50 as it is completely null for this broken even

# > **Seventh Broken**: To get back to normal state, it took around 1 hour

# > Plotting second and fifth broken event, once again with more window to see how much time it takes to recover

# In[ ]:


fig, ax = plt.subplots(figsize=(18,5))
plt.xticks(rotation=90)
ax.plot(data.loc['2018-04-17 12:00:00':'2018-04-20 12:00:00', 'machine_status'],marker='o', linestyle='-')
plt.grid(True) 
ax.set_ylabel('Reading Unit')
ax.set_title('Second Broken: machine_status Reading')
# Set x-axis major ticks to weekly interval, on Mondays
ax.xaxis.set_major_locator(mdates.HourLocator())
# Format x-tick labels as 3-letter month name and day number
ax.xaxis.set_major_formatter(mdates.DateFormatter('%HH:%MM'))


# **Second Broken**: To get back to normal state, it took around 52 hours

# In[ ]:



fig, ax = plt.subplots(figsize=(18,5))
plt.xticks(rotation=90)
ax.plot(data.loc['2018-06-28 12:00:00':'2018-07-06 12:00:00', 'machine_status'],marker='o', linestyle='-')
plt.grid(True) 
ax.set_ylabel('Reading Unit')
ax.set_title('Fifth Broken: machine_status Reading')
# Set x-axis major ticks to weekly interval, on Mondays
ax.xaxis.set_major_locator(mdates.DayLocator())
# Format x-tick labels as 3-letter month name and day number
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d'))


# > **Second Broken**: To get back to normal state, it took almost 6 days, this is the largest broken state in the dataset

# In[ ]:




