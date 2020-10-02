#!/usr/bin/env python
# coding: utf-8

# This phase is focused on simple analysis techniques - to get more info about the system.

# In[25]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **1. Load and convert data types**
# 
# First load the data and convert timesetamps

# In[26]:


df = pd.read_csv('../input/sensor.csv', engine='python', sep = ',', decimal = '.')
df['timestamp'] =  pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S' ,utc='True')
df.head()


# **2. Preliminary investigation**
# 
# Check if the dt is equal

# In[27]:


d = df['timestamp'].diff()
plt.plot(d[1:])


# dt is equal - we can treat every column as evenly sampled series of data - wich simplifies analysis. We can drop Unnamed column with row numbers:
# 

# In[28]:


df.drop(['Unnamed: 0'], axis=1, inplace=True)


# Now lets check NaNs:

# In[29]:



print(df.shape)
nan_stats = df.isnull().sum().sort_values(ascending = False)/df.shape[0]
nan_stats


# Sensor 15 has no data - we can drop the column:

# In[30]:


df.drop(['sensor_15'], axis=1, inplace=True)


# We will also check the sensors that have more than 2% of nans - lets see it together with the machine status:

# In[31]:


# plot in two windows with status - to asses nan's presence according to pump state.
# divide sesnro values by its max to get better scaling to status (actual values are not of our interest here)

fig, (ax1, ax2, ax3) = plt.subplots(3,1)
ax1.set_ylabel(nan_stats.index[1])
ax1.plot(df['timestamp'],df['machine_status'])
ax1.plot(df['timestamp'],df[nan_stats.index[1]]/max(df[nan_stats.index[1]]))
ax2.set_ylabel(nan_stats.index[2])
ax2.plot(df['timestamp'],df['machine_status'])
ax2.plot(df['timestamp'],df[nan_stats.index[2]]/max(df[nan_stats.index[2]]))
ax3.set_ylabel(nan_stats.index[3])
ax3.plot(df['timestamp'],df['machine_status'])
ax3.plot(df['timestamp'],df[nan_stats.index[3]]/max(df[nan_stats.index[3]]))
plt.show()


fig, (ax1, ax2, ax3) = plt.subplots(3,1)
ax1.set_ylabel(nan_stats.index[4])
ax1.plot(df['timestamp'],df['machine_status'])
ax1.plot(df['timestamp'],df[nan_stats.index[4]]/max(df[nan_stats.index[4]]))
ax2.set_ylabel(nan_stats.index[5])
ax2.plot(df['timestamp'],df['machine_status'])
ax2.plot(df['timestamp'],df[nan_stats.index[5]]/max(df[nan_stats.index[5]]))
ax3.set_ylabel(nan_stats.index[6])
ax3.plot(df['timestamp'],df['machine_status'])
ax3.plot(df['timestamp'],df[nan_stats.index[6]]/max(df[nan_stats.index[6]]))
plt.show()


# We can see that:
# * sensor 50 fails after 6th failure of the pump and its reading are not available until the end of our dataframe. Maybe its not crucial for system operation and its repair is not economicaly justified.
# * Sensor 51 fails before 5th failure of the pump but during recovering from that failure gets back to normal operation
# * Sensors 00, 07, 08, 06 - have significant period of nan's during recovering state after 5th failure of the pump, but we may assume that their data coverage for failure prediction is OK
# 
# From these plots we can also assume that 5th failure may correspond to different failure mode than the others - but we don't have any confirmation of that fact.
# 
# Lets remove also the sensor_50 - we don't have 1/3 of the data and to reproduce the missing ones - we would have to estimate nearly 2 months of readings:

# In[32]:


df.drop(['sensor_50'], axis=1, inplace=True)


# Initially there was 52 sensors, we removed the 15th and 50th. Now we have 50 sensors and we don't know what kind of sensors they are (temperature, flow, pressure, vibration, pump motor current?). Lets investigate data ranges etc:

# In[33]:


# features extractor for columns
def get_column_features(df):
    
    df_features = pd.DataFrame()
    col_list = ['Col','Max','Min','q_0.25','q_0.50','q_0.75']
    
    for column in df:
        tmp_q = df[column].quantile([0.25,0.5,0.75])
        tmp1 = pd.Series([column,max(df[column]),min(df[column]),tmp_q[0.25],tmp_q[0.5],tmp_q[0.75]],name='d')
        tmp1.index = col_list
        df_features = df_features.append(tmp1)
        
    return df_features

# get list os signal columns without timestamp and status:
signal_columns = [c for c in df.columns if c not in ['timestamp', 'machine_status']]
get_column_features(df[signal_columns])


# By these features we can see, that:
# * all these sensors are analog sensors (no relays with only 0 and 1 values, binary switches etc). All of them have positive values.
# * we can also see that some of the sensors have max values expressed as whole numbers. For analog sensors - it can be symptom of reading truncated at scale end. We remember sensors 50 and 51 from nan's investigation - and by going by to plots we can see that nan's periods (failures of these sensors?) are preceded by achieving value of 1000 (saturated sensor/electrical problems/ loose cable connection?)
# 
# We will have all of that in mind during further investigation
# 

# **3. Correlation check**
# 
# Now lets check if some of the sensors are correlated:

# In[34]:


corr = df[signal_columns].corr()
sns.heatmap(corr)


# We can see strongly correlated group of sensors - from sensor_14 to sensor_26.
# There also some other correlated groups but not as strong as the mentioned one.
# 

# **4. Failure data investigation**
# 
# Lets locate the failure events:

# In[35]:


#locate indices of failure events and recovering and normal state
normal_idx = df.loc[df['machine_status'] == 'NORMAL'].index
failure_idx = df.loc[df['machine_status'] == 'BROKEN'].index
recovering_idx = df.loc[df['machine_status'] == 'RECOVERING'].index

bef_failure_idx = list()
for j in failure_idx:
    for i in range(24*60):
        bef_failure_idx.append(j-i)

bef_failure_idx.sort()

#locate timestamps of failures:
failures_timestamps = df.loc[failure_idx,'timestamp']
print(failures_timestamps)


# We can see that first 6 failures happened during late evening/night - only 7th was at 14pm.
# Lets check the sensors behaviour before the failure and during the recovering state:

# In[36]:


for i in range(10):
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5,1)
    
    ax1.set_ylabel(signal_columns[i*5])
    ax1.plot(df[signal_columns[i*5]],linewidth=1)
    ax1.plot(df.loc[list(recovering_idx),signal_columns[i*5]],linestyle='none',marker='.',color='red',markersize=4)
    ax1.plot(df.loc[bef_failure_idx,signal_columns[i*5]],linestyle='none',marker='.',color='orange',markersize=4)
        
    ax2.set_ylabel(signal_columns[i*5+1])
    ax2.plot(df[signal_columns[i*5+1]],linewidth=1)
    ax2.plot(df.loc[list(recovering_idx),signal_columns[i*5+1]],linestyle='none',marker='.',color='red',markersize=4)
    ax2.plot(df.loc[bef_failure_idx,signal_columns[i*5+1]],linestyle='none',marker='.',color='orange',markersize=4)
    
    ax3.set_ylabel(signal_columns[i*5+2])
    ax3.plot(df[signal_columns[i*5+2]],linewidth=1)
    ax3.plot(df.loc[list(recovering_idx),signal_columns[i*5+2]],linestyle='none',marker='.',color='red',markersize=4)
    ax3.plot(df.loc[bef_failure_idx,signal_columns[i*5+2]],linestyle='none',marker='.',color='orange',markersize=4)
    
    ax4.set_ylabel(signal_columns[i*5+3])
    ax4.plot(df[signal_columns[i*5+3]],linewidth=1)
    ax4.plot(df.loc[list(recovering_idx),signal_columns[i*5+3]],linestyle='none',marker='.',color='red',markersize=4)
    ax4.plot(df.loc[bef_failure_idx,signal_columns[i*5+3]],linestyle='none',marker='.',color='orange',markersize=4)
    
    ax5.set_ylabel(signal_columns[i*5+4])
    ax5.plot(df[signal_columns[i*5+4]],linewidth=1)
    ax5.plot(df.loc[list(recovering_idx),signal_columns[i*5+4]],linestyle='none',marker='.',color='red',markersize=4)
    ax5.plot(df.loc[bef_failure_idx,signal_columns[i*5+4]],linestyle='none',marker='.',color='orange',markersize=4)
    plt.show()


# First findings:
# * sensor 37 has diminishing trend and its behaviour its a bit odd. Also has saturated readings on the botton of the scale
# 
# to be continued - but before digging into real datascience and ML stuff some info about the system/failure modes/repair& maintenance policies/constraints would be helpful
