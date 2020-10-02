#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# Importing importannt libraries

# In[2]:


df=pd.read_csv('../input/data.csv',encoding="ISO-8859-1")
df.head(5)


# **Some Defintions for our Understanding:**
# 
# **NO2:** Combustion from power sources or Transport.
# 
# **SO2:** Coal burning, Oil burning, Manufacturing of Sulphuric acid.
# 
# **spm:** Suspended particulate matter and are known to be the deadliest form of air pollution. They are microscopic in nature and are found to be suspended in earth's atmosphere.
# 
# **rspm:** Respirable suspended particulate matter. A sub form of spm and are responsible for respiratory diseases.
# 
# **pm2_5: **Suspended particulate matter with diameters less than 2.5 micrometres. They tend to remain suspended for longer durations and potentially very harmful.
# 
# 

# **Question 1**: Find out the states with minmum/maximum pollution parameters.

# In[ ]:


fig, axes= plt.subplots(figsize=(20, 12), ncols=5)
state_wise_max_so2 = df[['state','so2']].dropna().groupby('state').median().sort_values(by='so2')
state_wise_max_no2 = df[['state','no2']].dropna().groupby('state').median().sort_values(by='no2')
state_wise_max_rspm = df[['state','rspm']].dropna().groupby('state').median().sort_values(by='rspm')
state_wise_max_spm = df[['state','spm']].dropna().groupby('state').median().sort_values(by='spm')
state_wise_max_pm2_5 = df[['state','pm2_5']].dropna().groupby('state').median().sort_values(by='pm2_5')

sns.barplot(x='so2', y=state_wise_max_so2.index, data=state_wise_max_so2, ax=axes[0])
axes[0].set_title("Average so2 observed in a state")

sns.barplot(x='no2', y=state_wise_max_no2.index, data=state_wise_max_no2, ax=axes[1])
axes[1].set_title("Average no2 observed in a state")

sns.barplot(x='rspm', y=state_wise_max_rspm.index, data=state_wise_max_rspm, ax=axes[2])
axes[2].set_title("Average rspm observed in a state")

sns.barplot(x='spm', y=state_wise_max_spm.index, data=state_wise_max_spm, ax=axes[3])
axes[3].set_title("Average spm observed in a state")

sns.barplot(x='pm2_5', y=state_wise_max_pm2_5.index, data=state_wise_max_pm2_5, ax=axes[4])
axes[4].set_title("Average pm2_5 observed in a state")
plt.tight_layout()


# **GOA looks a suitable location overall :)**

# **Q2:** Top TEN Cities have higest risk of respiratory diseases?

# In[ ]:


state = df[['state','location','rspm']].groupby(['state','location']).median().reset_index()
state_location_max_rspm = state.loc[state.groupby('state')['rspm'].idxmax()].sort_values(by='rspm', ascending=False).head(10)
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x='rspm', y= 'location', data=state_location_max_rspm, palette='coolwarm', axes=ax)
sns.despine(left=True)
ax.set_title("Bars showing average rspm values of the cities")
plt.tight_layout()


# **Q3:** Show RSPM variation in ghaziabad over time

# In[ ]:


rspm_data = df[df['location']=="Ghaziabad"][['date', 'rspm']].dropna()
fig, ax = plt.subplots(figsize=(12,8))
ax.xaxis.set_major_locator(plt.MaxNLocator(20))
sns.lineplot(x='date', y='rspm', data=rspm_data, axes=ax, label="rspm")
fig.autofmt_xdate()
plt.tight_layout()


# **Q4: **Top five states with highest number of monitoring stations 

# In[ ]:


mon_station = df.drop_duplicates(subset=['location_monitoring_station'])
grouped_mon_station= mon_station[['state', 'location_monitoring_station']].groupby('state').count().sort_values(by='location_monitoring_station', ascending=False).head(5)
fig,ax = plt.subplots(figsize=(12,6))
sns.set_style("dark")
sns.barplot(x=grouped_mon_station.index, y='location_monitoring_station', data=grouped_mon_station, axes=ax)
sns.despine()
plt.tight_layout()


# **Q5:** Top 5 sampling stations to produce highest samples

# In[ ]:


location_mon = df[['state', 'location_monitoring_station', 'date']].groupby(['state','location_monitoring_station']).count().reset_index()
location_mon.loc[location_mon.groupby('state')['date'].idxmax()].sort_values(by='date', ascending=False).head(5)


# **Q4:** 5 sampling stations to produce least samples

# In[ ]:


location_mon = df[['state', 'location_monitoring_station', 'date']].groupby(['state','location_monitoring_station']).count().reset_index()
location_mon.loc[location_mon.groupby('state')['date'].idxmax()].sort_values(by='date').head(5)


# **Q5:** Check if there is some correlation between so2 and no2 variation.
# 
# I am taking a sample of one city for this purpose instead of taking whole data.

# In[16]:


relation_df_so2_no2 = df[(df['state']=='Uttar Pradesh') & (df['location'] == 'Agra')][['so2','no2']].dropna()
sns.jointplot( x='so2', y='no2', data=relation_df_so2_no2 , size=12)

