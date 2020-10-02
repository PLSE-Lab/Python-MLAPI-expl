#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ### Load dataset

# In[ ]:


who_df = pd.read_csv('/kaggle/input/uncover/UNCOVER_v4/UNCOVER/WHO/world-health-organization-who-situation-reports.csv', parse_dates=True)
who_df.head()


# In[ ]:


who_df['date']=pd.to_datetime(who_df['date'])
who_df=who_df.set_index('date')


# In[ ]:


who_df.isna().sum()


# ### Fill missing values with 0 assuming no cases/tests/deaths found

# In[ ]:


who_df.fillna(0)


# In[ ]:


who_df[['new_cases','new_deaths','total_deaths']]=who_df[['new_cases','new_deaths','total_deaths']].astype(int)


# ### Total corona cases in the world

# In[ ]:


plt.figure(figsize=(12,10))
world_df = who_df.loc[who_df['location']=='World']
world_df.plot()
plt.show()


# ### Get country wise totals

# In[ ]:


country_grouped_df = who_df.groupby(['location']).sum().reset_index()
country_grouped_df = country_grouped_df[(country_grouped_df['location'] != 'World') & (country_grouped_df['total_deaths'] > 500)]


# In[ ]:


fig,ax = plt.subplots(figsize=(20,8))
ax.bar(country_grouped_df['location'],country_grouped_df['total_deaths'])
ax.set_xlabel('Location')
ax.set_ylabel('Total Deaths')
plt.xticks(rotation='vertical')
plt.yticks(np.arange(0, country_grouped_df['total_deaths'].max(), step=3000))
plt.subplots_adjust(bottom=0.15)
plt.show()


# In[ ]:


import datetime as dt
import matplotlib.dates as mdates
from datetime import timedelta as tdelta


fig, ax = plt.subplots(figsize=(10,8))
world_df = who_df.loc[who_df['location']=='World',['new_cases','total_cases']]

x = [dt.datetime(2020, 2, 17)]
count_spike_cases = world_df[world_df.index == dt.datetime(2020, 2, 17)]['new_cases'].values[0]

ax.plot_date(world_df.index, world_df['new_cases'])
plt.xlabel('Date')
plt.ylabel('Count of cases')
plt.title('Corona counts trend in the world')
ax.annotate(f'Sudden spike in cases ({count_spike_cases})', xy=(mdates.date2num(x[0]+tdelta(days=1)), 19600), xytext=(50, -20), textcoords='offset points', arrowprops=dict(arrowstyle='-|>'))
fig.autofmt_xdate()
plt.show()


# In[ ]:




