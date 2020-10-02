#!/usr/bin/env python
# coding: utf-8

# This is part of a [larger project](https://github.com/maxims94/life-tracking-project).

# # Preparation

# ## Read data

# In[ ]:


import pandas as pd
import numpy as np
import datetime as dt


# In[ ]:


input_file_path = '../input/life_data.csv'
output_interval_file_path = '../input/life_interval_data2.csv'
output_total_file_path = '../input/life_total_data.csv'


# In[ ]:


df = pd.read_csv(input_file_path,header=None,parse_dates=[[0,1]],infer_datetime_format=True)
df.columns = ['time','activity']
df.head()


# ## Fix dates

# There is a problem with the data: A time such as `2019-04-11 00:35` often refers to the day after (since it is after midnight)! This can be corrected by finding all dates that violate the chronological order and adding one day (see algorithm below).

# Before changing the data, a new column is introduced that keeps the original date. This way, no information is lost. It will later be useful to assign sleep duration to dates.

# In[ ]:


df['assigned_to'] = df.time.copy().map(lambda x: x.replace(hour=0,minute=0))
df.head()


# ### Algorithm
# 
# The algorithm below uses an inplace approach. It works with more than one faulty datetime -- such as having both `01:00,work` and `03:00,sleep` in one file. The reason is that adding one day to a faulty date creates another violation of chronological order that is caught in the next run of the loop. The only constraint is that the latest time of a day has to appear before the earliest of the next.

# In[ ]:


for i in range(len(df)):
    if i != 0 and df.iloc[i].time < df.iloc[i-1].time:
        print('Changed:', i)
        df.loc[i,'time'] += dt.timedelta(days=1)
df.head()


# ## df_i: Convert to intervals

# Convert the absolute times to intervals by calculating the differences. For instance, the two first rows...

# In[ ]:


df.drop('assigned_to',axis=1).head(2)


# ... should be converted to: `2019-05-01|15|prep`

# In[ ]:


df_i = pd.DataFrame(columns=['date','time','activity'])

for (i,row) in df.iterrows():
    if i != 0:
        add_row = pd.Series({
            'date':df.iloc[i-1].assigned_to,
            'time':row.time-df.iloc[i-1].time,
            'activity':df.iloc[i-1].activity,
            'start_time':df.iloc[i-1].time
        })
        df_i = df_i.append(add_row,ignore_index=True)

# Sanity check
print(len(df)==len(df_i)-1)

print(df_i.dtypes)
df_i.head()


# ## df_t: Sum up intervals

# Another useful view on the data is to store the total hours that were spent doing an activity, for every day and every activity.

# In[ ]:


unique_activities = df_i.activity.unique()

def by_activity(df):
    res = df.groupby('activity').sum()
    return pd.DataFrame([res.time.values],columns=res.index.values)

df_t = df_i.groupby('date').apply(by_activity)
df_t = df_t.fillna(dt.timedelta(0))
df_t = df_t.droplevel(1)
df_t.head()


# ## Write to files

# In[ ]:


def format_timedelta(td):
    # Hack for easy formatting
    return (td + pd.to_datetime('2000-01-01')).strftime('%H:%M')
    
print(format_timedelta(pd.to_timedelta('2h 30min')))


# In[ ]:


df_i_tmp = df_i.copy()
df_i_tmp.time = df_i_tmp.time.map(format_timedelta)
df_i_tmp.date = df_i_tmp.date.map(lambda x: x.strftime('%Y-%m-%d'))

#df_i_tmp.to_csv(output_interval_file_path,index=None)


# In[ ]:


#df_t.applymap(format_timedelta).to_csv(output_total_file_path)

