#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm


# In[ ]:


rootdir='/kaggle/input/ashrae-energy-prediction/'
train = pd.read_csv(rootdir + 'train.csv')
# convert timestamps
train['timestamp'] = pd.to_datetime(train['timestamp'],format='%Y-%m-%d %H:%M:%S')
# add log_meter_reading
train['log_meter_reading'] = np.log1p(train['meter_reading'])
train.head()


# **Averages**

# In[ ]:


def ave_before_date(df, date_value):
    dfsub = df.loc[df.timestamp<date_value,['log_meter_reading']]
    return np.nanmean(dfsub['log_meter_reading'])


# In[ ]:


def ave_after_date(df, date_value):
    dfsub = df.loc[df.timestamp>date_value,['log_meter_reading']]
    return np.nanmean(dfsub['log_meter_reading'])


# In[ ]:


date_value = datetime(2016,5,21)
print('Average BEFORE cut-off date...')
ave_bef = train.groupby(['building_id','meter']).apply(lambda x: ave_before_date(x, date_value)).to_frame()
ave_bef.columns=['before']
print('Average AFTER cut-off date...')
ave_aft = train.groupby(['building_id','meter']).apply(lambda x: ave_after_date(x, date_value)).to_frame()
ave_aft.columns=['after']
print('Merge...')
ave_total = pd.merge(ave_bef,ave_aft,on=['building_id','meter'])
ave_total.head()


# In[ ]:


# some time series do not have points for the entire year, so the average results in NaN's

# set nans to zero
for col in ave_total.columns:
    ave_total.loc[np.isnan(ave_total[col]), col] = 0


# In[ ]:


plt.figure(figsize=(8,6))
sns.scatterplot(ave_total.after, ave_total.before)
plt.grid();


# In[ ]:


plt.figure(figsize=(10,6))
sns.scatterplot(ave_total.after, ave_total.before)
plt.grid();
plt.xscale('log')
plt.yscale('log')
plt.xlim(0.001,15);
plt.ylim(0.001,15);


# In[ ]:


thr = 0.1 # arbitrary threshold
print(f'Series with average below {thr} before cut-off date: ' + str(sum(ave_total['before']<thr)))
print(f'Series with average below {thr} after cut-off date: ' + str(sum(ave_total['after']<thr)))


# In[ ]:


# rows where average is below threshold
ix_before = ave_total.loc[ave_total.before<thr].index.to_frame()
ix_before.head()


# In[ ]:


ix_after = ave_total.loc[ave_total.after<thr].index.to_frame()

#ix_after.head()


# **Count the number of time series with average below threshold by grouping them by meter**

# Most time series with average below threshold before cut-off date have meter=0, although some exist with other meter types

# In[ ]:


ix_before.meter.value_counts()


# There exist a few time series with average below threshold AFTER cut-off date

# In[ ]:


ix_after.meter.value_counts()


# In[ ]:


# time series with meter = 1,2, or 3
ix_before_not0meter=ix_before.loc[ix_before.meter>0]


# In[ ]:


k=-1


# Use the next cell to browse through the time series with average below threshold

# In[ ]:


k +=1
plt.figure(figsize=(8,6))
bid = ix_after.iloc[k,0]
mid = ix_after.iloc[k,1]
building_time_series = train.loc[(train.building_id==bid) & (train.meter==mid)]
bef_value = ave_total.loc[bid].loc[mid]['before']
aft_value = ave_total.loc[bid].loc[mid]['after']
plt.plot(building_time_series.timestamp, building_time_series.meter_reading);
plt.title('Building {:d} - Meter {:d} - Before {:.2f} After {:.2f}'.format(bid,mid,bef_value,aft_value));
plt.grid();


# In[ ]:


plt.figure(figsize=(8,5))
for col in ['before','after']:
    sns.distplot(ave_total[col],label=col,kde=False)
plt.grid();
plt.legend();
plt.xlabel('Average log meter');


# **Cleanup**

# In[ ]:


# copy dataframe
train_clean = train.copy()


# In[ ]:


# remove sections where before_average < threshold
for k in tqdm(range(len(ix_before))):
    bid = ix_before.iloc[k,0]
    mid = ix_before.iloc[k,1]
    ix_to_drop = train_clean[(train_clean.building_id == bid) & (train_clean.meter == mid) & (train_clean.timestamp < date_value)].index
    train_clean.drop(ix_to_drop, inplace = True)


# In[ ]:


# remove sections where after_average < threshold
for k in tqdm(range(len(ix_after))):
    bid = ix_after.iloc[k,0]
    mid = ix_after.iloc[k,1]
    ix_to_drop = train_clean[(train_clean.building_id == bid) & (train_clean.meter == mid) & (train_clean.timestamp < date_value)].index
    train_clean.drop(ix_to_drop, inplace = True)


# In[ ]:


# new length
print('New dataset length: ' + str(len(train_clean)))


# In[ ]:


# cleanup ratio
print('{:.1f}% of datapoints removed from original training dataset'.format(100*(1-len(train_clean)/len(train))))


# Export new dataset

# In[ ]:


train_clean.to_csv('train_cleanup01.csv',index=False)

