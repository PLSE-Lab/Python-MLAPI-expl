#!/usr/bin/env python
# coding: utf-8

# Thanks to this amazing memory-saving kernel  [here](https://www.kaggle.com/szelee/how-to-import-a-csv-file-of-55-million-rows), the below kernel was initially run on the full dataset, though the results still mostly hold for a small sample. If you follow that kernel and have plenty of memory (16G or more), you should be able to run this kernel on the entire training set (and probably with the addition of the test set as well).

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gc
gc.enable()


# In[ ]:


train_df = pd.read_csv('../input/train.csv',nrows=10e6)


# In[ ]:


#parsing time - straight from https://www.kaggle.com/szelee/how-to-import-a-csv-file-of-55-million-rows
train_df['pickup_datetime'] = train_df['pickup_datetime'].str.slice(0, 16)
train_df['pickup_datetime'] = pd.to_datetime(train_df['pickup_datetime'], utc=True, format='%Y-%m-%d %H:%M')


# In[ ]:


train_df.dtypes


# First see how many missing values there are

# In[ ]:


train_df.isna().sum().plot('bar')


# It looks like it's only the dropoff points we are missing - all the other values are there. Now just check that longitudes and longitudes are missing at the same time for all cases.

# In[ ]:


mss_long = train_df['dropoff_longitude'].isna()
mss_lat = train_df['dropoff_latitude'].isna()
print(f'Any missing values not occurring in pairs? {(mss_long!=mss_lat).any()}')
del mss_long, mss_lat


# So far so good. Now check let's give one last look to the rows with missing fields.

# In[ ]:


mss_df = train_df[train_df['dropoff_longitude'].isna()]
mss_df.head()


# So this gives us a hint that the missing values might be occurring together. Note how one set of pickup longitudes and latitudes here are zero - which doesn't make sense since we are talking about NYC and not somewhere off the coast of western Africa. Now on to outliers.

# In[ ]:


try:
    del mss_df
except NameError:
    pass
fares = train_df['fare_amount']
sns.boxplot(fares)


# Either some folks got a taxi to Latin America or that data's wrong. There seems to be 4 clear outliers here, so let's see the five top values for fares (note: this comment is based on the entire 55M rows).

# In[ ]:


fares.sort_values(ascending=False)[:5]


# Let's cap fares at a rather generous $5000 to remove the top 4 values and replot (again, based on the entire 55M rows).

# In[ ]:


plt.clf()
fares_capped = fares[fares<5000]
try:
    del fares
except NameError:
    pass
sns.boxplot(fares_capped)


# Still some clear outliers here, and it's possible that the .00 after the decimal points got confused with hundreds. We might need to do some more work to find a better cap for fares, but let's leave it at that for now and look at the other end.

# In[ ]:


plt.clf()
fares_capped.sort_values()[:5]


# While the negative values are clearly wrong, the minimum fare for NYC is $2.50, so anything below should count as an anomaly.

# In[ ]:


irreg_fares = train_df[(train_df['fare_amount']<2.5)]
display(irreg_fares.head())
print(f'there are {len(irreg_fares)} fare anomalies')


# Now assume anything under -$2.50 just had an extra minus sign in front.

# In[ ]:


red_irreg_fares = train_df[(train_df['fare_amount']<2.5)&(train_df['fare_amount']>-2.5)]
print(f'there are {len(red_irreg_fares)} fare anomalies still remaining')


# So there will need to be some further work in cleaning the fares up. On to the pickup times for now.

# In[ ]:


try:
    del irreg_fares, red_irreg_fares
except NameError:
    pass
pickup_times = train_df['pickup_datetime']
print(pickup_times.sort_values()[:10])
print(pickup_times.sort_values(ascending=False)[:10])


# Now check out the distribution of the volume of taxi trips.

# In[ ]:


pickup_dates = pickup_times.dt.date
volume_series = pickup_dates.groupby(pickup_dates).count()
fig, ax = plt.subplots(1, 1, figsize = (10, 5))
sns.distplot(volume_series)
ax.set_xlabel('trip volume')
ax.set_ylabel('density')
plt.show()
plt.clf()


# An interesting double-peak - or a bimodal distribution if you will. It'll also be interesting to see how the distribution of the test set data pans out. Let's throw in a hypothesis here (maybe a gut feel from the respective heights of the curves) that trip volume for weekends vs weekdays come from different distributions.

# In[ ]:


#in the pandas world, Monday=0 and Sunday=6, so anything <5 is a weekday
dayofweek = pd.to_datetime(pickup_dates).dt.dayofweek
weekday_pickup_dates = pickup_dates[dayofweek<5]
weekend_pickup_dates = pickup_dates[dayofweek>4]
weekday_volume_series = weekday_pickup_dates.groupby(weekday_pickup_dates).count()
weekend_volume_series = weekend_pickup_dates.groupby(weekend_pickup_dates).count()


# In[ ]:


fig, ax = plt.subplots(1, 1, figsize = (10, 5))
sns.distplot(weekday_volume_series)
ax.set_xlabel('weekday trip volume')
ax.set_ylabel('density')
plt.show()
plt.clf()
fig, ax = plt.subplots(1, 1, figsize = (10, 5))
sns.distplot(weekend_volume_series)
ax.set_xlabel('weekend trip volume')
ax.set_ylabel('density')
plt.show()
plt.clf()

del weekday_volume_series, weekend_volume_series
gc.collect()


# So while for the weekend trip volumes, the peak on the LHS is ever so slightly more pronounced, the weekday-weekend split doesn't fully explain the bimodality of the distribution, since the weekday volume plot still shows that sliver of a peak on the left (hint: can you think of what will?).
# 
# Why dig into trip volumes at all? Here's the logic: any time that demand outstrips supply for taxis in NYC in aggregate (hint: when would this happen?) and taxies are operating at 100% capacity, a greater number of taxi trips will mean a lower average fare. That's a lot of business logic strung together, so let's see if this assumed correlation holds up in the data.

# In[ ]:


fares = train_df['fare_amount'].copy()

#setting a lower cap right now to prevent distortion
fares[fares>1000] = 1000
mean_fares = fares.groupby(pickup_dates).mean()
print(f'Correlation between daily trip volume and mean fares is {mean_fares.corr(volume_series)}')


# As can be seen, the correlation is very weak right now, so how to strengthen this signal is another interesting route of attack.
# 
# Thanks for reading and see you in part II!
