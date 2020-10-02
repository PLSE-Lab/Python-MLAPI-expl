#!/usr/bin/env python
# coding: utf-8

# # Abstract
# The fastest route dataset is used to estimated the average speed of trips. This speed value is used to estimate the trip duration aswell. It turns out that this estimated trip duration is a better estimation for the trip duration than the total travel time variable of the fastest route dataset that is commonly used in models. A very simple LightGBM regressor is trained to verify that the 2 estimated features can improve a model's performance. This regressor scores 0.388 on the leaderboard. Without the estimated features that model scores 0.002 worse.

# # Introduction
# 
# oscarleo shared a pretty nice dataset about the fastest routes of the trips: https://www.kaggle.com/c/nyc-taxi-trip-duration/discussion/37033
# This dataset contains some features (total_distance, total_travel_time, number_of_steps) that improve the accuracy of a model by a pretty good amount. However such a valuable dataset may allow to extract more valuable features.
# 
# So this kernel focuses on estimating the speed a trip will travel along its shortest path. This will be done by averaging the speed of all trips that have used the respective street. The distance one trip has travelled on one street will be used as a weight such that as longer one trip was on a street as more will it contribute to the street's speed average.
# 
# It can be expected that the average speed a trip will have on one street depends on the time that trip happens. This will be explored first and so will the average speed on a street calculated for different groups of time.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime as dt

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler

sns.set()


# # Data preparation
# At first the data will be loaded, some necessary features added that will be needed later and at last merged with the fastest route dataset.

# In[ ]:


df_train = pd.read_csv('../input/new-york-city-taxi-with-osrm/train.csv')
df_test = pd.read_csv('../input/new-york-city-taxi-with-osrm/test.csv')


# In[ ]:


def add_features(data):
    data.loc[:, 'pickup_datetime'] = pd.to_datetime(data['pickup_datetime'])

    data.loc[:, 'pickup_time'] = data.pickup_datetime.apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second)
    data.loc[:, 'pickup_yearday'] = data.pickup_datetime.apply(lambda x: x.timetuple().tm_yday)
    data.loc[:, 'pickup_weekday'] = data.pickup_datetime.apply(dt.datetime.weekday)
    data.loc[:, 'pickup_yearweek'] = data.pickup_datetime.apply(lambda x: x.isocalendar()[1])
    data.loc[:, 'pickup_hour'] = np.floor(data.pickup_time / 3600)
    data.loc[:, 'holiday'] = data.pickup_yearday.apply(lambda x: 1 if x in [1, 18, 46, 151] else 0)
    
    return data
df_train = add_features(df_train)
df_test = add_features(df_test)


# In[ ]:


df_train_osrm1 = pd.read_csv('../input/new-york-city-taxi-with-osrm/fastest_routes_train_part_1.csv')
df_train_osrm2 = pd.read_csv('../input/new-york-city-taxi-with-osrm/fastest_routes_train_part_2.csv')
df_train_osrm = pd.concat([df_train_osrm1, df_train_osrm2])
df_test_osrm = pd.read_csv('../input/new-york-city-taxi-with-osrm/fastest_routes_test.csv')

df_train = pd.merge(df_train, df_train_osrm, how = 'left', on = 'id')
df_test = pd.merge(df_test, df_test_osrm, how = 'left', on = 'id') 


# In[ ]:


ind = df_train.total_travel_time > 0
print('{} rows without fastest route in training data'.format(len(df_train[ ~ind ])))
df_train.loc[ind, 'speed'] = df_train.loc[ind, 'total_distance'] / df_train.loc[ind, 'trip_duration']
df_train.loc[~ind, 'speed'] = 0


# # Exploration
# As mentioned earlier it is expected that the average speed depends on the time and that the average speed of trips on a street should be calculated for different time bins. However for accurate averages it's desired to get as many values as possible for every single time bin of a street. Therefore the bins should be chosen carefully.
# 

# In[ ]:


df_gby_d = df_train.loc[ind & (df_train.speed < 20), ['pickup_yearday', 'speed']].groupby('pickup_yearday').agg(np.mean).reset_index()

plt.figure(figsize=(16, 6))
plt.step(df_gby_d.pickup_yearday, df_gby_d.speed, where = 'mid')

plt.xlabel('day')
plt.ylabel('average speed [m/s]')
plt.xticks(range(0, 7*26, 7))
plt.ylim(0, 8)
plt.show()


# This plot shows a periodic behaviour of the average speeds. This is mostly likely caused by the respective day of the week. So it seems reasonable to calculate the average speed of a street not together for all days of a week.

# In[ ]:


df_gby_h = df_train.loc[ind, ['pickup_hour', 'pickup_weekday', 'speed']].groupby(['pickup_hour', 'pickup_weekday']).agg(np.mean).reset_index()

dict_weekday = {0: 'monday', 1: 'tuesday', 2: 'wednesday', 3: 'thursday', 
                4: 'friday', 5: 'saturday', 6: 'sunday'}
colors = ['black', 'purple', 'blue', 'cyan', 'magenta', 'orange', 'red']
plt.figure(figsize=(16, 6))
for d in range(7):
    plt.errorbar(x = df_gby_h[ df_gby_h.pickup_weekday == d ].pickup_hour + 0.5, 
                 y = df_gby_h[ df_gby_h.pickup_weekday == d ].speed, 
                 xerr = 0.5, marker = '', linestyle = '', label = dict_weekday[d], color = colors[d]
                )
plt.xlabel('hour')
plt.ylabel('average speed [m/s]')
plt.legend(loc = 0)
plt.ylim(0, 11)
plt.xticks(range(25))
plt.yticks(range(12))
plt.show()


# Reading this plot takes a little. But one can see that the distribution for saturday and sunday are similar and the distributions for the remaining days are a little similar aswell. However tuesday, wednesday and thursday are closer to each other than they are to monday and friday. This can be partly explained by the fact that the only holidays in the first half of 2016 were on monday and on friday. At least these are the days the holidays package returns:  https://www.kaggle.com/c/nyc-taxi-trip-duration/discussion/37192. 
# 
# The evening (after 20:00) is different though. One would expect the distributions of friday and saturday are similar and that the distributions of the remaining ones are also similar. At least this is true for the first statement.
# 
# However calculating the average speed of each street for each hour bin for each day of a week would be very much so this will be reduced further. At first there will only 12 hour bins taken. The next plot will show that the information of the distributions differences is still very well contained. The second next plot will show that the average speed of mondays and fridays that were holidays (1st, 18th, 46th and 151th day of the year) is rather comparable to the weekend.
# 
# So the average speed of a street will be calculated in 12 different hour bins and 2 bins that represent whether the respective day is a working day or not. At least as long as the trip started before 20:00. If it started later the trip will only be categorized as a trip on a day without work if the next day is a saturday, sunday or holiday. 
# 
# Furthermore all trips with speed larger than 20 m/s (72 km/h, 45 miles/h) will not be considered for the speed average calculations of a street. I don't know the actual speed limits in NYC but these can most likely be considered as wrong data.

# In[ ]:


df_train.loc[:, 'pickup_hour_bin'] = np.floor(df_train.pickup_time / (2*3600))
df_test.loc[:, 'pickup_hour_bin'] = np.floor(df_test.pickup_time / (2*3600))
df_gby_h = df_train.loc[ind & (df_train.holiday == 0), ['pickup_hour_bin', 'pickup_weekday', 'speed']].groupby(['pickup_hour_bin', 'pickup_weekday']).agg(np.mean).reset_index()

plt.figure(figsize=(16, 6))
for d in range(7):
    plt.errorbar(x = df_gby_h[ df_gby_h.pickup_weekday == d ].pickup_hour_bin * 2 + 1, 
                 y = df_gby_h[ df_gby_h.pickup_weekday == d ].speed, 
                 xerr = 1, marker = '', linestyle = '', label = dict_weekday[d], color = colors[d]
                )
plt.xlabel('hour')
plt.ylabel('average speed [m/s]')
plt.legend(loc = 0)
plt.ylim(0, 11)
plt.xticks(range(25))
plt.yticks(range(12))
plt.show()


# These distributions still contain the differences very well.

# In[ ]:


df_gby_wh = df_train.loc[ind & (df_train.speed < 20), ['pickup_weekday', 'holiday', 'speed']].groupby(['pickup_weekday', 'holiday'])['speed'].agg([np.mean, np.std]).reset_index()

plt.figure(figsize=(16, 6))
(_, caps, _) = plt.errorbar(data = df_gby_wh[ df_gby_wh.holiday == 1 ], 
            x = 'pickup_weekday', y = 'mean', yerr = 'std', capsize = 10,
            marker = 'o', linestyle = '', label = 'holiday', color = 'blue')
for cap in caps:
    cap.set_markeredgewidth(1)
    
(_, caps, _) = plt.errorbar(data = df_gby_wh[ df_gby_wh.holiday == 0 ], 
            x = 'pickup_weekday', y = 'mean', yerr = 'std', capsize = 10,
            marker = 'o', linestyle = '', label = 'no holiday', color = 'green')
for cap in caps:
    cap.set_markeredgewidth(1)
    
plt.axhline(y=np.mean(df_train.loc[ind & (df_train.speed < 20) & (df_train.holiday == 1), 'speed']), 
            color='blue', linestyle='dashed', alpha = 0.5)
plt.axhline(y=np.mean(df_train.loc[ind & (df_train.speed < 20) & (df_train.holiday == 0), 'speed']),  
            color='green', linestyle='dashed', alpha = 0.5)

plt.ylabel('average speed [m/s]')
plt.xticks([k for k in dict_weekday.keys()], [v for v in dict_weekday.values()])
plt.ylim(0, 10)
plt.legend(loc = 3)
plt.show()


# The average speed of mondays and fridays that were holidays are closer to the average speed of sundays. Surprisingly even larger though the standard deviations are very large.

# In[ ]:


def add_work(data):
    for i in range(len(data)):
        # not evening 
        if data.at[i, 'pickup_hour'] < 20:
            data.at[i, 'work'] = 1 if ((data.at[i, 'pickup_weekday'] < 5) & 
                                       (data.at[i, 'holiday'] == 0)
                                      ) else 0
        # evening
        else: 
            # is next day a holiday?
            if (data.at[i, 'pickup_yearday'] + 1) in [1, 18, 46, 151]:
                data.at[i, 'work'] = 0
            else:
                if data.at[i, 'pickup_weekday'] < 6:
                    data.at[i, 'work'] = 0 if ((data.at[i, 'pickup_weekday'] == 4) |
                                               (data.at[i, 'pickup_weekday'] == 5)
                                              ) else 1
                else:
                    data.at[i, 'work'] = 1
    return data

df_train = add_work(df_train)
df_test = add_work(df_test)


# In[ ]:


df_gby_wh = df_train.loc[ind, ['work', 'pickup_hour_bin', 'speed']].groupby(['work', 'pickup_hour_bin']).agg(np.mean).reset_index()

plt.figure(figsize=(16, 6))
plt.errorbar(x = df_gby_wh[ df_gby_wh.work == 1 ].pickup_hour_bin * 2 + 1, xerr = 1,
             y = df_gby_wh[ df_gby_wh.work == 1 ].speed, 
             marker = '', linestyle = '', label = 'work', color = 'blue')
plt.errorbar(x = df_gby_wh[ df_gby_wh.work == 0 ].pickup_hour_bin * 2 + 1, xerr = 1,
             y = df_gby_wh[ df_gby_wh.work == 0 ].speed, 
             marker = '', linestyle = '', label = 'no work', color = 'green')

plt.axhline(y=np.mean(df_train.loc[ind & (df_train.speed < 20) & (df_train.work == 1), 'speed']), 
            color='blue', linestyle='dashed', alpha = 0.5)
plt.axhline(y=np.mean(df_train.loc[ind & (df_train.speed < 20) & (df_train.work == 0), 'speed']),  
            color='green', linestyle='dashed', alpha = 0.5)

plt.xlabel('hour')
plt.ylabel('average speed')
plt.ylim(0, 10)
plt.xticks(range(25))
plt.legend()
plt.show()


# # Calculate the average speed of a street

# Create a DataFrame with one row per unique street, pickup_hour_bin and work category. This will be done by averaging the speed (calculated by trip_duration and total_travel_time) of all trips that belong to a specific street/pickup_hour_bin/work value. The distance the respective trip travelled on the particular street will be used as a weight for averaging.

# In[ ]:


df_speed_by_hour_bin_day = df_train[['pickup_hour_bin', 'work', 'speed']].groupby(['pickup_hour_bin', 'work']).agg(np.mean)

lines =  df_train.loc[ind].street_for_each_step.apply(lambda x: str(x).split(sep="|"))
streets = [item for l in lines for item in l]
print('{} unique streets used in training data'.format(len(set(streets))))

df_speed_by_hour_bin_day_street = pd.DataFrame(index = df_speed_by_hour_bin_day.index)
df_empty = df_speed_by_hour_bin_day_street.copy()
df_speed_by_hour_bin_day_street['street'] = ''
df_speed_by_hour_bin_day_street.set_index('street', append=True, inplace=True)

for i in set(streets):
    df_i = df_empty.copy()
    df_i['street'] = i
    df_i = df_i.set_index('street', append = True)
    df_speed_by_hour_bin_day_street = pd.concat([df_speed_by_hour_bin_day_street, df_i])
    
df_speed_by_hour_bin_day_street = df_speed_by_hour_bin_day_street[ df_speed_by_hour_bin_day_street.index.get_level_values(2) != '' ]
df_speed_by_hour_bin_day_street['speed'] = 0
df_speed_by_hour_bin_day_street.head()  


# In[ ]:


street_dict = {}
for i, s in enumerate(set(streets)):
    street_dict[s] = i

list_speed_by_tds = [0.0] * 12 * 2 *len(set(streets))
list_speed_by_tds_count = [0.0] * 12 * 2 *len(set(streets))

for i, row in enumerate(df_train.loc[ind & (df_train.speed < 20)].itertuples()):
    speed = row.speed
    weights = [float(k) for k in row.distance_per_step.split(sep="|")]
    sum_weight = sum(weights)
        
    if sum_weight > 0:
        for j, s in enumerate(row.street_for_each_step.split(sep="|")):
            index = int(street_dict[str(s)] * 24 + row.pickup_hour_bin * 2 + row.work)
            list_speed_by_tds[index] += speed * weights[j] / sum_weight
            list_speed_by_tds_count[index] += 1 * weights[j] / sum_weight
    if i % 300000 == 0:
        print(str(i) + ' rows iterated')
print('finished')

df_speed_by_hour_bin_day_street.loc[:, 'speed'] = [i/j if j > 0 else 0.0 for i,j in zip(list_speed_by_tds, list_speed_by_tds_count)]


# In[ ]:


print('{} rows'.format(len(df_speed_by_hour_bin_day_street)))
print('{} filled rows'.format(len(df_speed_by_hour_bin_day_street[ df_speed_by_hour_bin_day_street.speed > 0 ])))
df_speed_by_hour_bin_day_street[ df_speed_by_hour_bin_day_street.speed > 0 ].head()


# There are a lot of street/hour bin/work bins that are still empty. This isn't too much of a problem due to the fact that the trips in the test dataset won't need this values very often. Nonetheless these empty bins will be dealt with.
# 
# So at first the calculated averaged speeds will be plotted to make sure that something didn't went very wrong.

# In[ ]:


plt.figure(figsize=(16, 6))
df_speed_by_hour_bin_day_street[ (df_speed_by_hour_bin_day_street.speed > 0) &
                                 (df_speed_by_hour_bin_day_street.index.get_level_values(1) == 0 )
                                ].speed.hist(bins = 60, alpha = 1.0, color = 'green', 
                                             histtype = 'step', label = 'no work', normed = True)
df_speed_by_hour_bin_day_street[ (df_speed_by_hour_bin_day_street.speed > 0) &
                                 (df_speed_by_hour_bin_day_street.index.get_level_values(1) == 1 )
                                ].speed.hist(bins = 60, alpha = 1.0, color = 'blue', 
                                             histtype = 'step', label = 'work', normed = True)
plt.legend()
plt.xlabel('speed [m/s]')
plt.ylabel('normed count (a.u.) / (1/3 m/s)')
plt.show()

plt.figure(figsize=(16, 6))
df_gby_hw = df_speed_by_hour_bin_day_street[ df_speed_by_hour_bin_day_street.speed > 0 ].groupby(['pickup_hour_bin', 'work']).agg(np.mean)

df_gby_hw_work = df_gby_hw[ df_gby_hw.index.get_level_values(1) == 1].reset_index()
df_gby_hw_nowork = df_gby_hw[ df_gby_hw.index.get_level_values(1) == 0].reset_index()

plt.errorbar(x = df_gby_hw_work.pickup_hour_bin*2+1, y = df_gby_hw_work.speed, marker='', 
            xerr = 1, color = 'blue', label = 'work', linestyle = '')
plt.axhline(y=np.mean(df_gby_hw_work.speed), color='blue', linestyle='dashed', alpha = 0.5)

plt.errorbar(x = df_gby_hw_nowork.pickup_hour_bin*2+1, y = df_gby_hw_nowork.speed, marker='', 
            xerr = 1, color = 'green', label = 'no work', linestyle = '')
plt.axhline(y=np.mean(df_gby_hw_nowork.speed), color='green', linestyle='dashed', alpha = 0.5)

plt.legend(loc = 3)
plt.xlabel('hour of day')
plt.ylabel('average speed [m/s]')
plt.xticks(np.linspace(0,24,25))
plt.yticks(range(11))
plt.ylim(0, 10)
plt.show()


# As mentioned earlier there are a lot of street/hour bin/work bins without an actual value. These missing values will now be estimated. At first the value of the other work category is used. If that one is missing aswell the mean of the different hour bins is used. If that isn't possible the mean of all streets is used.
# 
# Instead of using the mean of all hour bins in the second step it would most likely improve the result if just the mean of the adjacent hour bins would be used. 

# In[ ]:


df_speed_by_street = df_speed_by_hour_bin_day_street[ df_speed_by_hour_bin_day_street.speed > 0 ].groupby('street').agg(np.mean)

df_speed_by_hour_bin_street = df_speed_by_hour_bin_day_street[ df_speed_by_hour_bin_day_street.speed > 0 ].groupby(['pickup_hour_bin', 'street']).agg(np.mean)
street_mean = np.mean(df_speed_by_hour_bin_day_street[ df_speed_by_hour_bin_day_street.speed > 0 ])

print('{} rows without speed value'.format(len(df_speed_by_hour_bin_day_street[ df_speed_by_hour_bin_day_street.speed == 0 ])))
indices = set(df_speed_by_hour_bin_street.index)
df_speed_by_hour_bin_day_street.speed = df_speed_by_hour_bin_day_street.apply(lambda x: df_speed_by_hour_bin_street.loc[x.name[0], x.name[2]].speed 
                                              if ((x.name[0], x.name[2]) in indices) & (x.speed == 0) else x.speed, axis = 1)

print('{} rows without speed value'.format(len(df_speed_by_hour_bin_day_street[ df_speed_by_hour_bin_day_street.speed == 0 ])))

indices = set(df_speed_by_street.index)
df_speed_by_hour_bin_day_street.speed = df_speed_by_hour_bin_day_street.apply(lambda x: df_speed_by_street.loc[x.name[2]].speed
                                             if (x.name[2] in indices) & (x.speed == 0) else x.speed, axis = 1)
print('{} rows without speed value'.format(len(df_speed_by_hour_bin_day_street[ df_speed_by_hour_bin_day_street.speed == 0 ])))

df_speed_by_hour_bin_day_street.speed = df_speed_by_hour_bin_day_street.apply(lambda x: street_mean.values
                                             if x.speed == 0 else x.speed, axis = 1)
print('{} rows without speed value'.format(len(df_speed_by_hour_bin_day_street[ df_speed_by_hour_bin_day_street.speed == 0 ])))


# In[ ]:


df_speed_by_hour_bin_day_street['speed'] = df_speed_by_hour_bin_day_street.speed.apply(lambda x: float(x)) 
df_street_loc = pd.DataFrame(df_speed_by_hour_bin_day_street.groupby('street').agg(np.mean))
#df_street_loc.set_index('street', drop=True, inplace=True)

df_street_loc['longitude'] = 0.0
df_street_loc['latitude'] = 0.0

t0 = dt.datetime.now()
for i in range(len(df_train.head(n=50000))):
    l = df_train.iloc[i].step_location_list.split('|')
    for j, s in enumerate(df_train.iloc[i].street_for_each_step.split('|')):
        df_street_loc.loc[s,'longitude'] = float(l[j].split(',')[0])
        df_street_loc.loc[s,'latitude'] = float(l[j].split(',')[1])

t1 = dt.datetime.now()
print('Spent time: ' + str(t1-t0))

from matplotlib.colors import ListedColormap

cmap = ListedColormap(sns.cubehelix_palette(8))

plt.scatter(data = df_street_loc[ (df_street_loc.speed > 4) & (df_street_loc.speed < 16)], 
            x = 'longitude', y = 'latitude', marker = '.', 
            c = 'speed', cmap = cmap, alpha = 1)
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.xlim(-74.03, -73.77)
plt.ylim(40.63, 40.85)
plt.show()


# This plot shows the average speed of the streets. As brighter the point as lower is the average speed. As expected the average speed of streets in Manhattan are smallest and get larger as further away from Manhattan the street is. Please note that only a single (basically randomly selected) point is drawn for each street. So one doesn't see a nice pattern of NYC as one would see by drawing the pickup and dropoff locations.

# # Estimating the average speed of a trip
# The calculated average speed of a street will now be used to estimate the speed of all trips in the training and test datasets. Similar to before the distance fraction one trip has travelled on a specific street will be used as a weight for estimating the total average speed of the respective trip. 
# 
# It will be taken care of cases where the average speed of a street isn't available such that the distance on that street is subtracted from the total distance of the trip for the weight calculation. This can happen if either no trip in the training data used this street or the rows in the training data were cut because its speed was above 20 m/s.

# In[ ]:


from itertools import compress
df_train = df_train[ ~pd.isnull(df_train.distance_per_step) ]

indices = set(df_speed_by_hour_bin_day_street.index)
street_speed = df_speed_by_hour_bin_day_street.speed.values

def extract_speed(x):
    we = [float(d) for d in x.distance_per_step.split(sep="|")]
    st = [s for s in x.street_for_each_step.split(sep="|")]
    bo = [(x.pickup_hour_bin, x.work, s) in indices for s in st]
    
    if (len(we) == len(st)):
        we = list(compress(we, bo))
        st = list(compress(st, bo))
        sum_we = sum(we)
        sp = 0.0
        if sum_we > 0:
            for w, s in zip(we, st):
                if (x.pickup_hour_bin, x.work, s) in indices:
                    index = int(street_dict[s] * 24 + x.pickup_hour_bin * 2 + x.work)
                    sp += street_speed[index] * w / sum_we
                else:
                    print('Cannot find indices, this shouldnt happen.')
        return sp
    else:
        print('Lengths are not equal, this shouldnt happen.')
        return -1.0

print('Starting')
t0 = dt.datetime.now()
train_estimated_speed = df_train[['pickup_hour_bin', 'work', 'distance_per_step', 
                                  'street_for_each_step']].apply(lambda x: extract_speed(x), 
                                                                 axis = 1)
df_train['estimated_speed'] = train_estimated_speed
print('training sample finished')
t1 = dt.datetime.now()
print('Spent time: ' + str(t1-t0))


# In[ ]:


print('Starting')
t0 = dt.datetime.now()
test_estimated_speed = df_test[['pickup_hour_bin', 'work', 'distance_per_step', 
                                'street_for_each_step']].apply(lambda x: extract_speed(x), 
                                                               axis = 1)
df_test['estimated_speed'] = test_estimated_speed
    
t1 = dt.datetime.now()
print('Spent time: ' + str(t1-t0))


# In[ ]:


df_train[['speed', 'estimated_speed']].head()


# # Exploring the estimated speed
# The estimated speed will now be compared with the speed (calculated by the trip_duration and the total_distance variables). In addition the estimated speed distributions of the training and test data will be compared to each other.

# In[ ]:


plt.scatter(data = df_train, x = 'speed', y = 'estimated_speed', marker = '.', alpha = 0.3)
plt.xlabel('speed')
plt.ylabel('estimated speed')
plt.xlim(0, 20)
plt.ylim(0, 20)
plt.show()


# In[ ]:


df_train.estimated_speed.hist(bins = 60, normed=True, alpha=0.5)
df_test.estimated_speed.hist(bins = 60, normed=True, alpha=0.5)
plt.yscale('log')
plt.xlabel('estimated speed')
plt.ylabel('log normed count (a.u.) / (1/3 m/s)')
plt.show()


# ## Estimating trip duration
# The estimated speed together with the total distance (of the fastest route dataset) allows to estimated the trip duration aswell. This estimated trip duration actually scores ~0.55 on the leaderboard without any machine learning at all.

# In[ ]:


ind = df_train.estimated_speed > 0
df_train.loc[ind, 'estimated_trip_duration'] = df_train.total_distance.loc[ind] / df_train.estimated_speed.loc[ind]
df_train.loc[~ind, 'estimated_trip_duration'] = 0

ind = df_test.estimated_speed > 0
df_test.loc[ind,'estimated_trip_duration'] = df_test.total_distance.loc[ind] / df_test.estimated_speed.loc[ind]                                    
df_test.loc[~ind,'estimated_trip_duration'] = 0


# In[ ]:


df_train[['id', 'estimated_speed', 'estimated_trip_duration']].to_csv('train_estimated_speed.csv')
df_test[['id', 'estimated_speed', 'estimated_trip_duration']].to_csv('test_estimated_speed.csv')


# In[ ]:


plt.scatter(data = df_train, x = 'trip_duration', y = 'estimated_trip_duration', marker = '.')
plt.xlabel('trip duration')
plt.ylabel('estimated trip duration')
plt.xlim(0, 16000)
plt.ylim(0, 16000)
plt.show()


# In many cases the trip duration is much larger than the estimated trip duration. This isn't surprising. But cases where the opposite is true are surprising. But in that cases the actual speed is very unrealistic:

# In[ ]:


df_train.loc[((df_train.estimated_trip_duration > 5000) & (df_train.trip_duration < 2000)) |
             ((df_train.estimated_trip_duration > 3000) & (df_train.trip_duration < 1000)),
             ['vendor_id', 'passenger_count', 'estimated_trip_duration', 'trip_duration',
              'total_distance', 'speed']]


# ## Comparing total_travel_time with estimated_trip_duration
# Next the estimated trip duration will be compared with the total travel time (from the fastest route dataset) to ensure all this work wasn't for nothing. In such a case the linear dependency would have a slope of 1. 

# In[ ]:


plt.figure(figsize=(7, 7))
plt.plot(np.linspace(0, 100000, 10000), np.linspace(0, 100000, 10000), '--', 
         linewidth=2, color = 'black', alpha = 0.5)
plt.scatter(data = df_train[ df_train.estimated_speed > 0 ],
            x = 'total_travel_time', y = 'estimated_trip_duration',
            marker = '.')

plt.xlabel('total travel time')
plt.ylabel('estimated trip duration')
plt.xlim(0, 10000)
plt.ylim(0, 10000)
plt.show()


# Luckily the slope is clearly different from 1. The next plot will compare which one of those 2 features models the real trip duration better.

# In[ ]:


plt.figure(figsize=(16, 8))
#plt.plot(np.linspace(0, 100000, 10000), np.linspace(0, 100000, 10000), '--', 
#         linewidth=2, color = 'black', alpha = 0.5)
df_temp = df_train[ df_train.estimated_speed > 0 ]

plt.subplot(121)
plt.scatter(x = df_temp.trip_duration,
            y = df_temp.total_travel_time,
            marker = '.', label = 'total travel time', alpha = 0.2)
plt.xlabel('trip duration')
plt.ylabel('total travel time')
plt.xlim(0, 10000)
plt.ylim(0, 10000)

plt.subplot(122)
plt.scatter(x = df_temp.trip_duration,
            y = df_temp.estimated_trip_duration,
            marker = '.', label = 'estimated trip duration', alpha = 0.2)
plt.xlabel('trip duration')
plt.ylabel('estimated trip duration')
plt.xlim(0, 10000)
plt.ylim(0, 10000)

#plt.legend()
plt.show()


# Ideally the slope of the linear dependency of these 2 plots would be 1 again. As it turns out the estimated trip duration is closer to this than the total travel time is. 

# # Building a simple model
# To verify the generated features estimated_speed and estimated_trip_duration can increase a model's performance a LightGBM regressor will be used with just a very small amount of features. This model will be trained with and without the engineered features and their scores compared afterwards.
# Due to the fact that rmsle is used for evaluation the logarithm of the trip duration will be used as label for training.

# In[ ]:


for c in df_train.columns:
    if df_train[c].isnull().values.any():
        print('Column {} contains NaNs'.format(c))
        indices = df_train[c].isnull()
        mean = np.mean(df_train[ ~indices ][c])
        print('mean: ' + str(mean))
        print('n: ' + str(len(df_train.loc[indices])))
        df_train.loc[indices, c] = 0#mean


# In[ ]:


df_train['log_trip_duration'] = np.log(df_train['trip_duration'] + 1)
                                                
features = ['vendor_id', 'passenger_count', 'holiday'
           ,'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude'
           ,'pickup_time', 'pickup_yearday', 'pickup_weekday', 'pickup_yearweek'
           ,'number_of_steps', 'total_distance', 'total_travel_time'
           ,'estimated_speed', 'estimated_trip_duration'
           ]
label = ['log_trip_duration']

X_train, X_test, y_train, y_test = train_test_split(df_train[features], df_train[label], 
                                                    test_size=0.3, random_state=42)


# In[ ]:


def rmsle(predicted, real):
    sum=0.0
    for x in range(len(predicted)):
        p = np.log(predicted[x]+1)
        r = np.log(real[x]+1)
        sum = sum + (p - r)**2
    return (sum/len(predicted))**0.5

model = LGBMRegressor(objective='regression', boosting_type = 'gbdt', metric = 'rmse', 
                      max_depth = -1,  learning_rate = 0.05, n_estimators = 500, num_leaves = 160, 
                      min_data_in_leaf = 700, max_bin = 511
                     )
pl = Pipeline(steps=[('scaler', RobustScaler()), ('regressor', model)])


# In[ ]:


# train
t0 = dt.datetime.now()
pl.fit(X_train[features], y_train[label].values.ravel())
t1 = dt.datetime.now()
print('Spent time for training: ' + str(t1-t0))

# predict
p = np.exp(pl.predict(X_test)) - 1
if len(p[ p < 0 ]) > 0:
    print('{} negative predictions'.format(len(p[ p < 0 ])))
    p[ p < 0 ] = 0
score = rmsle(p, np.exp(np.array(y_test[label])) - 1)

print('Score: ' + str(score))

# plot feature importances
fig = plt.figure(figsize=(8, 6))

feature_importance = model.feature_importances_
sorted_idx = np.argsort(feature_importance)

plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), np.array(features)[sorted_idx])
plt.xlabel('Importance')
plt.title('Feature importance')
plt.draw()
plt.show()


# In[ ]:


for dont_use in [['total_travel_time'], ['estimated_speed', 'estimated_trip_duration'], 
                 ['estimated_speed'], 'estimated_trip_duration']:
    print('training without ' + str(dont_use))
    new_features = [f for f in features if f not in dont_use]
    
    t0 = dt.datetime.now()
    pl.fit(X_train[new_features], y_train[label].values.ravel())
    t1 = dt.datetime.now()
    print('Spent time for training: ' + str(t1-t0))
    
    p = np.exp(pl.predict(X_test[new_features])) - 1
    if len(p[ p < 0 ]) > 0:
        print('{} negative predictions'.format(len(p[ p < 0 ])))
        p[ p < 0 ] = 0
    score = rmsle(p, np.exp(np.array(y_test[label])) - 1)

    print('Score: ' + str(score))


# ## Predict the test dataset

# In[ ]:


t0 = dt.datetime.now()
pl.fit(df_train[features], df_train[label].values.ravel())
t1 = dt.datetime.now()
print('Spent time for training: ' + str(t1-t0))

p = np.exp(pl.predict(df_test[features])) - 1
if len(p[ p < 0 ]) > 0:
    print('{} negative predictions'.format(len(p[ p < 0 ])))
    p[ p < 0 ] = 0
           
result = pd.DataFrame(df_test['id'])
result.set_index('id', inplace=True)
result['trip_duration'] = (p)

result.to_csv('submission.csv')
result.head()


# # Discussion
# Even without cross-validation the probability is large that the features estimated_speed and estimated_trip_duration does improve the score of a model. It also turns out that most likely not both the estimated speed as well as the estimated trip duration aren't needed. This isn't surprising due to the fact that there the estimated trip duration is just calculated by using the estimated speed as well as the total distance. This very simple model with just 16 features scores 0.388 on the leaderboard.
# 
# The hyperparameters weren't actually tuned for this small set of features so there might still be room for improvement. It might also be worth exploring if using a different binning than the hour bin/work binning used in this kernel improves the result.

# In[ ]:




