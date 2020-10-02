#!/usr/bin/env python
# coding: utf-8

# # NYC Taxi Fares: Exploration, Insights, Models
# *By: Ian Chu Te (dinosaur)*
# <br/>
# <img style="height:300px; margin-left:0;" src="https://upload.wikimedia.org/wikipedia/commons/e/ef/NYC_Taxi_Ford_Crown_Victoria.jpg"/>
# 
# ## Key Insights:
# ### 1. Taxi fares are distributed log-normally
# > **Log-normal distribution parameters (MLE fit - 1M samples):**<br>
# > - &mu;: 2.21<br>
# > - &sigma;: 0.60<br>
# 
# > **Parameters mapped back to original dollar space:**
# > - exp(&mu;): USD 9.17<br>
# > - exp(&sigma;): USD 1.82
# 
# ### 2. Overwhelming majority of rides (98%) have pick-up and drop-off points from and to a specific central location
# > **Central Location:**<br>
# > - Longitude: from -76 to -72<br>
# > - Latitude: from 39 to 42.25<br>
# 
# ### 3. Some sections of pick-up/drop-off locations have higher fare rates
# 
# ### 4. An almost-linear relationship exists between Manhattan distance from pick-up to drop-off and fare amount
# >**Linear Models CV-MAE:**<br>
# > - Ridge Regression: 2.85<br>
# > - Isotonic Regression: 2.59
# 
# ### 5. Overwhelming majority of rides (98%) have one to six passengers
# > **Breakdown:**<br>
# > - 1 passenger: 69%<br>
# > - 2 passengers: 14%<br>
# > - 3 passengers: 4%<br>
# > - 4 passengers 2%<br>
# > - 5 passengers: 7%<br>
# > - 6 passengers: 2%
# 
# ### 6. A piecewise-linear relationship exists between number of passengers and fare amount
# >**Linear Model CV-MAE:**<br>
# > - Ridge Regression: 4.56<br>
# > - Isotonic Regression: 3.7
# 
# ### 7. Massive taxi fare hike on September 2012 (<a href="https://www.nytimes.com/2012/09/04/nyregion/new-york-taxis-to-start-charging-increased-rates.html">click here to go to related news article</a>)
# > - Median fare increase: USD 1.40
# 
# ### 8. Low-fare season: rides on January, February and March have lower fares than rest of the months
# > - Median fare decrease: USD 0.40
# 
# ### 9. Peak and off-peak hours
# > Mean fare increase: USD 0.85
# 
# > **Peak hours:**<br>
# > - 12mn - 2am (from bars)<br>
# > - 10am - 1pm (commute to work)<br>
# > - 6pm (commute back home)
# 
# > **Off-peak hours:**<br>
# > - 3am - 6am (unholy hours)<br>
# > - 5pm (just finishing work)<br>
# > - 7pm - 8pm (everyone having dinner)<br><br>
# 
# 

# In[ ]:


import gc
import numpy as np
import pandas as pd
import scipy.ndimage
import seaborn as sns
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot as plt


# In[ ]:


plt.rcParams['figure.figsize'] = (12, 8)
plt.style.use('seaborn-white')


# In[ ]:


cols = ['fare_amount', 'pickup_datetime', 'pickup_longitude',
       'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
       'passenger_count']
train = pd.read_csv('../input/train.csv', usecols=cols, engine='c')
test = pd.read_csv('../input/test.csv', usecols=cols[1:], engine='c')


# *Round longitude and latitudes - higher precision might result in better accuracy:*

# In[ ]:


float_cols = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']
train[float_cols] = np.round(train[float_cols].astype('float16'), 2)
test[float_cols] = np.round(test[float_cols].astype('float16'), 2)
gc.collect()


# *Prepare date features:*

# In[ ]:


train['year'] = train.pickup_datetime.str[:4].astype('uint16')
test['year'] = test.pickup_datetime.str[:4].astype('uint16')
gc.collect()

train['month'] = train.pickup_datetime.str[5:7].astype('uint8')
test['month'] = test.pickup_datetime.str[5:7].astype('uint8')
gc.collect()

train['day'] = train.pickup_datetime.str[8:10].astype('uint8')
test['day'] = test.pickup_datetime.str[8:10].astype('uint8')
gc.collect()

train['hour'] = train.pickup_datetime.str[11:13].astype('uint8')
test['hour'] = test.pickup_datetime.str[11:13].astype('uint8')
gc.collect()

# train['minute'] = train.pickup_datetime.str[14:16].astype('uint8')
# test['minute'] = test.pickup_datetime.str[14:16].astype('uint8')
# gc.collect()

# train['second'] = train.pickup_datetime.str[17:19].astype('uint8')
# test['second'] = test.pickup_datetime.str[17:19].astype('uint8')
# gc.collect()

train = train.drop('pickup_datetime', axis=1)
test = test.drop('pickup_datetime', axis=1)
gc.collect()


# # A. Preview

# ## A1. [Train] Row count

# In[ ]:


len(train)


# ## A2. [Test] Row count

# In[ ]:


len(test)


# ## A3. [Train] First few rows

# In[ ]:


train.head()


# ## A4. [Test] First few rows

# In[ ]:


test.head()


# ## A5. Sampling
# *Train row count is too high; take only 25% uniform random sample of the data:*

# In[ ]:


# train = train.sample(frac=0.25).reset_index(drop=True)
# gc.collect()


# # B. Target Analysis

# In[ ]:


Y = train.fare_amount


# ## B1. Top value frequencies

# In[ ]:


Y.value_counts().head(10)


# *Using the mode as general prediction we get an error of 2.4 dollars:*

# In[ ]:


from sklearn.metrics import median_absolute_error, make_scorer
print('MAE Error (in USD):', median_absolute_error(Y, np.ones_like(Y.values) * 6.5))


# *Using the second mode as general prediction:*

# In[ ]:


print('MAE Error (in USD):', median_absolute_error(Y, np.ones_like(Y.values) * 4.5))


# ## B2. Histograms

# *Histogram:*

# In[ ]:


_ = Y.plot.hist(100, color='teal')


# *Log histogram:*

# In[ ]:


_ = np.log1p(Y).plot.hist(bins=100, color='teal')


# ### [Key Insight] Fare amounts are distributed log-normally:

# In[ ]:


# Credit: https://stackoverflow.com/questions/8747761/scipy-lognormal-distribution-parameters
# commented out due to slowness
# from scipy.stats import lognorm
# shape, loc, scale = lognorm.fit(Y.sample(1000000).values, loc=0)

shape, loc, scale = 0.6009456315880513, -0.01272662981718287, 9.165648701197032
print('--Log-normal Distribution--')
print('\tmu:', np.log(scale))
print('\tsigma:', shape)
print('\t[linear space / in dollars] mu:', scale)
print('\t[linear space / in dollars] sigma:', np.exp(shape))


# *Rounded integer histogram (0 to 30 dollars):*

# In[ ]:


_ = np.round(Y[(Y>=0)&(Y<=30)]).astype(int).hist(bins=30, color='teal')


# ## B3. Statistics

# *General statistics:*

# In[ ]:


np.round(Y.describe(), 2)


# *Using the mean as general prediction:*

# In[ ]:


mean = Y.mean()
print('Arithmetic mean:', mean)
print('MAE Error (in USD):', median_absolute_error(Y, mean * np.ones_like(Y)))


# *Using the median as general prediction:*

# In[ ]:


median = Y.median()
print('Median:', median)
print('MAE Error (in USD):', median_absolute_error(Y, median * np.ones_like(Y)))


# *Using the geometric mean as general prediction:*

# In[ ]:


geom_mean = np.expm1(np.log1p(Y).mean())
print('Geometric mean:', geom_mean)
print('MAE Error (in USD):', median_absolute_error(Y, geom_mean * np.ones_like(Y)))


# # C. Feature Analysis

# ## C1. Pick-up and drop-off locations

# *Mean fare at each rounded pickup long-lat pair:*

# In[ ]:


pickup_map = train.groupby(['pickup_longitude', 'pickup_latitude'])['fare_amount'].mean().reset_index()
pickup_map = pickup_map.pivot('pickup_longitude', 'pickup_latitude', 'fare_amount').fillna(0)
pickup_map = pickup_map.loc[pickup_map.index[~np.isinf(pickup_map.index)], 
               pickup_map.columns[~np.isinf(pickup_map.columns)]]
gc.collect()
_ = plt.contour(
    pickup_map.columns, 
    pickup_map.index, 
    np.log1p(pickup_map.values), 
    cmap='viridis')
_ = plt.colorbar()


# *[Zoom-in at central] Mean fare at each rounded pickup long-lat pair:*

# In[ ]:


pickup_map = train.groupby(['pickup_longitude', 'pickup_latitude'])['fare_amount'].mean().reset_index()
pickup_map = pickup_map.pivot('pickup_longitude', 'pickup_latitude', 'fare_amount').fillna(0)
pickup_map = pickup_map.loc[pickup_map.index[(~np.isinf(pickup_map.index)) & (pickup_map.index>-76) & (pickup_map.index<-72)], 
               pickup_map.columns[~np.isinf(pickup_map.columns) & (pickup_map.columns>39) & (pickup_map.columns<42.25)]]
gc.collect()
_ = plt.contour(
    pickup_map.columns, 
    pickup_map.index, 
    np.log1p(pickup_map.values), 
    cmap='viridis')
_ = plt.colorbar()


# *98% of pickups are in the central:*

# In[ ]:


train['pickup_central'] = ((train.pickup_longitude>-76) & (train.pickup_longitude<-72) & (train.pickup_latitude>39) & (train.pickup_latitude<42.25)).astype(int)
train.pickup_central.value_counts(True)


# *For the test set, all pickups are in the central:*

# In[ ]:


test['pickup_central'] = ((test.pickup_longitude>-76) & (test.pickup_longitude<-72) & (test.pickup_latitude>39) & (test.pickup_latitude<42.25)).astype(int)
test.pickup_central.value_counts(True)


# *Mean fare at each rounded pickup long-lat pair:*

# In[ ]:


dropoff_map = train.groupby(['dropoff_longitude', 'dropoff_latitude'])['fare_amount'].mean().reset_index()
dropoff_map = dropoff_map.pivot('dropoff_longitude', 'dropoff_latitude', 'fare_amount').fillna(0)
dropoff_map = dropoff_map.loc[dropoff_map.index[~np.isinf(dropoff_map.index)], 
               dropoff_map.columns[~np.isinf(dropoff_map.columns)]]
gc.collect()
_ = plt.contour(
    dropoff_map.columns, 
    dropoff_map.index, 
    np.log1p(dropoff_map.values), 
    cmap='viridis')
_ = plt.colorbar()


# *[Zoom-in at central] Mean fare at each rounded pickup long-lat pair:*

# In[ ]:


dropoff_map = train.groupby(['dropoff_longitude', 'dropoff_latitude'])['fare_amount'].mean().reset_index()
dropoff_map = dropoff_map.pivot('dropoff_longitude', 'dropoff_latitude', 'fare_amount').fillna(0)
dropoff_map = dropoff_map.loc[dropoff_map.index[(~np.isinf(dropoff_map.index)) & (dropoff_map.index>-76) & (dropoff_map.index<-72)], 
               dropoff_map.columns[~np.isinf(dropoff_map.columns) & (dropoff_map.columns>39) & (dropoff_map.columns<42.25)]]
gc.collect()
_ = plt.contour(
    dropoff_map.columns, 
    dropoff_map.index, 
    np.log1p(dropoff_map.values), 
    cmap='viridis')
_ = plt.colorbar()


# *98% of drop-offs are in the central:*

# In[ ]:


train['dropoff_central'] = ((train.dropoff_longitude>-76) & (train.dropoff_longitude<-72) & (train.dropoff_latitude>39) & (train.dropoff_latitude<42.25)).astype(int)
train.dropoff_central.value_counts(True)


# *For the test set, all drop-offs are in the central:*

# In[ ]:


test['dropoff_central'] = ((test.dropoff_longitude>-76) & (test.dropoff_longitude<-72) & (test.dropoff_latitude>39) & (test.dropoff_latitude<42.25)).astype(int)
test.dropoff_central.value_counts(True)


# ### [Key Insight] Focus on central location; disregard location outliers
# *Since both pick-ups and drop-offs are concentrated in the central for both train and test sets, we can focus only on the central for the rest of the analysis*

# In[ ]:


n_train = len(train)
train = train[(train['pickup_central']==1)].reset_index(drop=True)
train = train[(train['dropoff_central']==1)].reset_index(drop=True)
train = train.drop(['pickup_central','dropoff_central'],axis=1)
gc.collect()
print('Remaining:', len(train)/n_train)


# ### [Key Insight] Some sections of pick-up/drop-off locations have higher fare rates
# *A decision tree can be used to determine these sections:*

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
X = train[['pickup_longitude', 'pickup_latitude']]
Y = train.fare_amount.values
idx = np.isfinite(X.pickup_longitude) & np.isfinite(X.pickup_latitude) & np.isfinite(Y)
X = X[idx]
Y = Y[idx]
print(cross_val_score(DecisionTreeRegressor(min_samples_split=30000), X, Y))
m = DecisionTreeRegressor(min_samples_split=30000).fit(X, Y)
X['prediction'] = np.log1p(m.predict(X))
_ = X.groupby(['pickup_longitude', 'pickup_latitude']).prediction.mean().reset_index().plot.scatter('pickup_longitude', 'pickup_latitude', c='prediction', s=1, cmap='viridis')
del X, Y, m, idx
gc.collect()


# In[ ]:


X = train[['dropoff_longitude', 'dropoff_latitude']]
Y = train.fare_amount.values
idx = np.isfinite(X.dropoff_longitude) & np.isfinite(X.dropoff_latitude) & np.isfinite(Y)
X = X[idx]
Y = Y[idx]
print(cross_val_score(DecisionTreeRegressor(min_samples_split=30000), X, Y))
m = DecisionTreeRegressor(min_samples_split=30000).fit(X, Y)
X['prediction'] = np.log1p(m.predict(X))
_ = X.groupby(['dropoff_longitude', 'dropoff_latitude']).prediction.mean().reset_index().plot.scatter('dropoff_longitude', 'dropoff_latitude', c='prediction', s=1, cmap='viridis')
del X, Y, m, idx
gc.collect()


# ## C2. Distances between pickup and dropoff

# *Euclidean distance exhibits moderate heteroskedasticity (bad):*

# In[ ]:


pickups = train[['pickup_longitude', 'pickup_latitude']].values
dropoffs = train[['dropoff_longitude', 'dropoff_latitude']].values
train['distance'] = np.sqrt(np.square(pickups - dropoffs).sum(axis=1)).round(5)
del pickups, dropoffs
gc.collect()
_ = train.groupby('distance').fare_amount.median().reset_index().plot.scatter('distance', 'fare_amount', s=5)


# *Manhattan distance exhibits very high homoskedasticity (very good):*

# In[ ]:


pickups = train[['pickup_longitude', 'pickup_latitude']].values
dropoffs = train[['dropoff_longitude', 'dropoff_latitude']].values
train['distance'] = np.abs(pickups - dropoffs).sum(axis=1).round(5)
del pickups, dropoffs
gc.collect()
_ = train.groupby('distance').fare_amount.median().reset_index().plot.scatter('distance', 'fare_amount', s=5)


# ### [Key Insight] An almost-linear relationship exists between Manhattan distance and fare amount

# *Ridge regression yields a CV MAE score of 2.64:*

# In[ ]:


from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
X, Y = train.distance.values, train.fare_amount.values
idx = np.isfinite(X) & np.isfinite(Y)
X = X[idx].reshape(-1, 1)
Y = Y[idx]
gc.collect()
cross_val_score(Ridge(), X, Y, scoring=make_scorer(median_absolute_error))


# *An isotonic regression yields a CV MAE score of 2.59:*

# In[ ]:


from sklearn.isotonic import IsotonicRegression
cross_val_score(IsotonicRegression(3, 140), X.flatten(), Y, scoring=make_scorer(median_absolute_error))


# ## C3. Number of passengers

# ### [Key Insight] 98% of rides have one to six passengers

# In[ ]:


print('[Train] Percentage of rides with 1-6 passengers:', (train.passenger_count.isin(range(1,6))).sum()/len(train))
print('[Test] Percentage of rides with 1-6 passengers:', (test.passenger_count.isin(range(1,6))).sum()/len(test))


# In[ ]:


train.passenger_count = train.passenger_count.clip(1, 6)
test.passenger_count = test.passenger_count.clip(1, 6)


# *Passenger count vs. mean fare amount:*

# In[ ]:


_ = train.groupby('passenger_count').fare_amount.mean().reset_index().plot.scatter('passenger_count', 'fare_amount')


# ### [Key Insight] A monotonic piecewise relationship exists between passenger and fare amount

# *A linear regression yields a CV MAE score of 4.56:*

# In[ ]:


X, Y = train[['passenger_count']].values, train.fare_amount.values
gc.collect()
cross_val_score(Ridge(), X, Y, scoring=make_scorer(median_absolute_error))


# *An isotonic regression yields a CV MAE score of 3.7:*

# In[ ]:


cross_val_score(IsotonicRegression(3, 140), X.flatten(), Y.flatten(), scoring=make_scorer(median_absolute_error))


# ## C4. Pickup Date & Time

# ### [Key Insight] Taxi fare hike on September 2012 (<a href="https://www.nytimes.com/2012/09/04/nyregion/new-york-taxis-to-start-charging-increased-rates.html">click here to go to related news article</a>):

# *Monthly mean fare amount (skyblue lines = start of year; pink highlight = fare hike):*

# In[ ]:


monthly_fare = train.groupby(['year', 'month']).fare_amount.mean().reset_index()
monthly_fare.index = pd.to_datetime(monthly_fare.year.astype(str) + '-' + monthly_fare.month.astype(str).str.zfill(2) + '-01')
gc.collect()
_ = monthly_fare.fare_amount.plot()
_ = [plt.axvline(pd.to_datetime(m), linestyle='dashed', color='skyblue') for m in monthly_fare.index if m.month==1]
_ = plt.axvspan(pd.to_datetime('2012-08-01'), pd.to_datetime('2012-09-01'), color='pink')


# In[ ]:


train['hike_status'] = 'pre-hike'
train.loc[(train.year >= 2012) & (train.month >= 9), 'hike_status'] = 'post-hike'

test['hike_status'] = 'pre-hike'
test.loc[(test.year >= 2012) & (test.month >= 9), 'hike_status'] = 'post-hike'


# *Median USD 1.40 dollar increase post-hike:*

# In[ ]:


train.groupby('hike_status').fare_amount.median()


# ### [Key Insight] Low-fare season: rides on January, February and March have lower fares than average:

# *Monthly average fare amount - compared yearly (pink highlight has lower fare amounts):*

# In[ ]:


years = [2009, 2010, 2011, 2012, 2013, 2014, 2015]
for y in years:
    year_data = train[train.year==y].reset_index(drop=True).groupby('month').fare_amount.mean().reset_index()
    year_data.fare_amount.plot.line(marker='.')
    gc.collect()
_ = plt.xticks(range(12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
_ = plt.axvspan(-0.5, 2.5, color='pink')
_ = plt.legend(years)


# In[ ]:


train['fare_season'] = 'high fare season'
train.loc[train.month <= 3, 'fare_season'] = 'low fare season'

test['fare_season'] = 'high fare season'
test.loc[test.month <= 3, 'fare_season'] = 'low fare season'


# *Median 40 cent decrease in low-fare season:*

# In[ ]:


train.groupby('fare_season').fare_amount.median()


# ### [Key Insight] Peak and off-peak hours
# > **Higher fares**<br>
# > - 12mn - 2am (from bars)<br>
# > - 10am - 1pm (commute to work)<br>
# > - 6pm (commute back home)
# 
# > **Lower fares:**<br>
# > - 3am - 6am (unholy hours)
# > - 5pm (just finishing work)
# > - 7pm - 8pm (everyone having dinner)

# In[ ]:


hourly_fare = train.groupby('hour').fare_amount.mean().reset_index()
# UTC time adjustment to NYC (GMT-4)
hourly_fare.hour -= 4
hourly_fare.hour %= 24
hourly_fare = hourly_fare.sort_values('hour').reset_index(drop=True)
gc.collect()
_ = hourly_fare.fare_amount.plot.line(marker='.')
hours = ['12mn'] + [str(i) + 'am' for i in range(1,12)] + ['12nn'] + [str(i) + 'pm' for i in range(1,12)]
_ = plt.xticks(range(24), hours)
_ = plt.axhline(11.5, color='pink', linestyle='dashed')
_ = plt.axhline(11, color='pink', linestyle='dashed')


# In[ ]:


hours_high = set([0, 1, 2, 10, 11, 12, 13, 18])
hours_low = set([3, 4, 5, 6, 17, 19, 20])

train['hourly_seasonality'] = 'normal'
train.loc[(train.hour-4).isin(hours_high), 'hourly_seasonality'] = 'peak'
train.loc[(train.hour-4).isin(hours_low), 'hourly_seasonality'] = 'off-peak'
gc.collect()

test['hourly_seasonality'] = 'normal'
test.loc[(test.hour-4).isin(hours_high), 'hourly_seasonality'] = 'peak'
test.loc[(test.hour-4).isin(hours_low), 'hourly_seasonality'] = 'off-peak'
gc.collect()


# *85 cents increase in mean fare from off-peak to peak hours:*

# In[ ]:


train.groupby('hourly_seasonality').fare_amount.mean().sort_values(ascending=False)


# # D. Predictive Modelling

# ## D1. Isotonic Regression per segment (TODO)
# *Isotonic regression will most likely fare well because of the almost-linear relationship between Manhattan distance and taxi fare.*<br>
# *Also, the isotonic regression is done in segments because the isotonic relationships are heterogeneous for each segment.*
# 
# Segments are based on the combination of the following factors:
# - passenger count
# - fare season
# - hike status
# - hourly seasonality

# In[ ]:


# factors = [
#     'passenger_count',
#     'fare_season',
#     'hike_status',
#     'hourly_seasonality',
# ]
# train['segment'] = train[factors].astype(str).T.apply(lambda x: ':'.join(x))
# gc.collect()


# In[ ]:


# train.segment.value_counts()


# In[ ]:


# segments = sorted(train.segment.unique())
# scores = {}
# for segment in segments:
#     X = train[train.segment == segment]['distance'].values.flatten()
#     Y = train[train.segment == segment]['fare_amount'].values.flatten()
#     idx = np.isfinite(X) & np.isfinite(Y)
#     X = X[idx]
#     Y = Y[idx]
#     cv_score = cross_val_score(IsotonicRegression(out_of_bounds='clip'), X, Y, scoring=make_scorer(median_absolute_error))
#     scores[segment] = cv_score.mean()
#     print(segment, cv_score)
#     gc.collect()

