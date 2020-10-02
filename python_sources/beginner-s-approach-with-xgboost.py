#!/usr/bin/env python
# coding: utf-8

# ### Importing Libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geojson
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

from mpl_toolkits.basemap import Basemap
from matplotlib import cm
from descartes import PolygonPatch
from haversine import haversine

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import MiniBatchKMeans

get_ipython().run_line_magic('matplotlib', 'inline')


# ### Loading Data Sets

# In[ ]:


from subprocess import check_output
print(check_output(["ls", "../input/"]).decode("utf8"))


# #### Kaggle Data
# 
# ---

# In[ ]:


df = pd.read_csv('../input/nyc-taxi-trip-duration/train.csv')
df['pickup_datetime']  = pd.to_datetime(df['pickup_datetime'])
df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'])

df.head()


# #### 2016 NYC Holidays
# 
# ---

# In[ ]:


hf = pd.read_csv('../input/nyc2016holidays/NYC_2016Holidays.csv', sep=';')

hf['Date'] = hf['Date'].apply(lambda x: x + ' 2016')
hf['Date']  = pd.to_datetime(hf['Date'])
hf


# #### Weather

# In[ ]:


wf = pd.read_csv('../input/weather-data-in-new-york-city-2016/weather_data_nyc_centralpark_2016.csv')
wf['date']  = pd.to_datetime(wf['date'])

wf.head()


# ### Map Based Visualizations

# In[ ]:


min_lat = df['pickup_latitude'].min()
max_lat = df['pickup_latitude'].max()
min_lon = df['pickup_longitude'].min()
max_lon = df['pickup_longitude'].max()

print(min_lat, max_lat)
print(min_lon, max_lon)


# Some points are far away from the boundaries of NYC. Let's limit the scope for visualization purposes, at least.

# In[ ]:


west, south, east, north = -74.26, 40.50, -73.70, 40.92
df = df[(df['pickup_longitude'] > west) & (df['pickup_longitude'] < east)]
df = df[(df['pickup_latitude'] > south) & (df['pickup_latitude'] < north)]

fig = plt.figure(figsize=(14,8))

m = Basemap(projection='merc', llcrnrlat=south, urcrnrlat=north,
            llcrnrlon=west, urcrnrlon=east, lat_ts=south, resolution='c')
x, y = m(df['pickup_longitude'].values, df['pickup_latitude'].values)
m.hexbin(x, y, gridsize=300, bins='log', cmap=cm.YlOrRd_r, lw=0.4)

plt.title('Pickup Locations')
plt.show()


# In[ ]:


west, south, east, north = -74.26, 40.50, -73.70, 40.92
df = df[(df['dropoff_longitude'] > west) & (df['dropoff_longitude'] < east)]
df = df[(df['dropoff_latitude'] > south) & (df['dropoff_latitude'] < north)]

fig = plt.figure(figsize=(14,8))

m = Basemap(projection='merc', llcrnrlat=south, urcrnrlat=north,
            llcrnrlon=west, urcrnrlon=east, lat_ts=south, resolution='c')
x, y = m(df['dropoff_longitude'].values, df['dropoff_latitude'].values)
m.hexbin(x, y, gridsize=300, bins='log', cmap=cm.YlOrRd_r, lw=0.4)

plt.title('Dropoff Locations')
plt.show()


# As can be seen, most of the tax trips pickups occurs in Manhattan and at JFK. Dropoffs, on the other hand, are better distributed in NYC. We can also note that some pickups/drop off points come from outside NYC while other seem to be locate overseas.  

# ### Feature Engineering
# 
# For feature engineering, I will create new date based attributes, as well as two new fields representing the distance covered and the average speed.

# In[ ]:


coords = np.vstack((df[['pickup_latitude',  'pickup_longitude']].values,
                    df[['dropoff_latitude', 'dropoff_longitude']].values,
                    df[['pickup_latitude',  'pickup_longitude']].values,
                    df[['dropoff_latitude', 'dropoff_longitude']].values))
kmeans = MiniBatchKMeans(n_clusters=100, batch_size=10000).fit(coords)


# In[ ]:


def derive_features(df):
    df = df.copy()
    # Pickup-based features
    df['pickup_date'] = df['pickup_datetime'].dt.date
    df['pickup_weekday'] = df['pickup_datetime'].dt.weekday
    df['pickup_hour'] = df['pickup_datetime'].dt.hour
    df['pickup_month'] = df['pickup_datetime'].dt.month
    df['pickup_day'] = df['pickup_datetime'].dt.day
    # Brand new features
    df['haversine_distance'] = df.apply(lambda x: haversine((x['pickup_latitude'],  x['pickup_longitude']), 
                                                            (x['dropoff_latitude'], x['dropoff_longitude'])), axis=1)
    df['pickup_cluster']  = kmeans.predict(df[['pickup_latitude', 'pickup_longitude']])
    df['dropoff_cluster'] = kmeans.predict(df[['dropoff_latitude', 'dropoff_longitude']])
    return df


# In[ ]:


df = derive_features(df)


# In[ ]:


df.head()


# ### Feature Visualizations

# In[ ]:


json_data = geojson.load(open('../input/pediacitiesnycneighborhoods/community-districts-polygon.geojson'))
json_data.keys()
polygons = json_data['features']
FILL = '#6699cc'
CONT = '#1a169e'


# In[ ]:


fig = plt.figure(figsize=(14,8))
ax = fig.gca() 

for i in range(len(polygons)):
    coordlist = polygons[i]['geometry']['coordinates']
    poly = {'type':'Polygon', 'coordinates':coordlist}
    ax.add_patch(PolygonPatch(poly, fc=FILL, ec=CONT, alpha=0.5, zorder=1))
    ax.axis('scaled')

ax.scatter(df['pickup_longitude'].values, df['pickup_latitude'], s=10, lw=0,
           c=df['pickup_cluster'].values, cmap='tab20', zorder=2)
plt.title('Pickup Clusters')
plt.grid(False)


# In[ ]:


fig = plt.figure(figsize=(14,8))
ax = fig.gca() 

for i in range(len(polygons)):
    coordlist = polygons[i]['geometry']['coordinates']
    poly = {'type':'Polygon', 'coordinates':coordlist}
    ax.add_patch(PolygonPatch(poly, fc=FILL, ec=CONT, alpha=0.5, zorder=1))
    ax.axis('scaled')

ax.scatter(df['dropoff_longitude'].values, df['dropoff_latitude'], s=10, lw=0,
           c=df['dropoff_cluster'].values, cmap='tab20', zorder=2)
plt.title('Dropoff Clusters')
plt.grid(False)


# In[ ]:


plt.figure(figsize=(14, 5))
plt.title('Vendor ID')
sns.countplot(x='vendor_id', data=df)


# As can be seen, `vendor_id=2` performed more trips than `vendor_id=1`.

# In[ ]:


plt.figure(figsize=(14, 5))
plt.title('Trip Duration in Seconds')
sns.distplot(df['trip_duration'], hist=False)


# Lot's of long trips, which will be discarded.

# In[ ]:


print('Mean trip duration (min): {0:.2f}'.format(df['trip_duration'].mean()/60))
print('Max trip duration (min): {0:.2f}'.format(df['trip_duration'].max()/60))
print('Median trip duration (min): {0:.2f}'.format(df['trip_duration'].median()/60))


# In[ ]:


plt.figure(figsize=(14, 5))
plt.title('Distance Traveled in km')
sns.distplot(df['haversine_distance'], hist=False)


# In[ ]:


print('Mean trip distance (km): {0:.2f}'.format(df['haversine_distance'].mean()))
print('Max trip distance (km) {0:.2f}'.format(df['haversine_distance'].max()))
print('Median trip distance (km): {0:.2f}'.format(df['haversine_distance'].median()))


# In[ ]:


plt.figure(figsize=(14, 5))
plt.title('Pickup per Weekday')
sns.countplot(x='pickup_weekday', data=df)


# In[ ]:


plt.figure(figsize=(14, 5))
plt.title('Pickup per Hour')
sns.countplot(x='pickup_hour', data=df)


# More pickups during rush hours, decreasing after midnight...

# In[ ]:


plt.figure(figsize=(14, 5))
plt.title('Passenger')
sns.countplot(x='passenger_count', data=df)


# Most passengers travel alone.

# In[ ]:


plt.figure(figsize=(14, 5))
plt.title('Pickups per Month')
sns.countplot(x='pickup_month', data=df)


# ### Trip Duration Relations

# In[ ]:


plt.figure(figsize=(14, 5))
plt.title('Duration per Weekday')
sns.pointplot(x='pickup_weekday', y='trip_duration', hue='vendor_id', data=df)


# In[ ]:


plt.figure(figsize=(14, 5))
plt.title('Duration per Hour')
sns.pointplot(x='pickup_hour', y='trip_duration', hue='vendor_id', data=df)


# In[ ]:


plt.figure(figsize=(14, 5))
plt.title('Duration per Month')
sns.pointplot(x='pickup_month', y='trip_duration', hue='vendor_id', data=df)


# In[ ]:


plt.figure(figsize=(14, 5))
plt.title('Duration per Passenger Count')
sns.pointplot(x='passenger_count', y='trip_duration', hue='vendor_id', data=df)


# `vendor_id=2`, besides of taking more rides, also presents longer average trip duration in any weekdays. We can see that both vendors present a similar trip duration during 6am. However, in this case, `vendor_id=1` shows higher variability in the average trip duration. 

# ### Data Cleaning

# Before modeling some outliers should be removed, as extreme long trips and records with null or extreme speeds.
# I'll remove trips with the following features:
# 
# - Trip duration near 0s
# - Trip duration longer than a day
# - Near zero traveled distance

# In[ ]:


def clean_data(df):
    df = df.copy()
    df = df[(df['trip_duration'] > 60) & (df['trip_duration'] < 3600 * 24)] # Trip duration filtering
    df = df[(df['haversine_distance'] > 0.01)] # Distance filtering
    return df


# In[ ]:


df = clean_data(df)


# #### Sanity Checking
# 
# ---

# In[ ]:


plt.figure(figsize=(14, 5))
plt.title('Distance Traveled in km')
sns.distplot(df['haversine_distance'], hist=False)


# In[ ]:


plt.figure(figsize=(14, 5))
plt.title('Trip Duration in Seconds')
sns.distplot(df['trip_duration'], hist=False)


# In[ ]:


print('Min trip distance (km): {0:.2f}'.format(df['haversine_distance'].min()))
print('Min trip duration (s): {0:.2f}'.format(df['trip_duration'].min()))


# Okay, everything seems fine.

# ### External Data Sets

# First, let's merge all data set with the weather data set. 

# It's necessary to convert the traces (`T`) values into small numeric amounts and to convert columns that accept this value into numeric ones.

# In[ ]:


wf = wf.replace(to_replace='T', value=0.01)

wf['precipitation'] = pd.to_numeric(wf['precipitation'])
wf['snow fall'] = pd.to_numeric(wf['snow fall'])
wf['snow depth'] = pd.to_numeric(wf['snow depth'])

wf.tail()


# In[ ]:


df['pickup_date']   = pd.to_datetime(df['pickup_date'])
mf = df.merge(wf, left_on='pickup_date', right_on='date', how='inner')
mf.tail()


# In[ ]:


mf.dtypes


# All done.
# 
# Let's check how these variables correlates to each other.
# 
# ### Correlation Between Features

# In[ ]:


corr = mf.select_dtypes(include=[np.number]).corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True

plt.figure(figsize=(10, 10))
with sns.axes_style('white'):
    sns.heatmap(corr, mask=mask, linewidths=0.01, square=True, linecolor='white')
plt.xticks(rotation=90)
plt.title('Correlation between features')


# We can note a positive and strong correlation between (excluding correlation between features that present some sort of dependency on each other):
# 
# - (haversine_distance, pickup_longitude)
# - (trip_duration, pickup_longitude)
# - (_temperature, pickup_month)
# 
# 
# Curiously there is a negative correlation between  (haversine_distance, pickup_latitude)
# 
# *Note that it doesn't make sense to analyse the correlation between the pickup and dropoff cluster variables with any other one. The same applies to vendor_id.* 

# Just for fun, I'll also combine this data set with holidays flags. I believe that distance traveled and some other features may be dependant on holidays. However, as there are so few holidays in the year, I don't believe this information will be significant. Anyways, let's take a look 

# In[ ]:


mf = mf.merge(hf, left_on='pickup_date', right_on='Date', how='left')
mf.head()


# In[ ]:


mf.loc[~mf['Holiday'].isnull(), 'Holiday'] = 1 
mf.loc[mf['Holiday'].isnull(), 'Holiday'] = 0

mf['Holiday'] = pd.to_numeric(mf['Holiday'])

mf['Holiday'].value_counts()


# I'll drop any `datetime64` columns.

# In[ ]:


mf = mf.drop(['Day', 'Date', 'date', 'pickup_date'], axis=1)


# ### Feature Importances
# 
# Time for some modeling! I'll make use of sklearn in a first step to visualize feature importances.

# In[ ]:


mf.dtypes


# In[ ]:


le = LabelEncoder()

for cat in ['store_and_fwd_flag', 'vendor_id']:
    mf[cat] = le.fit_transform(mf[cat])


# In[ ]:


X = mf.select_dtypes(include=[np.number])
X = X.drop(['trip_duration'], axis=1)
y = np.log1p(mf['trip_duration'])


# Here I'm going to define a custom metric, used in this competition, to measure the models' performance.

# In[ ]:


def rmsle_eval(y, y0):
    y0 = y0.get_label()    
    assert len(y) == len(y0)
    return 'error', np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y0), 2)))


# In[ ]:


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)
dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_valid, label=y_valid)
watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

params = {}
params["objective"] = "reg:linear"
params["eta"] = 0.1
params["min_child_weight"] = 30
params["subsample"] = 0.8
params["colsample_bytree"] = 0.3
params["scale_pos_weight"] = 1.0
params["silent"] = 1
params["max_depth"] = 10
params["nthread"] = -1

plst = list(params.items())

model = xgb.train(plst, dtrain, 200, watchlist, early_stopping_rounds=50, maximize=False, 
                  verbose_eval=20, feval=rmsle_eval)


# In[ ]:


def plot_importances(clf):
    importances = clf.get_fscore()
    importances = sorted(importances.items(), key=lambda x: x[1])
    x = list(zip(*importances))[0]
    y = list(zip(*importances))[1]
    x_pos = np.arange(len(x)) 
    plt.figure(figsize=(10,5))
    plt.title('Feature importances')
    plt.barh(range(len(y)), y, align='center')
    plt.yticks(range(len(y)), x)
    plt.ylim([-1, len(importances)])
    plt.xlabel('F score')
    plt.show()


# In[ ]:


plot_importances(model)


# ### Submission
# 
# It will be necessary apply all the transformations and feature engineering to the test data set.

# In[ ]:


tf = pd.read_csv('../input/nyc-taxi-trip-duration/test.csv')
ids = tf['id']
tf['pickup_datetime']  = pd.to_datetime(tf['pickup_datetime'])


# In[ ]:


tf = derive_features(tf)


# In[ ]:


tf['pickup_date']   = pd.to_datetime(tf['pickup_date'])
tf = tf.merge(wf, left_on='pickup_date', right_on='date', how='inner')


# In[ ]:


tf = tf.merge(hf, left_on='pickup_date', right_on='Date', how='left')
tf.loc[~tf['Holiday'].isnull(), 'Holiday'] = 1 
tf.loc[tf['Holiday'].isnull(), 'Holiday'] = 0

tf['Holiday'] = pd.to_numeric(tf['Holiday'])

tf['Holiday'].value_counts()


# In[ ]:


le = LabelEncoder()

for cat in ['store_and_fwd_flag', 'vendor_id']:
    tf[cat] = le.fit_transform(tf[cat])


# In[ ]:


tf = tf.drop(['Day', 'Date', 'date', 'pickup_date'], axis=1)


# In[ ]:


tf = tf.select_dtypes(include=[np.number])


# In[ ]:


dtest = xgb.DMatrix(tf)
ytest = model.predict(dtest)


# In[ ]:


submission = pd.DataFrame({'id': ids, 'trip_duration': np.expm1(ytest)})
submission.to_csv('nyc-output.csv', index=False)

