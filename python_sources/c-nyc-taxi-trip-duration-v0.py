#!/usr/bin/env python
# coding: utf-8

# Checking files:

# In[ ]:


import numpy  as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Summoning some libs:

# In[ ]:


import datetime
import warnings
import gc

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn           as sns
from pandas.plotting import scatter_matrix

from sklearn                 import metrics
from sklearn.model_selection import train_test_split
from sklearn.impute          import SimpleImputer
from sklearn.compose         import ColumnTransformer
from sklearn.preprocessing   import OrdinalEncoder,                                      OneHotEncoder
from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import StandardScaler

from sklearn.cluster      import KMeans, DBSCAN
from sklearn.linear_model import LinearRegression
from sklearn.tree         import DecisionTreeRegressor

from sklearn.metrics import mean_absolute_error,                             mean_squared_log_error

from IPython.display import display,                             FileLink

#

warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = [13, 7]
np.random.seed(1642)


# Defining some functions:

# In[ ]:


# constants and functions

def var_cleaner(s):
    """
    ('var1, var2, ..., varN') -> None
    """
    trash = list()
    miss  = list()
    for v in s.replace(' ', '').split(','):
        if v in globals():
            del globals()[v]
            trash.append(v)
        else:
            miss.append(v)
    print('- DELETED:     {}'.format( ', '.join(trash) ))
    print('- NOT DEFINED: {}'.format( ', '.join(miss) ))

from math import sin, cos, sqrt, atan2, radians
def lat_lon_converter(lat1, lon1, lat2, lon2, unit):
    """
    ref: https://stackoverflow.com/questions/19412462/getting-distance-between-two-points-based-on-latitude-longitude
    """
    try:
        R = 6373.0
        dlon = radians(lon2) - radians(lon1)
        dlat = radians(lat2) - radians(lat1)
        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        distance = R * c

        if unit == 'm':
            return distance * 10e3
        elif unit == 'km':
            return distance
    except ValueError:
        return np.nan

    
def dbscan_predict(model, X):
    """
    ref: https://stackoverflow.com/questions/27822752/scikit-learn-predicting-new-points-with-dbscan
    """
    nr_samples = X.shape[0]

    y_new = np.ones(shape=nr_samples, dtype=int) * -1

    for i in range(nr_samples):
        diff = model.components_ - X[i, :]   # NumPy broadcasting
        dist = np.linalg.norm(diff, axis=1)  # Euclidean distance
        shortest_dist_idx = np.argmin(dist)

        if dist[shortest_dist_idx] < model.eps:
            y_new[i] = model.labels_[model.core_sample_indices_[shortest_dist_idx]]

    return y_new


import scipy as sp
def dbscan_predict2(dbscan_model, X_new, metric=sp.spatial.distance.euclidean):
    """
    ref: https://stackoverflow.com/questions/27822752/scikit-learn-predicting-new-points-with-dbscan
    """
    # Result is noise by default
    y_new = np.ones(shape=len(X_new), dtype=int)*-1 

    # Iterate all input samples for a label
    for j, x_new in enumerate(X_new):
        # Find a core sample closer than EPS
        for i, x_core in enumerate(dbscan_model.components_): 
            if metric(x_new, x_core) < dbscan_model.eps:
                # Assign label of x_core to x_new
                y_new[j] = dbscan_model.labels_[dbscan_model.core_sample_indices_[i]]
                break

    return y_new


# # Exploring

# In[ ]:


df_train = pd.read_csv('/kaggle/input/nyc-taxi-trip-duration/train.csv')
df_test  = pd.read_csv('/kaggle/input/nyc-taxi-trip-duration/test.csv')

print('train: ', df_train.shape)
print('test:  ', df_test.shape)

display( df_train.head() )
display( df_test.head() )


# In[ ]:


_TARGET      = 'trip_duration'
_NON_FEATURE = set(df_train.columns) - set(df_test.columns)
_FEATURES    = set(df_train.columns).intersection(set(df_test.columns)) - set(['id'])
display(_FEATURES)


# In[ ]:


train = df_train[_FEATURES]


# ## Overview

# In[ ]:


train.info()


# In[ ]:


train.describe().apply( lambda s: s.apply( lambda x: format(x, '.3f') ) )


# In[ ]:


# scatter_matrix(train[train.dtypes[train.dtypes != object].index].sample(frac=0.005), diagonal='kde'); it doesnt bring any insight
train[train.dtypes[train.dtypes != object].index].hist(bins=75, grid='off');


# ## Dealing with outliers

# In[ ]:


_FILTER_OBJECT = train.dtypes[train.dtypes != object].index


# In[ ]:


for _col in _FILTER_OBJECT:
    #train['{}_quantile'.format(_col)] = pd.qcut(train[_col], 10, labels=False, duplicates='drop')
    _std  = train[_col].std()
    _mean = train[_col].mean()
    train['{}_outlier'.format(_col)] = train[_col].apply( lambda x: True if ( abs(x) > abs(_mean + 1.5*_std) ) else False)


# In[ ]:


# train[train['trip_duration_outlier'] == False].hist(column = 'trip_duration', 
#                                                     by     = 'passenger_count',
#                                                     bins   = 50, grid='off', alpha = 0.5)
# plt.title('Trip Duration Distributions per Passenger Count')
# plt.legend();


# <s>Binning trip duration:</s>

# In[ ]:


# _TRIP_BINS = [0.0, 300.0, 600.0, np.inf]
# train['trip_duration_cat'] = pd.cut( train['trip_duration'],
#                                      bins   = _TRIP_BINS,
#                                      labels = [i for i in range(len(_TRIP_BINS)-1)])
# train['trip_duration_cat'].value_counts(normalize=True).sort_index()


# ## Geographical data

# In[ ]:


_FILTER = (train['dropoff_latitude_outlier'] == False) & (train['dropoff_longitude_outlier'] == False)
# display( train['dropoff_longitude_outlier'].value_counts() )
# display( train['dropoff_latitude_outlier'].value_counts() )
# display( (_FILTER).value_counts() )


# In[ ]:


_p = train[_FILTER].plot(kind='scatter', x='pickup_longitude', y='pickup_latitude', alpha=0.25, color='b', label='pickup')
train[_FILTER].plot(kind='scatter', x='dropoff_longitude', y='dropoff_latitude', alpha=0.25, color='r', label='dropoff', ax=_p)
plt.legend();


# - - -
# 
# # Model

# In[ ]:


var_cleaner('df_train,  df_test, _TARGET, _NON_FEATURE, _FEATURES, train')

#

df_train = pd.read_csv('/kaggle/input/nyc-taxi-trip-duration/train.csv')
df_test  = pd.read_csv('/kaggle/input/nyc-taxi-trip-duration/test.csv')

print('train: ', df_train.shape)
print('test:  ', df_test.shape)

_TARGET      = 'trip_duration'
_NON_FEATURE = set(df_train.columns) - set(df_test.columns)
_FEATURES    = set(df_train.columns).intersection(set(df_test.columns)) - set(['id', 'vendor_id'])
train = df_train[_FEATURES]
test  = df_test[_FEATURES]

display(_FEATURES)


# ## Simple Feature Engineering

# ### Clustering

# In[ ]:


_sample = train[_FILTER].sample(3000, random_state=159)[['pickup_longitude', 'pickup_latitude']]


# #### Kmeans

# In[ ]:


kmeans = KMeans(n_clusters=5, random_state=51).fit(_sample)

display(kmeans.cluster_centers_)
_p = _sample.plot(kind='scatter', x='pickup_longitude', y='pickup_latitude', alpha=0.25, color='b', label='pickup')
_p.scatter(x=kmeans.cluster_centers_[:,0],y=kmeans.cluster_centers_[:,1], marker='o', color='g', s=150);


# #### DBSCAN

# In[ ]:


dbscan = DBSCAN(eps=0.01, min_samples=15).fit(_sample)

core_samples_mask = np.zeros_like( dbscan.labels_, dtype=bool )
core_samples_mask[dbscan.core_sample_indices_] = True
labels = dbscan.labels_

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_    = list(labels).count(-1)

print('Estimated number of clusters:     %d'    % n_clusters_)
print('Estimated number of noise points: %d'    % n_noise_)
# print("Homogeneity:                      %0.3f" % metrics.homogeneity_score(labels_true, labels))
# print("Completeness:                     %0.3f" % metrics.completeness_score(labels_true, labels))
# print("V-measure:                        %0.3f" % metrics.v_measure_score(labels_true, labels))
# print("Adjusted Rand Index:              %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
# print("Adjusted Mutual Information:      %0.3f" % metrics.adjusted_mutual_info_score(labels_true, labels, average_method='arithmetic'))
print("Silhouette Coefficient:           %0.3f" % metrics.silhouette_score(_sample[['pickup_longitude', 'pickup_latitude']], labels))

#

unique_labels = set(labels)
colors        = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]


# In[ ]:


for k, col in zip(unique_labels, colors):
    if k == -1:
        col = [0, 0, 0, 1] # Black used for noise.

    class_member_mask = (labels == k)

    xy = _sample[class_member_mask & core_samples_mask].as_matrix()
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=14)

    xy = _sample[class_member_mask & ~core_samples_mask].as_matrix()
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()


# Predicting all observations:

# In[ ]:


train['db_predict']     = dbscan_predict(dbscan, train[['pickup_longitude', 'pickup_latitude']].as_matrix())
train['kmeans_predict'] = kmeans.predict(train[['pickup_longitude', 'pickup_latitude']])


# In[ ]:


display( train['db_predict'].value_counts().sort_index() )
display( train['kmeans_predict'].value_counts().sort_index() )


# In[ ]:


train.plot(kind='scatter', x='pickup_longitude', y='pickup_latitude', c='db_predict', cmap=plt.get_cmap('jet'));
train.plot(kind='scatter', x='pickup_longitude', y='pickup_latitude', c='kmeans_predict', cmap=plt.get_cmap('jet'));


# ### Additional feature combination

# In[ ]:


train['lon_lat_manhattan']    = abs(train['dropoff_longitude']-train['pickup_longitude']) + abs(train['dropoff_latitude']-train['pickup_latitude'])
train['dist_manhattan_meter'] = train.apply( lambda x: lat_lon_converter(x['pickup_latitude'], 
                                                                         x['pickup_longitude'],
                                                                         x['dropoff_latitude'], 
                                                                         x['dropoff_longitude'],
                                                                         'm'), axis=1 )


# In[ ]:


#train['pickup_dt']   = train['pickup_datetime'].apply( lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
#train['dropoff_dt']  = train['dropoff_datetime'].apply( lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
train['pickup_dt']    = pd.to_datetime(train['pickup_datetime'], format='%Y-%m-%d %H:%M:%S', errors='ignore')
# train['dropoff_dt'] = pd.to_datetime(train['dropoff_datetime'], format='%Y-%m-%d %H:%M:%S', errors='ignore')
# train['delta_time'] = (train['dropoff_dt'] - train['pickup_dt']).dt.total_seconds()

# train[['pickup_dt', 'pickup_datetime']]


# In[ ]:


train['pick_minute']     = train['pickup_dt'].dt.minute
train['pick_hour']       = train['pickup_dt'].dt.hour
train['pick_day']        = train['pickup_dt'].dt.day
train['pick_month']      = train['pickup_dt'].dt.month
train['pick_year']       = train['pickup_dt'].dt.year
train['pick_quarter']    = train['pickup_dt'].dt.quarter
train['pick_weekofyear'] = train['pickup_dt'].dt.weekofyear


# In[ ]:


# train['avg_speed']           = train['dist_manhattan_meter'] / train['delta_time']
# train['dist_per_passenger']  = train['dist_manhattan_meter'] / train['passenger_count']
# train['speed_per_passenger'] = train['avg_speed'] / train['passenger_count']


# ## Data Cleaning

# In[ ]:


# _FILTER_NUM = set(train.dtypes[(train.dtypes != np.dtype('object')) & (train.dtypes != np.dtype('<M8[ns]'))].index.to_list())
_FILTER_INT    = set(train.dtypes[(train.dtypes == np.dtype('int64'))].index.to_list())
_FILTER_FLOAT  = set(train.dtypes[(train.dtypes == np.dtype('float64'))].index.to_list())
_FILTER_CAT    = set(train.dtypes[(train.dtypes == np.dtype('object'))].index.to_list())
_FILTER_DT     = set(train.dtypes[(train.dtypes == np.dtype('<M8[ns]'))].index.to_list())

train[ _FILTER_INT.union(_FILTER_FLOAT).union(_FILTER_CAT).union(_FILTER_DT) ].describe().apply( lambda s: s.apply( lambda x: format(x, '.3f') ) )


# In[ ]:


# train['dist_per_passenger']  = train['dist_per_passenger'].replace([np.inf, -np.inf], np.nan)
# train['speed_per_passenger'] = train['speed_per_passenger'].replace([np.inf, -np.inf], np.nan)


# In[ ]:


display(_FILTER_INT)
display(_FILTER_FLOAT)
display(_FILTER_CAT)
display(_FILTER_DT)


# In[ ]:


_FILTER_CAT = _FILTER_CAT - {'pickup_datetime'}


# In[ ]:


# imputer = SimpleImputer(strategy='median')
# imputer.fit(train[(_FILTER_OBJECT)])
# X = imputer.transform(train[_FILTER_OBJECT])
# train_tr = pd.DataFrame(X, columns=train[(_FILTER_OBJECT)].columns)
int_pipeline = Pipeline([ ('imputer', SimpleImputer(strategy="constant", fill_value=-1))])

float_pipeline = Pipeline([ ('imputer', SimpleImputer(strategy="median")),
                          ('std_scaler', StandardScaler())
                        ])


# In[ ]:


full_pipeline = ColumnTransformer([
                 ('int',   int_pipeline,    list(_FILTER_INT)),
                 ('float', float_pipeline,  list(_FILTER_FLOAT)),
                 ('cat',   OneHotEncoder(), list(_FILTER_CAT))
                 ])

d


# ## Train & Test

# In[ ]:


train_y['trip_duration'] = train_y['trip_duration'].mask(train_y['trip_duration'].lt(0), 0)
_TRIP_BINS = [0.0, 300.0, 600.0, np.inf]
train_y['trip_duration_cat'] = pd.cut( train_y['trip_duration'],
                                     bins   = _TRIP_BINS,
                                     labels = [i for i in range(len(_TRIP_BINS)-1)])
train_y['trip_duration_cat'].value_counts(normalize=True).sort_index()


# In[ ]:





# In[ ]:


X_train, X_holdout, y_train, y_holdout = train_test_split(train_prepared, train_y, 
                                          test_size    = 0.2, 
                                          random_state = 13,
                                          stratify     = train_y['trip_duration_cat'])

display( y_train['trip_duration_cat'].value_counts(normalize=True).sort_index() )
display( y_holdout['trip_duration_cat'].value_counts(normalize=True).sort_index() )

y_train = y_train.drop(columns=['trip_duration_cat'])
y_holdout = y_holdout.drop(columns=['trip_duration_cat'])


# Fit:

# In[ ]:


# linreg = LinearRegression(fit_intercept=False)
# linreg.fit(X_train, y_train)
treereg = DecisionTreeRegressor(random_state=40, max_depth=7).fit(X_train, y_train)


# Predict:

# In[ ]:


y_pred = treereg.predict(X_holdout)
mean_squared_log_error( y_holdout, y_pred )


# # Submission

# In[ ]:


test['db_predict']     = dbscan_predict(dbscan, test[['pickup_longitude', 'pickup_latitude']].as_matrix())
test['kmeans_predict'] = kmeans.predict(test[['pickup_longitude', 'pickup_latitude']])

test['lon_lat_manhattan']    = abs(test['dropoff_longitude']-test['pickup_longitude']) + abs(test['dropoff_latitude']-test['pickup_latitude'])
test['dist_manhattan_meter'] = test.apply( lambda x: lat_lon_converter(x['pickup_latitude'], 
                                                                         x['pickup_longitude'],
                                                                         x['dropoff_latitude'], 
                                                                         x['dropoff_longitude'],
                                                                         'm'), axis=1 )
test['pickup_dt']    = pd.to_datetime(test['pickup_datetime'], format='%Y-%m-%d %H:%M:%S', errors='ignore')
test['pick_minute']     = test['pickup_dt'].dt.minute
test['pick_hour']       = test['pickup_dt'].dt.hour
test['pick_day']        = test['pickup_dt'].dt.day
test['pick_month']      = test['pickup_dt'].dt.month
test['pick_year']       = test['pickup_dt'].dt.year
test['pick_quarter']    = test['pickup_dt'].dt.quarter
test['pick_weekofyear'] = test['pickup_dt'].dt.weekofyear

test_x = test[ _FILTER_INT.union(_FILTER_FLOAT).union(_FILTER_CAT) ]
test_prepared = full_pipeline.fit_transform(test_x)


# In[ ]:


pred = treereg.predict(test_prepared)

df_test['trip_duration'] = pred.astype(int)
out = df_test[['id', 'trip_duration']]
out.to_csv('pred_treereg.csv',index=False)


# In[ ]:


FileLink(r'pred_treereg.csv')


# - - -
