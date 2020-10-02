#!/usr/bin/env python
# coding: utf-8

# Applying discoveries and ideas from both, [my first attempt](https://www.kaggle.com/thiagodsd/c-nyc-taxi-trip-duration-v0) to solve the probleam and the [study of regression techniques](https://www.kaggle.com/thiagodsd/study-nyc-taxi-trip-duration) applied to the problem data.
# 
# Since my first submission I think the most important insight was to notice the predictions residuals increasing with the trip duration. Besides, I experimented 5 models - Linear, Polynomial, Ridge, Lasso and Elastic Net Regression - and the Polynomial Regression of degree 3 performed slightly better than other regressors.
# 
# So I think a good strategy might be **to split train set into "inliers" and "outliers" and then to blend the models giving tiny weights to outliers models and a little more importance to polynomial models**.
# 
# - - -
# 
# Main references:
# 
# 1. [Outlier Detection Practice: uni/multivariate | Kaggle](https://www.kaggle.com/kevinarvai/outlier-detection-practice-uni-multivariate) - since data cleaning was a step which stuck me a little. Thanks to [Kevin Arvai](https://www.kaggle.com/kevinarvai).
# 2. [7 Regression Types and Techniques in Data Science](https://www.analyticsvidhya.com/blog/2015/08/comprehensive-guide-regression/)
#     - [Linear, Ridge and Lasso Regression comprehensive guide for beginners](https://www.analyticsvidhya.com/blog/2017/06/a-comprehensive-guide-for-linear-ridge-and-lasso-regression/)
#     - [Neural Network Model for House Prices (Keras) | Kaggle](https://www.kaggle.com/diegosiebra/neural-network-model-for-house-prices-keras) - thanks to [Diego Siebra Ferreira](https://www.kaggle.com/diegosiebra).
#     - [Regression, Blending and Stacking V.02 | Kaggle](https://www.kaggle.com/zugariy/regression-blending-and-stacking-v-02) - thanks to [ZugariY](https://www.kaggle.com/zugariy).
# 3. [Blending of 6 Models (Top 10%) | Kaggle](https://www.kaggle.com/sandeepkumar121995/blending-of-6-models-top-10) - thanks to [Sandeep Kumar](https://www.kaggle.com/sandeepkumar121995).
# 4. [Feature Engineering Benchmarks | Kaggle](https://www.kaggle.com/nanomathias/feature-engineering-benchmarks) - thanks to [NanoMathias](https://www.kaggle.com/nanomathias)
# 
# - - -
# 
# Summoning some libs:

# In[ ]:


import datetime
import warnings
import pickle
import gc
import os
import math
import pprint
import tqdm

import numpy  as np
import pandas as pd

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn           as sns
from pandas.plotting import scatter_matrix

from sklearn import metrics
from sklearn.model_selection import train_test_split, ShuffleSplit, GridSearchCV, RandomizedSearchCV, KFold
from sklearn.impute          import SimpleImputer
from sklearn.compose         import ColumnTransformer
from sklearn.preprocessing   import OrdinalEncoder, OneHotEncoder, PolynomialFeatures
from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import StandardScaler

from sklearn.cluster      import KMeans, DBSCAN
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree         import DecisionTreeRegressor
from sklearn.ensemble     import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_log_error

from IPython.display import display, FileLink

warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = [12, 6]
#np.random.seed(1642)


# Defining some functions:

# In[ ]:


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


# In[ ]:


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
            return distance
        elif unit == 'km':
            return distance
    except ValueError:
        return np.nan


# In[ ]:


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


# In[ ]:


def out_std(data, n_std=3.0, return_thresholds=False):
    """
    ref: https://www.kaggle.com/kevinarvai/outlier-detection-practice-uni-multivariate#Parametric-methods:-Univariate
    """
    mean, std    = data.mean(), data.std()
    cutoff       = std * n_std
    lower, upper = mean - cutoff, mean + cutoff
    if return_thresholds:
        return lower, upper
    return [True if i < lower or i > upper else False for i in data]

def out_iqr(data, k=1.5, return_thresholds=False):
    """
    ref: https://www.kaggle.com/kevinarvai/outlier-detection-practice-uni-multivariate#Parametric-methods:-Univariate
    """
    q25, q75     = np.percentile(data, 25), np.percentile(data, 75)
    iqr          = q75 - q25
    cutoff       = iqr * k
    lower, upper = q25 - cutoff, q75 + cutoff
    if return_thresholds:
        return lower, upper
    return [True if i < lower or i > upper else False for i in data]


# In[ ]:


def get_feature_names(columnTransformer):
    """
    ref: https://stackoverflow.com/questions/57528350/can-you-consistently-keep-track-of-column-labels-using-sklearns-transformer-api
    """
    output_features = []

    for transformers in columnTransformer.transformers_:

        if transformers[0]!='remainder':
            pipeline = transformers[1]
            features = transformers[2]

            for i in pipeline:
                trans_features = []
                if hasattr(i,'categories_'):
                    trans_features.extend(i.get_feature_names(features))
                else:
                    trans_features = features
            output_features.extend(trans_features)

    return output_features


# In[ ]:


import sys
def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)


# # 1. Loading Data & Overview

# In[ ]:


KAGGLE = True

if KAGGLE == True:
  PATH = '/kaggle/input'
else:
  from google.colab import drive
  drive.mount('/content/drive/', force_remount=True)

  DRIVE_PATH = '/content/drive/My Drive/statistical-learning'
  #PATH = '/root/data'
  PATH = DRIVE_PATH + '/data'


for dirname, _, filenames in os.walk('{}'.format(PATH)):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df_train = pd.read_csv('{}/nyc-taxi-trip-duration/train.csv'.format(PATH))
df_test  = pd.read_csv('{}/nyc-taxi-trip-duration/test.csv'.format(PATH))

print('train: ', df_train.shape)
print('test:  ', df_test.shape)

display( df_train.head() )
#display( df_test.head() )

_TARGET      = 'trip_duration'
_NON_FEATURE = set(df_train.columns) - set(df_test.columns)
_FEATURES    = set(df_train.columns).intersection(set(df_test.columns)) - set(['id'])

display(_FEATURES)


# In[ ]:


df_train.describe().apply( lambda s: s.apply( lambda x: format(x, '.3f') ) )


# # 2. Exploration

# **Features distributions:**
# 
# It will be clear that all the relevant features have outliers, specially `trip_duration`.

# In[ ]:


df_train[df_train.dtypes[df_train.dtypes != object].index].hist(bins=100, grid='off');


# A look to the time evolution of `trip_duration`. Here it can be seen that the older the faster the trips, so mayme it makes sense to create a time tag with the format "YEAR-MONTH-DAY".

# In[ ]:


get_ipython().run_cell_magic('time', '', "fig, axes = plt.subplots(ncols=2)\nax1, ax2 = axes\n\ndf_train['pickup_dt'] = pd.to_datetime(df_train['pickup_datetime'], format='%Y-%m-%d %H:%M:%S', errors='ignore')\ndf_train.set_index('pickup_dt').groupby(pd.Grouper(freq='D'))['trip_duration'].mean().plot(ax=ax1)\ndf_train.set_index('pickup_dt').groupby(pd.Grouper(freq='D'))['trip_duration'].count().plot(ax=ax2)\nax1.set_title('mean')\nax2.set_title('count');")


# **Dealing with outiliers:**
# 
# Here I prefer to define two kinds of outliers: _spatial outliers_ and _temporal outliers_.

# In[ ]:


get_ipython().run_cell_magic('time', '', "space_out_pick = df_train[['pickup_latitude', 'pickup_longitude']].apply(out_iqr, k=1.5)\nspace_out_drop = df_train[['dropoff_latitude', 'dropoff_longitude']].apply(out_iqr, k=1.5)\ntime_out       = df_train[['trip_duration']].apply(out_iqr, k=1.5)")


# In[ ]:


f, ((ax1, ax2, ax3)) = plt.subplots(ncols=3, nrows=1)

sns.heatmap(space_out_pick, cmap='binary', ax=ax1)
sns.heatmap(space_out_drop, cmap='binary', ax=ax2)
sns.heatmap(time_out,       cmap='binary', ax=ax3)

ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)
ax1.axes.get_yaxis().set_visible(False)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90)
ax2.axes.get_yaxis().set_visible(False)
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=90)
ax3.axes.get_yaxis().set_visible(False)

display( np.unique(space_out_pick, return_counts=True) )
display( np.unique(space_out_drop, return_counts=True) )
display( np.unique(time_out,       return_counts=True) )


# In[ ]:


df_train_nout = df_train[~( np.any(np.concatenate([space_out_pick, space_out_drop, time_out], axis=1), axis=1) )]

display( df_train.shape )
display( df_train_nout.shape )


# Here we go again. Now the skewed shape of `trip_duration` is more discernible. So it makes sense to log-transform it.

# In[ ]:


df_train_nout[df_train_nout.dtypes[df_train_nout.dtypes != object].index].hist(bins=100, grid='off');


# In[ ]:


df_train['log_trip_duration']      = np.log1p(df_train['trip_duration'])
df_train_nout['log_trip_duration'] = np.log1p(df_train_nout['trip_duration'])


# Finally, a look on the spatial distribution features.

# In[ ]:


get_ipython().run_cell_magic('time', '', "f, ((ax1, ax2)) = plt.subplots(ncols=2, nrows=1)\n\ndf_train_nout.plot(kind='scatter', x='pickup_longitude', y='pickup_latitude', ax=ax1, alpha=0.25)\nax1.set_title('pickup')\ndf_train_nout.plot(kind='scatter', x='dropoff_longitude', y='dropoff_latitude', ax=ax2, alpha=0.25)\nax2.set_title('dropoff');")


# It is clear then that there is a region which most of trips are associated with...
# 
# ```
# > typing "df_train_nout.sample(1)[['pickup_latitude', 'pickup_longitude']]"
#                 pickup_latitude	pickup_longitude
#         632855	40.763489	    -73.981239
# > googling "lat lon finder"
# ```
# 
# <div style="text-align:center;">
#     <div style="display:table;">
#          <div style="display:table-cell; vertical-align:middle;padding-right:5px;">
#               <img src="https://emojis.slackmojis.com/emojis/images/1450319446/63/fry.png?1450319446">
#          </div>
#          <div style="display:table-cell; vertical-align:middle;padding-left:5px;">
#               <img src="https://2b1ohome.files.wordpress.com/2019/10/nyc.png">
#          </div>
#     </div>
# </div>
# 
# .
# .
# .

# I might be wrong, but I have strong feelings that this is the shape of New York City. 
# 
# Moreover pickup and dropoff distributions are slightly different.

# Clustering coordinates might be a good way to identify particularities from different regions of the city.
# **DBSCAN** seems to be a good technique for this task, but since it is usually too expensive, it is reasonable to run it over a sample. 

# In[ ]:


sample = df_train_nout[['dropoff_latitude', 'dropoff_longitude',
                        'pickup_latitude',  'pickup_longitude',
                        'trip_duration']].sample(20000, random_state=13)


# In[ ]:


get_ipython().run_cell_magic('time', '', "f, ((ax1, ax2)) = plt.subplots(ncols=2, nrows=1, sharey=True, sharex=True)\n\nsample.plot(kind='scatter', x='pickup_longitude', y='pickup_latitude', ax=ax1, alpha=0.25)\nax1.set_title('pickup')\nsample.plot(kind='scatter', x='dropoff_longitude', y='dropoff_latitude', ax=ax2, alpha=0.25)\nax2.set_title('dropoff');")


# In[ ]:


get_ipython().run_cell_magic('time', '', "X       = StandardScaler().fit_transform(sample[['pickup_longitude', 'pickup_latitude']].values)\ndb_pick = DBSCAN(eps=0.35, min_samples=10).fit(X)\nlabels  = db_pick.labels_\n\ndisplay( np.unique(labels, return_counts=True) )\n\nunique_labels = set(labels)\n\nfor label in unique_labels:\n    sample_mask = [True if l == label else False for l in labels]\n    plt.plot(sample['pickup_longitude'][sample_mask], sample['pickup_latitude'][sample_mask], 'o', label=label);\nplt.title('pickup')\nplt.legend()\nplt.show();\n\n#\n\nX       = StandardScaler().fit_transform(sample[['dropoff_longitude', 'dropoff_latitude']].values)\ndb_drop = DBSCAN(eps=0.25, min_samples=10).fit(X)\nlabels  = db_drop.labels_\n\ndisplay( np.unique(labels, return_counts=True) )\n\nunique_labels = set(labels)\n\nfor label in unique_labels:\n    sample_mask = [True if l == label else False for l in labels]\n    plt.plot(sample['dropoff_longitude'][sample_mask], sample['dropoff_latitude'][sample_mask], 'o', label=label);\nplt.title('dropoff')\nplt.legend()\nplt.show();")


# In[ ]:


f, ((ax1, ax2)) = plt.subplots(ncols=2, nrows=1, sharey=True)

sample_p         = sample[['pickup_longitude', 'pickup_latitude', 'trip_duration']].copy()
sample_p['db_p'] = db_pick.labels_
sample_p.groupby('db_p')['trip_duration'].mean().plot(kind='bar', ax=ax1)

sample_d         = sample[['dropoff_longitude', 'dropoff_latitude', 'trip_duration']].copy()
sample_d['db_d'] = db_drop.labels_
sample_d.groupby('db_d')['trip_duration'].mean().plot(kind='bar', ax=ax2)


# Additionally, **KMEANS** is a good way to define important spots and then to associate pickups and dropoff coordinates to them. I don't know the dynamics and peculiarities of New York and Manhattan, so I thought bridges were good choices to set the KMEANS centroids.
# 
# ![](https://2b1ohome.files.wordpress.com/2019/10/maps.png)
# 
# Below there are the spots coordinates that I found by clicking in on Google Maps: 
# 
# ```
# Randalls Island
# New York, NY, USA
# 40.789674, -73.924192
# 
# Upper East Side
# New York, NY, USA
# 40.758493, -73.958586
# 
# Murray Hill
# New York, NY, USA
# 40.747579, -73.972331
# 
# Lower East Side
# New York, NY, USA
# 40.715072, -73.976137
# 
# Two Bridges
# New York, NY, USA
# 40.709359, -73.995355
# 
# Lower Manhattan
# New York, NY, USA
# 40.702863, -74.015257
# 
# Hudson Square
# New York, NY, USA
# 40.726009, -74.010788
# 
# Hell's Kitchen
# New York, NY, USA
# 40.759815, -74.002534
# ```

# In[ ]:


_CENTROIDS = np.array([[40.789674, -73.924192],
                        [40.758493, -73.958586],
                        [40.747579, -73.972331],
                        [40.715072, -73.976137],
                        [40.709359, -73.995355],
                        [40.702863, -74.015257],
                        [40.726009, -74.010788],
                        [40.759815, -74.002534]])

kmeans = KMeans(n_clusters=8, random_state=37)
kmeans.cluster_centers_ = _CENTROIDS
sample_p['kmeans_p'] = kmeans.predict(sample_p[['pickup_latitude', 'pickup_longitude']])
sample_d['kmeans_d'] = kmeans.predict(sample_d[['dropoff_latitude', 'dropoff_longitude']])


# In[ ]:


f, ((ax1, ax2)) = plt.subplots(ncols=2, nrows=1)

for label in sample_p['kmeans_p'].unique():
    sample_mask = [True if l == label else False for l in sample_p['kmeans_p']]
    ax1.plot(sample_p['pickup_latitude'][sample_mask], sample_p['pickup_longitude'][sample_mask], 'o', label=label)

for label in sample_d['kmeans_d'].unique():
    sample_mask = [True if l == label else False for l in sample_d['kmeans_d']]
    ax2.plot(sample_d['dropoff_latitude'][sample_mask], sample_d['dropoff_longitude'][sample_mask], 'o', label=label)

ax1.set_title('pickup')
ax2.set_title('dropoff')
plt.legend()
plt.show();


# In[ ]:


f, ((ax1, ax2)) = plt.subplots(ncols=2, nrows=1, sharey=True)

sample_p.groupby('kmeans_p')['trip_duration'].mean().plot(kind='bar', ax=ax1)
sample_d.groupby('kmeans_d')['trip_duration'].mean().plot(kind='bar', ax=ax2)


# # 3. Feature Engineering & Data Imputing

# Prophylactic reloadings:

# In[ ]:


var_cleaner("df_train, df_test, _TARGET, _NON_FEATURE, _FEATURES, space_out_pick, space_out_drop, time_out, sample, df_train_nout, _CENTROIDS, X, db_pick, db_drop, kmeans")


# In[ ]:


df_train = pd.read_csv('{}/nyc-taxi-trip-duration/train.csv'.format(PATH))
df_test  = pd.read_csv('{}/nyc-taxi-trip-duration/test.csv'.format(PATH))

print('train: ', df_train.shape)
print('test:  ', df_test.shape)

#display( df_train.head() )
#display( df_test.head() )

_TARGET      = 'trip_duration'
_NON_FEATURE = set(df_train.columns) - set(df_test.columns)
_FEATURES    = set(df_train.columns).intersection(set(df_test.columns)) - set(['id'])

# display(_FEATURES)


# In[ ]:


space_out_pick = df_train[['pickup_latitude', 'pickup_longitude']].apply(out_iqr, k=1.5)
space_out_drop = df_train[['dropoff_latitude', 'dropoff_longitude']].apply(out_iqr, k=1.5)
time_out       = df_train[['trip_duration']].apply(out_iqr, k=1.0)

df_train_nout = df_train[~( np.any(np.concatenate([space_out_pick, space_out_drop, time_out], axis=1), axis=1) )]

df_train['log_trip_duration']      = np.log1p(df_train['trip_duration'])
df_train_nout['log_trip_duration'] = np.log1p(df_train_nout['trip_duration'])

#

sample = df_train_nout[['dropoff_latitude', 'dropoff_longitude',
                        'pickup_latitude',  'pickup_longitude',
                        'trip_duration']].sample(20000, random_state=13)

_CENTROIDS = np.array([[40.789674, -73.924192],
                        [40.758493, -73.958586],
                        [40.747579, -73.972331],
                        [40.715072, -73.976137],
                        [40.709359, -73.995355],
                        [40.702863, -74.015257],
                        [40.726009, -74.010788],
                        [40.759815, -74.002534]])

X       = StandardScaler().fit_transform(sample[['pickup_longitude', 'pickup_latitude']].values)
db_pick = DBSCAN(eps=0.35, min_samples=10).fit(X)

X       = StandardScaler().fit_transform(sample[['dropoff_longitude', 'dropoff_latitude']].values)
db_drop = DBSCAN(eps=0.25, min_samples=10).fit(X)

kmeans = KMeans(n_clusters=8, random_state=37)
kmeans.cluster_centers_ = _CENTROIDS


# - - -

# Working on a sample:

# In[ ]:


sample = df_train_nout.sample(15000)
sample = sample.drop(columns=['id'])


# In[ ]:


def manual_feat_eng(_data):
    """
    pd.DataFrame -> pd.DataFrame
    """
    
    # temporal features
    _data['pickup_dt'] = pd.to_datetime(_data['pickup_datetime'], format='%Y-%m-%d %H:%M:%S', errors='ignore')
    _data              = _data.drop(columns=['pickup_datetime'])
    
    _data['pick_minute']     = _data['pickup_dt'].dt.minute
    _data['pick_hour']       = _data['pickup_dt'].dt.hour
    _data['pick_day']        = _data['pickup_dt'].dt.day
    _data['pick_month']      = _data['pickup_dt'].dt.month
    _data['pick_year']       = _data['pickup_dt'].dt.year
    _data['pick_quarter']    = _data['pickup_dt'].dt.quarter
    _data['pick_weekofyear'] = _data['pickup_dt'].dt.weekofyear
    #_data['yearmonthday']   = _data.apply( lambda x: int('{}{}{}'.format( str(x['pick_year']).zfill(4),str(x['pick_month']).zfill(2),str(x['pick_day']).zfill(2))),axis=1)
    _data['yearmonthday']    = _data.apply( lambda x: math.log((datetime.datetime(x['pick_year'],x['pick_month'],x['pick_day'],0,0) - datetime.datetime(2000,1,1)).total_seconds(), 1000), axis=1)
    
    # spatial features
    _data['lon_lat_manhattan']    = abs(_data['dropoff_longitude']-_data['pickup_longitude']) + abs(_data['dropoff_latitude']-_data['pickup_latitude'])
    _data['dist_manhattan_meter'] = _data.apply( lambda x: lat_lon_converter(x['pickup_latitude'], 
                                                                             x['pickup_longitude'],
                                                                             x['dropoff_latitude'], 
                                                                             x['dropoff_longitude'],
                                                                             'km'), axis=1 )
    
    _data['db_pick'] = dbscan_predict(db_pick, StandardScaler().fit_transform(_data[['pickup_longitude', 'pickup_latitude']].values))
    _data['db_drop'] = dbscan_predict(db_drop, StandardScaler().fit_transform(_data[['dropoff_longitude', 'dropoff_latitude']].values))
    
    _data['kmeans_pick'] = kmeans.predict(_data[['pickup_latitude', 'pickup_longitude']])
    _data['kmeans_drop'] = kmeans.predict(_data[['dropoff_latitude', 'dropoff_longitude']])
    
    _data['db_change'] = _data['db_pick']     != _data['db_drop']
    _data['km_change'] = _data['kmeans_pick'] != _data['kmeans_drop']

    _data['pick_km_dist'] =  _data.apply( lambda x: lat_lon_converter(  x['pickup_latitude'], 
                                                                          x['pickup_longitude'],
                                                                         _CENTROIDS[x['kmeans_pick']][0], 
                                                                         _CENTROIDS[x['kmeans_pick']][1],
                                                                         'km'), axis=1 )
    _data['drop_km_dist'] =  _data.apply( lambda x: lat_lon_converter(  x['dropoff_latitude'], 
                                                                          x['dropoff_longitude'],
                                                                         _CENTROIDS[x['kmeans_drop']][0], 
                                                                         _CENTROIDS[x['kmeans_drop']][1],
                                                                         'km'), axis=1 )

    _data['pick_drop_km_dist'] =  _data.apply( lambda x: lat_lon_converter(_CENTROIDS[x['kmeans_pick']][0], 
                                                                             _CENTROIDS[x['kmeans_pick']][1],
                                                                             _CENTROIDS[x['kmeans_drop']][0], 
                                                                             _CENTROIDS[x['kmeans_drop']][1],
                                                                             'm'), axis=1 )

    return _data


# In[ ]:


get_ipython().run_cell_magic('time', '', 'mfe_sample  = manual_feat_eng(sample)')


# In[ ]:


def reg_feat_imput(_data):
    """
    pd.DataFrame -> np.array
    """
    
    _FILTERS = {
            'int'   : [ [None], [np.dtype('int64')]   ],
            'float' : [ [None], [np.dtype('float64')] ]
           }

    for k in _FILTERS:
        _FILTERS[k][0] = set(_data.dtypes[ _data.dtypes.isin(_FILTERS[k][1]) ].index.to_list())
        #    print(k)
        #    pprint.pprint( _FILTERS[k][0], indent=4)
    
    int_pipeline   = Pipeline([ ('imputer', SimpleImputer(strategy="constant", fill_value=0)),
                                ('std_scaler', StandardScaler())])
    float_pipeline = Pipeline([ 
                              ('imputer'   , SimpleImputer(strategy="median")),
                              ('std_scaler', StandardScaler())
                              ])

    full_pipeline = ColumnTransformer([
                                      ('int'  , int_pipeline,    list(_FILTERS['int'][0]) ),
                                      ('float', float_pipeline,  list(_FILTERS['float'][0]) )
                                      ])
    
    train_x = full_pipeline.fit_transform( _data[set.union(*[_FILTERS['int'][0], _FILTERS['float'][0]])] )
    
    return train_x, full_pipeline


# In[ ]:


df_train_nout.shape


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nsample = df_train_nout.sample(frac=0.1)\ntrain  = manual_feat_eng(sample)\n#train = feat_eng(df_train_nout)')


# In[ ]:


_FILTERS = {
            'int'   : [ [None], [np.dtype('int64')] ],
            'float' : [ [None], [np.dtype('float64')] ],
            'cat'   : [ [None], [np.dtype('object')] ],
            'bool'  : [ [None], [np.dtype('bool')] ],
            'date'  : [ [None], [np.dtype('<M8[ns]')] ]
           }

for k in _FILTERS:
    _FILTERS[k][0] = set(train.dtypes[ train.dtypes.isin(_FILTERS[k][1]) ].index.to_list())
    print(k)
    pprint.pprint( _FILTERS[k][0], indent=4)


# # Regressors

# In[ ]:


train_x, full_pipeline = reg_feat_imput(train[['passenger_count', 
                                               'pick_day', 
                                               'pick_hour',
                                               'pick_minute', 
                                               'pick_month', 
                                               'pick_quarter', 
                                               'pick_weekofyear', 
                                               'pick_year',
                                               'dist_manhattan_meter', 
                                               'drop_km_dist', 
                                               'dropoff_latitude',
                                               'dropoff_longitude', 
                                               'lon_lat_manhattan',
                                               'pick_drop_km_dist', 
                                               'pick_km_dist', 
                                               'pickup_latitude', 
                                               'pickup_longitude',
                                               'yearmonthday'
                                              ]])

train_y = df_train.iloc[train.index][['log_trip_duration', 'trip_duration']]    
train_y = train_y.mask(train_y['trip_duration'].lt(0), 0)

X_train, X_holdout, y_train, y_holdout = train_test_split(train_x, train_y, 
                                                          test_size    = 0.2, 
                                                          random_state = 37)

N      = int(0.5*y_holdout.shape[0])
_models = dict() 
plt.rcParams['figure.figsize'] = [18, 6]


# In[ ]:


def plot_results(_y_pred_train, _y_train, _y_pred_holdout, _y_holdout, _reg, _pipeline, _N):
    f, ((ax1, ax2)) = plt.subplots(ncols=2, nrows=1)

    ax1.hist(_y_pred_train, bins=250, alpha=0.25, color='r', label='pred', normed=True)
    ax1.hist(_y_train['log_trip_duration'], bins=250, alpha=0.25, color='b', label='train', normed=True)
    ax1.set_title('TRAIN')
    plt.legend()

    ax2.hist(_y_pred_holdout, bins=250, alpha=0.25, color='r', label='pred', normed=True)
    ax2.hist(_y_holdout['log_trip_duration'], bins=250, alpha=0.25, color='b', label='holdout', normed=True)
    ax2.set_title('HOLD-OUT')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    #

    fig, axes = plt.subplots(ncols=2)
    ax1, ax2 = axes
    
    if _reg:
        pd.Series( _reg.coef_, get_feature_names(_pipeline) ).sort_values().plot(kind='bar', ax=ax1)

    _t = pd.DataFrame(list(zip(_y_train['log_trip_duration'].values, _y_pred_train.reshape(-1) )),
                      columns=['y_train', 'y_pred_train'])
    
    _h = pd.DataFrame(list(zip(_y_holdout['log_trip_duration'].values,_y_pred_holdout.reshape(-1) )), 
                      columns=['y_holdout', 'y_pred_holdout'])
    
    _t['res_train']   = _t['y_train']   - _t['y_pred_train']
    _h['res_holdout'] = _h['y_holdout'] - _h['y_pred_holdout']

    _samp = _t.sample(_N)
    _samp.plot(x='y_pred_train', y='res_train', kind='scatter', alpha=0.25, ax=ax2, label='train', color='b')
    _samp = _h.sample(_N)
    _samp.plot(x='y_pred_holdout', y='res_holdout', kind='scatter', alpha=0.25, ax=ax2, label='holdout', color='r')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    #
    
    gc.collect()


# ## Linear Regressor

# In[ ]:


linreg = LinearRegression(fit_intercept=True)
linreg.fit(X_train, y_train['log_trip_duration'])

y_pred_train   = linreg.predict(X_train)
y_pred_holdout = linreg.predict(X_holdout)

display( mean_absolute_error( y_train['trip_duration']  , np.exp(y_pred_train)-1    ) )
display( mean_absolute_error( y_holdout['trip_duration'] , np.exp(y_pred_holdout)-1 ) )
print('RMSLE:', np.sqrt(mean_squared_log_error( y_holdout['trip_duration'], (np.exp(y_pred_holdout)-1) )))

plot_results(y_pred_train, y_train, y_pred_holdout, y_holdout, linreg, full_pipeline, N)

#

_models['linreg']   = linreg
y_holdout['linreg'] = _models['linreg'].predict(X_holdout)

if KAGGLE == False:
    pickle.dump(linreg,     open(r'{}/nyc-taxi-trip-duration/model_linreg.p'.format(PATH), 'wb'))


# ## Polynomial Regressor, degree 2

# In[ ]:


d = 2
polyreg = PolynomialFeatures(degree=d)
x_poly  = polyreg.fit_transform(X_train)

linreg_p  = LinearRegression(fit_intercept=True)
linreg_p.fit(x_poly,  y_train['log_trip_duration'])

# accuracy evaluation

y_pred_train   = linreg_p.predict(x_poly)
del x_poly; gc.collect()
x_poly         = polyreg.fit_transform(X_holdout)
y_pred_holdout = linreg_p.predict(x_poly)

_models['linreg_p']   = linreg_p
y_holdout['linreg_p'] = _models['linreg_p'].predict(x_poly)
del x_poly;gc.collect()

print('\n degree={}'.format(d))
display( mean_absolute_error( y_train['trip_duration']  , np.exp(y_pred_train)-1    ) )
display( mean_absolute_error( y_holdout['trip_duration'] , np.exp(y_pred_holdout)-1 ) )
print('RMSLE:', np.sqrt(mean_squared_log_error( y_holdout['trip_duration'], (np.exp(y_pred_holdout)-1) )))

plot_results(y_pred_train, y_train, y_pred_holdout, y_holdout, None, full_pipeline, N)

if KAGGLE == False:
    pickle.dump(linreg_p,     open(r'{}/nyc-taxi-trip-duration/model_linreg_p.p'.format(PATH), 'wb'))


# ## Polynomial Regressor, degree 3

# In[ ]:


d = 3
polyreg = PolynomialFeatures(degree=d)
x_poly  = polyreg.fit_transform(X_train)

linreg_p  = LinearRegression(fit_intercept=True)
linreg_p.fit(x_poly,  y_train['log_trip_duration'])

# accuracy evaluation

y_pred_train   = linreg_p.predict(x_poly)
del x_poly; gc.collect()
x_poly         = polyreg.fit_transform(X_holdout)
y_pred_holdout = linreg_p.predict(x_poly)

_models['linreg_p']   = linreg_p
y_holdout['linreg_p'] = _models['linreg_p'].predict(x_poly)
del x_poly;gc.collect()

print('\n degree={}'.format(d))
display( mean_absolute_error( y_train['trip_duration']  , np.exp(y_pred_train)-1    ) )
display( mean_absolute_error( y_holdout['trip_duration'] , np.exp(y_pred_holdout)-1 ) )
print('RMSLE:', np.sqrt(mean_squared_log_error( y_holdout['trip_duration'], (np.exp(y_pred_holdout)-1) )))

plot_results(y_pred_train, y_train, y_pred_holdout, y_holdout, None, full_pipeline, N)

if KAGGLE == False:
    pickle.dump(linreg_p,     open(r'{}/nyc-taxi-trip-duration/model_linreg_p.p'.format(PATH), 'wb'))


# ## Ridge Regressor

# In[ ]:


get_ipython().run_cell_magic('time', '', "ridge_cv   = GridSearchCV(Ridge(fit_intercept=True), {'alpha': [2.37**i for i in range(-8, 8)]}, scoring='neg_mean_absolute_error', cv=5)\nridge_cv.fit(X_train, y_train['log_trip_duration'])\nprint(ridge_cv.best_params_['alpha'])")


# In[ ]:


ridreg = Ridge( alpha=ridge_cv.best_params_['alpha'], fit_intercept=True )
ridreg.fit(X_train, y_train['log_trip_duration'])

y_pred_train   = ridreg.predict(X_train)
y_pred_holdout = ridreg.predict(X_holdout)

display( mean_absolute_error( y_train['trip_duration']  , np.exp(y_pred_train)-1    ) )
display( mean_absolute_error( y_holdout['trip_duration'] , np.exp(y_pred_holdout)-1 ) )
print('RMSLE:', np.sqrt(mean_squared_log_error( y_holdout['trip_duration'], (np.exp(y_pred_holdout)-1) )))

_models['ridreg']   = ridreg
y_holdout['ridreg'] = _models['ridreg'].predict(X_holdout)

plot_results(y_pred_train, y_train, y_pred_holdout, y_holdout, ridreg, full_pipeline, N)

#
if KAGGLE == False:
    pickle.dump(ridreg,     open(r'{}/nyc-taxi-trip-duration/model_ridreg.p'.format(PATH), 'wb'))


# ## Random Forest Regressor

# In[ ]:


def rf_feat_imput(_data):
    """
    pd.DataFrame -> np.array
    """
    
    _FILTERS = {
            'int'   : [ [None], [np.dtype('int64')] ],
            'float' : [ [None], [np.dtype('float64')] ],
            'cat'   : [ [None], [np.dtype('object')] ],
            'bool'  : [ [None], [np.dtype('bool')] ],
            'date'  : [ [None], [np.dtype('<M8[ns]')] ]
           }

    for k in _FILTERS:
        _FILTERS[k][0] = set(_data.dtypes[ _data.dtypes.isin(_FILTERS[k][1]) ].index.to_list())
        #    print(k)
        #    pprint.pprint( _FILTERS[k][0], indent=4)
    
    int_pipeline   = Pipeline([ ('imputer', SimpleImputer(strategy="constant", fill_value=0)),
                                ('std_scaler', StandardScaler())])
    float_pipeline = Pipeline([ 
                              ('imputer'   , SimpleImputer(strategy="median")),
                              ('std_scaler', StandardScaler())
                              ])

    full_pipeline = ColumnTransformer([
                                      ('int'  , int_pipeline,    list(_FILTERS['int'][0]) ),
                                      ('float', float_pipeline,  list(_FILTERS['float'][0]) ),
                                      ('bool',  'passthrough',   list(_FILTERS['bool'][0]))
                                      ])

    train_x = full_pipeline.fit_transform( _data[set.union(*[_FILTERS['int'][0], _FILTERS['float'][0], _FILTERS['bool'][0]])] )
    
    return train_x, full_pipeline


# In[ ]:


train_x_rf, full_pipeline_rf = rf_feat_imput(train[['db_drop', 
                                              'db_pick', 
                                              'passenger_count', 
                                              'pick_day', 
                                              'pick_hour',
                                              'pick_minute', 
                                              'pick_month', 
                                              'pick_quarter', 
                                              'pick_weekofyear', 
                                              'pick_year', 
                                              #'vendor_id', 
                                              #'dist_manhattan_meter',
                                              'drop_km_dist', 
                                              'dropoff_latitude',
                                              'dropoff_longitude', 
                                              #'lon_lat_manhattan',
                                              'pick_drop_km_dist', 
                                              'pick_km_dist', 
                                              'pickup_latitude', 
                                               'pickup_longitude',
                                              'yearmonthday', 
                                              'db_change', 
                                              'km_change'
                                                ]])
train_y = df_train.iloc[train.index][['log_trip_duration', 'trip_duration']]    
train_y = train_y.mask(train_y['trip_duration'].lt(0), 0)

X_train, X_holdout, y_train, y_holdout = train_test_split(train_x_rf, train_y, 
                                                          test_size    = 0.2, 
                                                          random_state = 37)

N      = int(0.5*y_holdout.shape[0])


# In[ ]:


"""
%%time
rf = RandomForestRegressor(random_state=13)

param_dist = {"max_depth"    : [None, 5, 10, 20],
              "n_estimators" : [10, 100, 1000],
              "max_features" : [None, 3, 6],
              "bootstrap": [True, False],
              "criterion": ["mae"]}
random_search = RandomizedSearchCV(rf,
                                   param_distributions = param_dist,
                                   n_iter = 7, 
                                   cv     = 3, 
                                   iid    = False)

random_search.fit(X_train, y_train['log_trip_duration'])
print(random_search.best_params_)
""";

# 'n_estimators': 1000, 'max_features': None, 'max_depth': None, 'criterion': 'mse', 'bootstrap': True


# In[ ]:


get_ipython().run_cell_magic('time', '', "# rf = RandomForestRegressor(random_state=13, n_estimators= 1000, max_features= None, max_depth= None, criterion='mse', bootstrap= True)\nrf = RandomForestRegressor(random_state=13)\nrf.set_params(**{'n_estimators':250, 'min_samples_split':0.5, 'min_samples_leaf':0.25, 'max_features':'sqrt', 'max_depth':6, 'criterion':'mae', 'bootstrap':False})\n\n\nrf.fit(X_train, y_train['log_trip_duration'])\n\ny_pred_train   = rf.predict(X_train)\ny_pred_holdout = rf.predict(X_holdout)\n\ndisplay( mean_absolute_error( y_train['trip_duration']  , np.exp(y_pred_train)-1    ) )\ndisplay( mean_absolute_error( y_holdout['trip_duration'] , np.exp(y_pred_holdout)-1 ) )\nprint('RMSLE:', np.sqrt(mean_squared_log_error( y_holdout['trip_duration'], (np.exp(y_pred_holdout)-1) )))")


# In[ ]:


f, ((ax1, ax2)) = plt.subplots(ncols=2, nrows=1)

ax1.hist(y_pred_train, bins=250, alpha=0.25, color='r', label='pred', normed=True)
ax1.hist(y_train['log_trip_duration'], bins=250, alpha=0.25, color='b', label='train', normed=True)
ax1.set_title('TRAIN')
plt.legend()

ax2.hist(y_pred_holdout, bins=250, alpha=0.25, color='r', label='pred', normed=True)
ax2.hist(y_holdout['log_trip_duration'], bins=250, alpha=0.25, color='b', label='holdout', normed=True)
ax2.set_title('HOLD-OUT')
plt.legend()
plt.show()


# In[ ]:


fig, axes = plt.subplots(ncols=2)
ax1, ax2 = axes

#pd.Series( rf.coef_, get_feature_names(full_pipeline) ).sort_values().plot(kind='bar', ax=ax1)
features    = get_feature_names(full_pipeline_rf)
importances = rf.feature_importances_
indices     = np.argsort(importances)

ax1.set_title('Feature Importances')
ax1.barh(range(len(indices)), importances[indices], color='b', align='center')
ax1.set_yticks(range(len(indices)), [features[i] for i in indices])
ax1.set_xlabel('Relative Importance')

_t = pd.DataFrame(list(zip(y_train['log_trip_duration'].values, y_pred_train.reshape(-1))),
                  columns=['y_train', 'y_pred_train'])
_t['res_train'] = _t['y_train'] - _t['y_pred_train']
_h = pd.DataFrame(list(zip(y_holdout['log_trip_duration'].values, y_pred_holdout.reshape(-1))), 
                  columns=['y_holdout', 'y_pred_holdout'])
_h['res_holdout'] = _h['y_holdout'] - _h['y_pred_holdout']

_samp = _t.sample(N)
_samp.plot(x='y_pred_train', y='res_train', kind='scatter', alpha=0.25, ax=ax2, label='train', color='b')
_samp = _h.sample(N)
_samp.plot(x='y_pred_holdout', y='res_holdout', kind='scatter', alpha=0.25, ax=ax2, label='holdout', color='r')

plt.legend()
gc.collect()


# In[ ]:


features = get_feature_names(full_pipeline_rf)
importances = rf.feature_importances_
indices = np.argsort(importances)

plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.tight_layout()
plt.show()


# - - -
# 
# |      model    | mae train | mae holdout |
# |:-------------:|:---------:|:-----------:|
# |     linear    |  227.075  |   224.390   |
# |  polynomial 2 |  177.369  |   178.051   |
# |  polynomial 3 |  159.872  |   165.059   |
# |     ridge     |  226.598  |   223.906   |
# | random forest |   65.618  |   168.172   |

# - - - 
