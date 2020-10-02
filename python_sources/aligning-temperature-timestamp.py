#!/usr/bin/env python
# coding: utf-8

# Hi guys!
# <br>In this notebook I want to share with you semi-automated approach of **timestamp alignment in weather data**
# <br>According to this [thread](https://www.kaggle.com/c/ashrae-energy-prediction/discussion/114483#latest-659257), with high probability there are discrepancies between measurement timestamps and weather timestamps. 
# <br>Particularly, it looks like weather data is not in **local time format**
# <br> Let's gently fix that!

# In[ ]:


get_ipython().system('pip install cufflinks')


# ### Load raw weather data

# In[ ]:


from os.path import join as pjoin
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cufflinks as cf
cf.go_offline(connected=False)  # to make it works without plotly account


# In[ ]:


RAW_DATA_DIR = '/kaggle/input/ashrae-energy-prediction/'

print('Loading init weather data...')
# load and concatenate weather data
weather_dtypes = {
    'site_id': np.uint8,
    'air_temperature': np.float32,
    'cloud_coverage': np.float32,
    'dew_temperature': np.float32,
    'precip_depth_1_hr': np.float32,
    'sea_level_pressure': np.float32,
    'wind_direction': np.float32,
    'wind_speed': np.float32,
}

weather_train = pd.read_csv(
    pjoin(RAW_DATA_DIR, 'weather_train.csv'),
    dtype=weather_dtypes,
    parse_dates=['timestamp']
)
weather_test = pd.read_csv(
    pjoin(RAW_DATA_DIR, 'weather_test.csv'),
    dtype=weather_dtypes,
    parse_dates=['timestamp']
)

weather = pd.concat(
    [
        weather_train,
        weather_test
    ],
    ignore_index=True
)
# del redundant dfs
del weather_train, weather_test

weather.head()


# In[ ]:


weather_key = ['site_id', 'timestamp']
temp_skeleton = weather[weather_key + ['air_temperature']].drop_duplicates(subset=weather_key).sort_values(by=weather_key).copy()
# check sample
temp_skeleton.head()


# What we are going to do is to calculate **ranks of hourly temperatures within date/site_id chunks**, then average it across all timestamps, re-scale them to [0,1] and  visualize via interactive heatmap
# 
# Then, assuming `(1,5,12)` tuple has the most correct temp peaks at 14:00, we can calculate offsets for the other `site_id`'s
# <br>As a side effect, we can notice clusters like `(0,8), (7,11), (1,5,12)`

# In[ ]:


# calculate ranks of hourly temperatures within date/site_id chunks
temp_skeleton['temp_rank'] = temp_skeleton.groupby(
    ['site_id', temp_skeleton.timestamp.dt.date],
)['air_temperature'].rank('average')

# create 2D dataframe of site_ids (0-16) x mean hour rank of temperature within day (0-23)
df_2d = temp_skeleton.groupby(
    ['site_id', temp_skeleton.timestamp.dt.hour]
)['temp_rank'].mean().unstack(level=1)

# align scale, so each value within row is in [0,1] range
df_2d = df_2d / df_2d.max(axis=1).values.reshape((-1,1))  

# sort by 'closeness' of hour with the highest temperature
site_ids_argmax_maxtemp = pd.Series(np.argmax(df_2d.values, axis=1)).sort_values().index

# assuming (1,5,12) tuple has the most correct temp peaks at 14:00
site_ids_offsets = pd.Series(df_2d.values.argmax(axis=1) - 14)

# align rows so that site_id's with similar temperature hour's peaks are near each other
df_2d = df_2d.iloc[site_ids_argmax_maxtemp]
df_2d.index = [f'idx={i:02d}_site_id={s:02d}' for (i,s) in zip(range(16), df_2d.index)]

# build heatmap
df_2d.T.iplot(
    kind='heatmap', 
    colorscale='ylorrd', 
    xTitle='hours, 0-23', 
    title='Mean temperature rank by hour (init timestamps)',
)


# In[ ]:


# check what offsets (in hours) we have
site_ids_offsets.index.name = 'site_id'
site_ids_offsets.sort_values()


# In[ ]:


temp_skeleton['offset'] = temp_skeleton.site_id.map(site_ids_offsets)

# add offset
temp_skeleton['timestamp_aligned'] = (
    temp_skeleton.timestamp 
    - pd.to_timedelta(temp_skeleton.offset, unit='H')
)

temp_skeleton.head()


# In[ ]:


# check difference now
temp_skeleton['temp_rank'] = temp_skeleton.groupby(
    ['site_id', temp_skeleton.timestamp_aligned.dt.date],
)['air_temperature'].rank('max')

# create 2D dataframe of site_ids (0-16) x mean hour rank of temperature within day (0-23)
df_2d = temp_skeleton.groupby(
    ['site_id', temp_skeleton.timestamp_aligned.dt.hour]
)['temp_rank'].mean().unstack(level=1)
df_2d = df_2d / df_2d.max(axis=1).values.reshape((-1,1))

df_2d.T.iplot(
    kind='heatmap', 
    colorscale='ylorrd', 
    xTitle='hours, 0-23', 
    yTitle='site_id', 
    title='Mean temperature rank by hour (aligned timestamps)',
)


# Well, this distribution looks **much more aligned to me**
# <br>We might stop here, but let us also check impact of our actions on **weather correlation with the log(target)**

# In[ ]:


# load train data
print('Reading train data...')
train = pd.read_csv(
    pjoin(RAW_DATA_DIR, 'train.csv'),
    dtype={
        'building_id': np.uint16,
        'meter': np.uint8,
        'meter_reading': np.float32,
    },
    parse_dates=['timestamp'],
)

TARGET_INIT = 'meter_reading'
TARGET = TARGET_INIT + '_log'
train[TARGET] = np.log1p(train[TARGET_INIT])

# load building metadata to get `site_id`s
print('Reading building metadata...')
building_data = pd.read_csv(
    pjoin(RAW_DATA_DIR, 'building_metadata.csv'),
    dtype={
        'site_id': np.uint8,
        'building_id': np.uint16,
        'square_feet': np.float32,
        'floor_count': np.float32,
    },
)

train['site_id'] = train.building_id.map(building_data.set_index('building_id')['site_id'])

print('filtering meter data...')
# drop irrelevant zeroes for site_id == 0, meter == 0 for first 140 days
corrupted_data_idx = (
    (train.meter == 0)
    & (train.site_id == 0)
    & (train.timestamp.dt.dayofyear < 140)
)

print(train.shape)
train = train[~corrupted_data_idx]
print(train.shape)
train.head()


# In[ ]:


# construct df with initial timestamps
df_init = pd.merge(
    left=train, 
    right=temp_skeleton,
    on=weather_key
)
df_init.head()


# ### Let's calculate correlations per `meter` group

# In[ ]:


from tqdm import tqdm_notebook as tqdm

groups = df_init[['meter', 'site_id']].drop_duplicates().values.tolist()
groups = list(tuple(e) for e in groups)  # make it immutable
correlations_init = dict()

weather_features = ['air_temperature']

# get correlations, spearman - to catch monotonic but less linear dependencies, that pearson allows
for (m, sid) in tqdm(groups):
    idx = (df_init.meter == m) & (df_init.site_id == sid)
    corrs = df_init.loc[idx, weather_features].corrwith(df_init.loc[idx, TARGET], method='spearman')
    correlations_init[(m, sid)] = dict(corrs)

# create dataframe from it
df_corr_init = pd.DataFrame(correlations_init).T.sort_index()
df_corr_init.index = df_corr_init.index.set_names(['meter', 'site_id'])
df_corr_init = df_corr_init.unstack(level=[0])
df_corr_init.style.highlight_null().format("{:.2%}")


# In[ ]:


# let's move to aligned timestamps
df_aligned = pd.merge(
    left=train,
    right=temp_skeleton,
    left_on=weather_key,
    right_on=['site_id', 'timestamp_aligned']
)
df_aligned.head()


# In[ ]:


# do the same for aligned timestamps
correlations_aligned = dict()

for (m, sid) in tqdm(groups):
    idx = (df_aligned.meter == m) & (df_aligned.site_id == sid)
    corrs = df_aligned.loc[idx, weather_features].corrwith(df_aligned.loc[idx, TARGET], method='spearman')
    correlations_aligned[(m, sid)] = dict(corrs)

# create dataframe from it
df_corr_aligned = pd.DataFrame(correlations_aligned).T.sort_index()
df_corr_aligned.index = df_corr_aligned.index.set_names(['meter', 'site_id'])
df_corr_aligned = df_corr_aligned.unstack(level=[0])
df_corr_aligned.style.highlight_null().format("{:.2%}")


# ### Let's visualize p.p. differences we get in correlations

# In[ ]:


def color_values(val):
    if val < 0:
        color = 'red'
    elif val == 0:
        color = 'blue'
    else:
        color = 'green'
    return 'color: %s' % color


(df_corr_aligned - df_corr_init).groupby(level=[0]).mean().style.format("{:.2%}").applymap(color_values).highlight_null()


# ### Well, looks like we got better results
# 
# However, maybe we missed with +-1 hour for `site_id` **14**, I didn't explicitly check that
# 
# Hope you found this kernel useful
# <br>Happy kaggling!
# 
# **P.s.** 
# <br>Comments, likes and new ideas are **always welcomed**!
# <br>Check my latest notebooks:
# 
# ---
# - [Faster stratified cross-validation](https://www.kaggle.com/frednavruzov/faster-stratified-cross-validation-upd)
# - [NaN restoration techniques for weather data](https://www.kaggle.com/frednavruzov/nan-restoration-techniques-for-weather-data)
