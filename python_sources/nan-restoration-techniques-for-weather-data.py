#!/usr/bin/env python
# coding: utf-8

# Hi guys!
# In this notebook I want to share with you my baseline approaches of handling missing values in time-dependent weather data, in particular:
# * finding consecutive missing blocks and their position/length
# * correspondent (interactive) visualization example with `cufflinks` library
# * creation of simple feature-lightgbm-based imputer to handle longer missing chunks
# * **have not yet figured out what to write next :) **

# ### I had to install latest `cufflinks` version due to internal `plotly` errors in original docker image

# In[ ]:


get_ipython().system('pip install -U cufflinks')


# In[ ]:


import pandas as pd
import numpy as np
import gc
import cufflinks as cf
cf.go_offline(connected=False)  # to make it works without plotly account
from os.path import join as pjoin


# ### Load weather data

# In[ ]:


RAW_DATA_DIR = '/kaggle/input/ashrae-energy-prediction/'

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

unique_site_ids = sorted(np.unique(weather['site_id']))
weather = weather.set_index(['site_id', 'timestamp'], drop=False).sort_index()

# construct full index w/o missing dates
full_index = pd.MultiIndex.from_product(
    [
        unique_site_ids, 
        pd.date_range(start='2016-01-01 00:00:00', end='2018-12-31 23:00:00', freq='H')
    ]
)

print(f'init shape: {weather.shape}')
weather = weather.reindex(full_index)
print(f'full shape: {weather.shape}')

weather['site_id'] = weather.index.get_level_values(0).astype(np.uint8)
weather['timestamp'] = weather.index.get_level_values(1)

# drop redundant dfs
del weather_train, weather_test
gc.collect()

print(weather.dtypes)

# check missing values
print(weather.isnull().sum())

# check data sample
weather.head()


# ### define supportives to work with NaNs

# In[ ]:


def get_nan_sequences(series: pd.Series, thld_nan: int = 2):
    """
    Given sequence with missing data, builds joint index
    from consecutive NaN blocks
    1) of len  < thld_nan
    2) of len >= thld_nan
    and returns them as 1-D np.arrays

    thld_nan >= 2
    solution is based on:
    https://stackoverflow.com/questions/42078259/indexing-a-numpy-array-using-a-numpy-array-of-slices
    """
    b = series.values

    idx0 = np.flatnonzero(np.r_[True, np.diff(np.isnan(b)) != 0, True])
    count = np.diff(idx0)
    idx = idx0[:-1]
    # >=
    valid_mask_gte = (count >= thld_nan) & np.isnan(b[idx])
    out_idx = idx[valid_mask_gte]
    out_count = count[valid_mask_gte]

    if len(out_idx) == 0:
        out_gte = np.empty(shape=0)
    else:
        out_gte = np.hstack([
            np.array(range(series, series + n))
            for (series, n) in zip(out_idx, out_count)
        ])

    # <
    valid_mask_lt = (count < thld_nan) & np.isnan(b[idx])
    out_idx = idx[valid_mask_lt]
    out_count = count[valid_mask_lt]

    if len(out_idx) == 0:
        out_lt = np.empty(shape=0)
    else:
        out_lt = np.hstack([
            np.array(range(st, st + n))
            for (st, n) in zip(out_idx, out_count)
        ])
    # check if gte + lt = all NaNs
    assert len(out_gte) + len(out_lt) == series.isnull().sum(), 'incorrect calculations'

    return out_lt, out_gte


# check distribution of consecutive NA parts in data
def plot_series_and_consequtive_nans(
    df: pd.DataFrame, 
    column: str, 
    site_id: int, 
    clip: int = 24,
    index_slice: slice = None
):
    """
    Estimates consecutive NA blocks and plots interactive timeseries with missing data
    If slice is passed - perform that steps for selected data slice only
    clips upper block length at 24 (hours)
    """
    series = weather.loc[site_id][c].copy()
    if index_slice:
        series = series.loc[index_slice]
    
    # define consecutive nan intervals
    nan_intervals = series.isnull().astype(int).groupby(
        series.notnull().astype(int).cumsum()
    ).sum().clip(0, 24).value_counts()
    
    nan_intervals = nan_intervals[nan_intervals.index != 0]
    
    nan_intervals.iplot(
        kind='bar', 
        dimensions=(240*3, 240), 
        title=f'consecutive NaNs in site "{site_id}" for column "{c}": {nan_intervals.sum()}',
        xTitle='block length, n points'
    )
    
    # to show missing values as simple interpolations
    interpolated = series.interpolate()
    to_plot = pd.DataFrame({c: series, 'missing': interpolated})
    to_plot.loc[~to_plot[c].isnull(), 'missing'] = np.nan
    
    to_plot.iplot(
        dimensions=(240*3, 320),
        title=f'site "{site_id}", timeseries for column "{c}"',
        xTitle='timestamp',
        yTitle=f'{c}'
    )


# In[ ]:


# let's see what percentage of missing values do we have in full 2016-2018 weather range
nulls_by_site_id = (weather.groupby(level=[0]).apply(
    lambda x: x.isnull().sum()) / len(weather))
nulls_by_site_id = nulls_by_site_id.loc[:, nulls_by_site_id.any()]
nulls_by_site_id.index.name = 'site_id'
nulls_by_site_id.style.format("{:.2%}").highlight_max(axis=0).highlight_min(axis=0, color='#11ff00')
# btw, we see exact duplication of site_id pairs:
# (0, 8) and (7, 11)


# In[ ]:


# let's explore some missing data
c = 'air_temperature'
site_id = 7
st  = '2017-10-01 00:00:00'
end = '2018-04-01 00:00:00'
index_slice = slice(st, end)

plot_series_and_consequtive_nans(weather, column=c, site_id=site_id, index_slice=index_slice)


# As we can see, there are **two different types** of missing blocks:
# * single (or near-single) gaps of len 1-2
# * much longer gaps **that is poorly restored** by simple interpolation techniques (see blue lines)
# 
# Let's treat them accordingly:
# * For simple missing points (even categorical) simple interpolation might be enough
# <br>this [page](https://docs.scipy.org/doc/scipy/reference/tutorial/interpolate.html) can guide you to different interpolation techniques under the hood of `pd.Series.interpolate`
# * For longer sequences feature-based imputer might come in handy
# 
# 

# In[ ]:


# let's fill shorter blocks (of len 1-2) with handy interpolation
# and then train imputer model to fit longer NaN sequences

thld = 3
indexes_gte = []
cols_to_fill = [
    'air_temperature',
    'dew_temperature',
    'precip_depth_1_hr',
    'cloud_coverage',
]

int_cols = [
    # it's ordered but lives in integer scale
    'cloud_coverage'
]

nans_total = weather[cols_to_fill].isnull().sum().sum()
nans_filled = 0
for col in cols_to_fill:
    print(f'filling short NaN series in col "{col}"')
    dtype = np.int8 if col in int_cols else np.float32
    for sid in sorted(weather.site_id.unique()):
        print(f'\tfor site_id: "{sid}"')
        s = weather.loc[sid, col].copy()
        idx_lt, idx_gte = get_nan_sequences(s, thld)
        interpolation = s.interpolate()
        nans_before = weather.loc[sid, col].isnull().sum()
        print(f'\t\tnans before: {nans_before}')
        weather.loc[sid, col].iloc[idx_lt] = interpolation.iloc[idx_lt].values.astype(dtype)
        nans_after = weather.loc[sid, col].isnull().sum()
        print(f'\t\tnans  after: {nans_before}')
        nans_filled += (nans_before - nans_after)

print(f'Nans filled: {nans_filled}/{nans_total}:   {np.round(nans_filled/nans_total*100, 2)}%')


# ### Let's create simple feature-based imputer for temperatures

# In[ ]:


# define simple imputer for longer missing sequences
import lightgbm as lgb
import os


def nan_imputer(data: pd.DataFrame, tcol: str, window: int = 24):
    
    df = data.copy()
    
    reg = lgb.LGBMRegressor(
        learning_rate=0.05,
        objective='mae',
        n_estimators=350,
        num_threads=os.cpu_count(),
        num_leaves=31,
        max_depth=8,
        subsample=0.8,
        min_child_samples=50,
        random_state=42,
    )

    init_cols = df.columns.tolist()

    dtime_col = 'timestamp'
    df['year'] = df[dtime_col].dt.year.astype(np.int16)
    df['hour'] = df[dtime_col].dt.hour.astype(np.uint8)
    df['month'] = df[dtime_col].dt.month.astype(np.uint8) - 1
    df['weekday'] = df[dtime_col].dt.weekday.astype(np.uint8)
    df['dayofyear'] = df[dtime_col].dt.dayofyear.astype(np.uint16) - 1
    df['weekofyear'] = df[dtime_col].dt.weekofyear.astype(np.uint8) - 1
    df['quarter'] = df[dtime_col].dt.quarter.astype(np.uint8) - 1
    df['monthday'] = df[dtime_col].dt.day.astype(np.uint8) - 1

    df['rolling_back'] = df.groupby(by='site_id')[tcol]        .rolling(window=window, min_periods=1).mean().interpolate().values

    # reversed rolling
    df['rolling_forw'] = df.iloc[::-1].groupby(by='site_id')[tcol]        .rolling(window=window, min_periods=1).mean().interpolate().values

    # rolling mean for same hour of the day
    df['rolling_back_h'] = df.groupby(by=['site_id', 'hour'])[tcol]        .rolling(window=3, min_periods=1).mean().interpolate().values

    df['rolling_back_h_f'] = df.iloc[::-1].groupby(by=['site_id', 'hour'])[tcol]        .rolling(window=3, min_periods=1).mean().interpolate().values
    
#     sampler = np.random.RandomState(42)
#     df['interpolation'] = df[tcol].interpolate() * (1 + sampler.randn(len(df)) * 0.25)
#     df.iloc[
#         np.random.choice(len(df), int(len(df)*3/5)),
#         df.columns.tolist().index('interpolation')
#     ] = np.nan

    tr_idx, val_idx = ~df[tcol].isnull(), df[tcol].isnull()

    features = [
        'site_id', 'hour', 'month', 'dayofyear', 'weekofyear', 'year',
        'rolling_back',
        'rolling_forw',
        'rolling_back_h',
        'rolling_back_h_f',
#         'interpolation'
    ]
    
    print(f'training model for col "{tcol}"...')
    reg.fit(
        X=df.loc[tr_idx, features], 
        y=df.loc[tr_idx, tcol],
        categorical_feature=['site_id', 'year'],
    )

    df[f'{tcol}_restored'] = np.nan
    df.loc[val_idx, f'{tcol}_restored'] = reg.predict(df.loc[val_idx, features])
    df.loc[val_idx, f'{tcol}'] = df.loc[val_idx, f'{tcol}_restored'].values
    
    # add simple rolling mean for comparison
    df[f'{tcol}_rolling_mean'] = df.groupby(by='site_id')[tcol]    .rolling(window=24*3, min_periods=1).mean().values
    
    # check from what features our imputer learned the most
    lgb.plot_importance(reg)
    
    return df.loc[:, init_cols + [f'{tcol}_restored', f'{tcol}_rolling_mean']]

tcols = ['air_temperature', 'dew_temperature']
restored = weather
for tcol in tcols:
    restored = nan_imputer(data=restored, tcol=tcol, window=24)


# In[ ]:


# check the sample of imputation results
tcol = 'air_temperature'

st =  (11, '2017-07-01 00:00:00')
end = (11, '2018-07-01 00:00:00')

restored.loc[st:end].set_index('timestamp')[[
    f'{tcol}', 
    f'{tcol}_restored', 
    f'{tcol}_rolling_mean'
]].iplot()


# At the end of the day, such (or similar) restoration techniques over ENTIRE possible time-grid allow us to **correctly** build some lag-based features, such as rolling means, shifts, etc.
# However, their importance for this particular competition remains unclear, we need to deep dive further
# 
# Hope you enjoyed this kernel and learned some hints & tricks!
# 
# **P.s.** Comments, likes, new ideas are highly welcomed!
# <br>Happy kaggling!
# 
# ---
# Check my latest notebooks:
# - [Aligning Temperature Timestamp](https://www.kaggle.com/frednavruzov/aligning-temperature-timestamp)
# - [Faster stratified cross-validation](https://www.kaggle.com/frednavruzov/faster-stratified-cross-validation-upd)
