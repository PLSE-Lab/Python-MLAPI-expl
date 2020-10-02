#!/usr/bin/env python
# coding: utf-8

# This kernel shows how to use [Shapley values](https://www.google.com/search?q=Shapley+values) for feature importance scoring. This is not a new thing and has been used in several competitions in the past e.g. [a old kernel.](https://www.kaggle.com/hmendonca/lightgbm-predictions-explained-with-shap-0-796). Shapley values are not the default metric from LGBM but can easily be implemented with the [SHAP library](https://github.com/slundberg/shap), which is also very useful for model explainability.   
# Once you start experimenting and adding many features to your model, these techniques became crucial to select only the best features and avoid over-fitting.

# ### ASHRAE - Great Energy Predictor III
# 
# I'll summarise below the insights from my brief data analysis and whatever is shared in public discussions.
# This is still highly under progress and will be updated when possible.
# 
# Quoting Chris Balbach (our Competition Host):
# > Consider the scenario laid out in this diagram. This competition simulates the modelling challenge presented at the end of the timeline when measured energy and weather conditions are known and the adjusted baseline energy must be calculated. 
# ![Energy Consumption/Demand vs Time](https://www.mdpi.com/make/make-01-00056/article_deploy/html/images/make-01-00056-g001-550.jpg)
# > For further details, I recommend reading [this paper](https://www.mdpi.com/2504-4990/1/3/56) by Clayton Miller.
# 
# We are actually using data from 2016 to predict the demand for both 2017 and 2018, which is a hard task. However, other public kernels have showed that you can give a relatively good estimate even with a very simple linear model.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os, gc
import random
import datetime

from tqdm import tqdm_notebook as tqdm

# matplotlib and seaborn for plotting
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# import plotly.offline as py
# py.init_notebook_mode(connected=True)
# from plotly.offline import init_notebook_mode, iplot
# init_notebook_mode(connected=True)
# import plotly.graph_objs as go
# import plotly.offline as offline
# offline.init_notebook_mode()

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, mean_squared_log_error

import lightgbm as lgb
import shap


# In[ ]:


path = '../input/ashrae-energy-prediction'
# Input data files are available in the "../input/" directory.
for dirname, _, filenames in os.walk(path):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Load data

# In[ ]:


get_ipython().run_cell_magic('time', '', '# unimportant features (see importance below)\n# unimportant_cols = [\'wind_direction\', \'wind_speed\', \'sea_level_pressure\'] ## orig\nunimportant_cols = [\'wind_direction\']\ntarget = \'meter_reading\'\n\ndef load_data(source=\'train\', path=path):\n    \'\'\' load and merge all tables \'\'\'\n    assert source in [\'train\', \'test\']\n    \n    building = pd.read_csv(f\'{path}/building_metadata.csv\', dtype={\'building_id\':np.uint16, \'site_id\':np.uint8})\n    weather  = pd.read_csv(f\'{path}/weather_{source}.csv\', parse_dates=[\'timestamp\'],\n                                                           dtype={\'site_id\':np.uint8, \'air_temperature\':np.float16,\n                                                                  \'cloud_coverage\':np.float16, \'dew_temperature\':np.float16,\n                                                                  \'precip_depth_1_hr\':np.float16},\n                                                           usecols=lambda c: c not in unimportant_cols)\n    df = pd.read_csv(f\'{path}/{source}.csv\', dtype={\'building_id\':np.uint16, \'meter\':np.uint8}, parse_dates=[\'timestamp\'])\n    df = df.merge(building, on=\'building_id\', how=\'left\')\n    df = df.merge(weather, on=[\'site_id\', \'timestamp\'], how=\'left\')\n    print("done merging")\n    ## joint key of meter X building\n    df["key"] = df["building_id"].astype(str)+"."+df["meter"].astype(str)\n    \n    \n    # building age - (lacks handling of missing values..)\n    \n    df[\'building_age\'] = np.uint8(df[\'timestamp\'].dt.year - df[\'year_built\'])\n    \n    # expand datetime components\n    df[\'hour\'] = np.uint8(df[\'timestamp\'].dt.hour)\n    df[\'day\'] = np.uint8(df[\'timestamp\'].dt.day)\n    df[\'weekday\'] = np.uint8(df[\'timestamp\'].dt.weekday)\n    df[\'month\'] = np.uint8(df[\'timestamp\'].dt.month)\n#         df[\'year\'] = np.uint8(df[\'timestamp\'].dt.year-2000)\n\n    print("done datetime components")\n    # parse and cast columns to a smaller type\n    df.rename(columns={"square_feet": "log_square_feet"}, inplace=True)\n    df[\'log_square_feet\'] = np.float16(np.log(df[\'log_square_feet\'])).round(3)\n#         df[\'year_built\'] = np.uint8(df[\'year_built\']-1900)\n    df[\'floor_count\'] = np.uint8(df[\'floor_count\'])\n\n\n    if source ==\'train\':\n        df[\'meter_reading\'] = np.log1p(df[\'meter_reading\']).round(5).astype(np.float32) # comp metric uses log errors\n        # comp uses 4 decimal places only \n            \n    return df\n\n# load and display some samples\ntrain = load_data(\'train\')\ntrain.sample(5)')


# In[ ]:


train.tail().meter_reading


# In[ ]:


train.sample(12345).nunique()


# ### Test data

# In[ ]:


test = load_data('test')
test.sample(5)


# # EDA and sample statistics

# In[ ]:


print(f'Training from {train.timestamp.min()} to {train.timestamp.max()}, and predicting from {test.timestamp.min()} to {test.timestamp.max()}')


# ### Building types on each site location

# In[ ]:


building_counts = train.groupby(['primary_use', 'site_id']).building_id.nunique().to_frame('counts')
building_counts = building_counts.reset_index().pivot(index='primary_use', columns='site_id', values='counts')

fig, ax = plt.subplots(figsize=(16,8))
_ = sns.heatmap(building_counts, annot=True, cmap='Reds',
                xticklabels=building_counts.columns.values,
                yticklabels=building_counts.index.values)


# In[ ]:


# target's log-log histogram:
ax = np.log1p(train.meter_reading).hist()
ax.set_yscale('log')

# describe raw values first
train.meter_reading.describe()


# In[ ]:


# check the distribution in the types of meters
meters = train.groupby('building_id').meter.nunique().to_frame()
ax = meters.hist()
_ = ax[0][0].set_title('Distribution of types of meters\n{0:electricity, 1:water, 2:steam, 3:hotwater}') # from the official starter kernel
# from the graphs it looks like steam and hotwater are reversed (e.g.: 3:steam, 2:hotwater) but that shouldn't make any difference to the model


# ### display a single time series (notice measurement errors and discontinuities)

# In[ ]:


building_id = 1258  # a building with all 4 meters:  meters[meters.meter == 4]
meters = train[train['building_id'] == building_id].meter.nunique()

for meter in range(meters):
    fig, ax = plt.subplots()
    plt.title(f'Building {building_id} Meter {meter}')
    ax2 = ax.twinx()
    # plot meter_reading
    idx = (train['building_id'] == building_id) & (train['meter'] == meter)
    dates = matplotlib.dates.date2num(train.loc[idx, 'timestamp'])
    ax2.plot_date(dates, train.loc[idx, 'meter_reading'], '-', label='meter_reading', alpha=0.8)
    # plot air_temperature
    dates = matplotlib.dates.date2num(train.loc[train['building_id'] == building_id, 'timestamp'])
    ax.plot_date(dates, train.loc[train['building_id'] == building_id, 'air_temperature'], '.', color='tab:cyan', label='air_temperature')
    ax.set_ylabel('air_temperature'); ax2.set_ylabel('meter_reading')
    ax.legend(loc='upper left'); ax2.legend(loc='upper right')


# ### now let's see what's the expected prediction in the test set for the same building

# In[ ]:


meter = 1 # pick a meter

train_sample = train[(train['building_id'] == building_id) & (train['meter'] == meter)]  # same train sample as above

test['meter_reading'] = 0.0
test_sample = test[(test['building_id'] == building_id) & (test['meter'] == meter)]  # and the same meter in the test set

fig, ax = plt.subplots(figsize=(16,4))
plt.title(f'Building {building_id} Meter {meter}')
ax.xaxis.set_tick_params(rotation=30, labelsize=10)
ax2 = ax.twinx()

# plot training sample
dates = matplotlib.dates.date2num(train_sample['timestamp'])
ax2.plot_date(dates, train_sample['meter_reading'], '-', label='train', alpha=0.8)
ax.plot_date(dates, train_sample['air_temperature'], '.', color='tab:cyan', label='air_temperature')

# plot test sample
dates = matplotlib.dates.date2num(test_sample['timestamp'])
ax2.plot_date(dates, test_sample['meter_reading'], '*', label='test', alpha=0.8)
ax.plot_date(dates, test_sample['air_temperature'], '.', color='tab:cyan', label='air_temperature')

ax.set_ylabel('air_temperature'); ax2.set_ylabel('meter_reading')
ax.legend(loc='upper left'); ax2.legend(loc='upper right')

del train_sample; del test_sample; del dates


# In[ ]:


# some feature stats
train.describe()


# In[ ]:


# the counts above expose the missing data (Should we drop or refill the missing data?)
print("Ratio of available data (not NAN's):")
data_ratios = train.count()/len(train)
data_ratios


# In[ ]:


# Is the same happening in the test set? Yes
print("Ratio of available data (not NAN's):")
test.count()/len(test)


# In[ ]:


# # we can refill with averages
# train.loc[:, data_ratios < 1.0].mean()


# # Preprocess data

# In[ ]:


# class ASHRAE3Preprocessor(object):
#     @classmethod
#     def fit(cls, df, data_ratios=data_ratios):
#         cls.avgs = df.loc[:,data_ratios < 1.0].mean()
#         cls.pu_le = LabelEncoder()
#         cls.pu_le.fit(df["primary_use"])

#     @classmethod
#     def transform(cls, df):
# #         df = df.fillna(cls.avgs) # refill NAN with averages
# #         df['primary_use'] = np.uint8(cls.pu_le.transform(df['primary_use']))  # encode labels

#         # expand datetime into its components
#         df['hour'] = np.uint8(df['timestamp'].dt.hour)
#         df['day'] = np.uint8(df['timestamp'].dt.day)
#         df['weekday'] = np.uint8(df['timestamp'].dt.weekday)
#         df['month'] = np.uint8(df['timestamp'].dt.month)
# #         df['year'] = np.uint8(df['timestamp'].dt.year-2000)
        
#         # parse and cast columns to a smaller type
#         df.rename(columns={"square_feet": "log_square_feet"}, inplace=True)
#         df['log_square_feet'] = np.float16(np.log(df['log_square_feet']))
# #         df['year_built'] = np.uint8(df['year_built']-1900)
#         df['floor_count'] = np.uint8(df['floor_count'])
        
#         # remove redundant columns
#         for col in df.columns:
#             if col in ['timestamp', 'row_id']:
#                 del df[col]
    
#         # extract target column
#         if 'meter_reading' in df.columns:
#             df['meter_reading'] = np.log1p(df['meter_reading'])#.astype(np.float32) # comp metric uses log errors

#         return df
        
# ASHRAE3Preprocessor.fit(train)


# In[ ]:


# train = ASHRAE3Preprocessor.transform(train)
# train.sample(7)


# In[ ]:


train.to_csv('train_v1.csv.gz', index=False,compression="gzip")
test.to_csv('test_v1.csv.gz', index=False,compression="gzip")
submission.head(9)

