#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import gc
import seaborn as sns
import matplotlib.pyplot as plt

import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold


# In[ ]:


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# In[ ]:


def save_models(models):
    i=0
    for model in models:
        model.save_model(f'model_{i}.txt')
        i+=1

def load_models():
    models = []
    for i in range(3):
        model = lgb.Booster(model_file=f'../input/modelst2/model_{i}.txt')
        models.append(model)
    return models


# In[ ]:


# Hour of T max by site_id
htmax = [19,14,0,19,0,12,20,0,19,21,0,0,14,0,20,20]


# In[ ]:


holidays = ["2016-01-01", "2016-01-18", "2016-02-15", "2016-05-30", "2016-07-04",
            "2016-09-05", "2016-10-10", "2016-11-11", "2016-11-24", "2016-12-26",
            "2017-01-01", "2017-01-16", "2017-02-20", "2017-05-29", "2017-07-04",
            "2017-09-04", "2017-10-09", "2017-11-10", "2017-11-23", "2017-12-25",
            "2018-01-01", "2018-01-15", "2018-02-19", "2018-05-28", "2018-07-04",
            "2018-09-03", "2018-10-08", "2018-11-12", "2018-11-22", "2018-12-25",
            "2019-01-01"]


# In[ ]:


building_metadata = pd.read_csv("../input/ashrae-energy-prediction/building_metadata.csv")
building_metadata = reduce_mem_usage(building_metadata)

train = pd.read_csv("../input/ashrae-energy-prediction/train.csv",parse_dates=['timestamp'])
train = reduce_mem_usage(train)

weather_train = pd.read_csv("../input/ashrae-energy-prediction/weather_train.csv",parse_dates=['timestamp'])
weather_train = reduce_mem_usage(weather_train)

test = pd.read_csv("../input/ashrae-energy-prediction/test.csv",parse_dates=['timestamp'])
test = reduce_mem_usage(test)


# ### Data Cleaning

# In[ ]:


train = train.query('not (building_id <= 104 & meter == 0 & timestamp <= "2016-05-20")')


# ### Feature Engineering (train)

# In[ ]:


building_metadata["year_built_fe"] = building_metadata["year_built"].fillna(1960) - 1900
building_metadata["year_built_fe"] = building_metadata["year_built_fe"] // 10


# In[ ]:


building_metadata['square_feet'] =  np.log1p(building_metadata['square_feet'])


# In[ ]:


default_floor = pd.DataFrame( [2,1,1,1,8,2,1,1,1,1,1,1,1,4,2,1],
                              columns=["floor"],
                              index=["Education",
                                     "Entertainment/public assembly",
                                     "Food sales and service",
                                     "Healthcare",
                                     "Lodging/residential",
                                     "Manufacturing/industrial",
                                     "Office",
                                     "Other",
                                     "Parking",
                                     "Public services",
                                     "Religious worship",
                                     "Retail",
                                     "Services",
                                     "Technology/science",
                                     "Utility",
                                     "Warehouse/storage"])

building_metadata['floor_count'] = building_metadata.apply(lambda x: 
                                                           default_floor.loc[x['primary_use']].floor if np.isnan(x['floor_count']) else x['floor_count'],
                                                           axis = 1)


# In[ ]:


w = pd.concat([train,test])
pivot = w.pivot_table(values='meter_reading', index=['building_id'], columns=['meter'])

building_metadata["has_0_meter"] = pivot[:][0].apply(lambda x: np.isfinite(x))
building_metadata["has_1_meter"] = pivot[:][1].apply(lambda x: np.isfinite(x))
building_metadata["has_2_meter"] = pivot[:][2].apply(lambda x: np.isfinite(x))
building_metadata["has_3_meter"] = pivot[:][3].apply(lambda x: np.isfinite(x))

del w, pivot


# In[ ]:


le = LabelEncoder()
building_metadata["primary_use"] = le.fit_transform(building_metadata["primary_use"])


# In[ ]:


weather_train['air_temperature'] = weather_train['air_temperature'].interpolate(method ='linear', limit_direction ='both')
weather_train['dew_temperature'] = weather_train['dew_temperature'].interpolate(method ='linear', limit_direction ='both')


# In[ ]:


weather_train["Q_cumulated"] = weather_train.air_temperature.add(273).rolling(24*7,min_periods=1).sum()


# In[ ]:


t = train.merge(building_metadata, on='building_id', how='left')
t = t.merge(weather_train, on=['site_id', 'timestamp'], how='left')


# In[ ]:


t['week']    = t['timestamp'].dt.week
t['weekday'] = t['timestamp'].dt.weekday
t["hour"]    = t["timestamp"].dt.hour
t["htmax"]   = t.site_id.apply (lambda x: htmax[x])
t["w_htmax"] = t.hour.sub(t.htmax).abs()
t["w_htmax"] = t.w_htmax.apply(lambda x: (12 - x) if x<12 else (x%12))
t["is_holiday"] = (t.timestamp.dt.date.astype("str").isin(holidays)).astype(int)

del t["htmax"]


# ### LGBM fit

# In[ ]:


cat_features  = ['building_id', 'meter', 'site_id', 'primary_use', 
                 'week', 'weekday', 'hour', 'is_holiday',
                 'year_built_fe',
                 'has_0_meter', 'has_1_meter', 'has_2_meter', 'has_3_meter' ]

cont_features = ['square_feet', 'floor_count', 
                 'air_temperature', 'dew_temperature', 'Q_cumulated',
                 # -- 'cloud_coverage', 'precip_depth_1_hr', 
                 # -- 'sea_level_pressure', 'wind_direction', 'wind_speed', 
                 'w_htmax' ]


# In[ ]:


for c in cat_features:
    t[c] = t[c].astype('category')


# In[ ]:


x = t[cat_features + cont_features]
y = np.log1p(t['meter_reading'])


# In[ ]:


params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': 1280,
            'learning_rate': 0.07,
            'feature_fraction': 0.85,
            'reg_lambda': 2
          }


# In[ ]:


del t
gc.collect()


# In[ ]:


kf = KFold(n_splits=3)

models = []
evals_results = []

for train_index,test_index in kf.split(x):
    train_features = x.loc[train_index]
    train_target = y.loc[train_index]
    
    test_features = x.loc[test_index]
    test_target = y.loc[test_index]
    
    d_training = lgb.Dataset(train_features, label=train_target, categorical_feature=cat_features, free_raw_data=False)
    d_test     = lgb.Dataset(test_features,  label=test_target,  categorical_feature=cat_features, free_raw_data=False)
    
    evals_result = {}  # to record eval results for plotting

    model = lgb.train(params, 
                      train_set=d_training, 
                      num_boost_round=1000, 
                      valid_sets=[d_training,d_test], 
                      verbose_eval=25, 
                      early_stopping_rounds=50,
                      evals_result = evals_result)
    
    models.append(model)
    evals_results.append(evals_result)
    
    del train_features, train_target, test_features, test_target, d_training, d_test
    gc.collect()


# In[ ]:


for model, evals_result in zip(models, evals_results):
    f, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize=(15, 6))
    lgb.plot_importance(model, ax=ax1)
    lgb.plot_metric(evals_result, metric='rmse', ax=ax2)

plt.show()


# In[ ]:


save_models(models)


# ### Feature Engineering (test)

# In[ ]:


weather_test = pd.read_csv("../input/ashrae-energy-prediction/weather_test.csv",parse_dates=['timestamp'])
weather_test = reduce_mem_usage(weather_test)


# In[ ]:


weather_test['air_temperature'] = weather_test['air_temperature'].interpolate(method ='linear', limit_direction ='both')
weather_test['dew_temperature'] = weather_test['dew_temperature'].interpolate(method ='linear', limit_direction ='both')


# In[ ]:


weather_test["Q_cumulated"] = weather_test.air_temperature.add(273).rolling(24*7,min_periods=1).sum()


# In[ ]:


tt = test.merge(building_metadata, on='building_id', how='left')
tt = tt.merge(weather_test, on=['site_id', 'timestamp'], how='left')


# In[ ]:


tt['week']    = tt['timestamp'].dt.week
tt['weekday'] = tt['timestamp'].dt.weekday
tt["hour"]    = tt["timestamp"].dt.hour
tt["htmax"]   = tt.site_id.apply (lambda x: htmax[x])
tt["w_htmax"] = tt.hour.sub(tt.htmax).abs()
tt["w_htmax"] = tt.w_htmax.apply(lambda x: (12 - x) if x<12 else (x%12))
tt["is_holiday"] = (tt.timestamp.dt.date.astype("str").isin(holidays)).astype(int)


del tt["htmax"]


# In[ ]:


for c in cat_features:
    tt[c] = tt[c].astype('category')

x = tt[cat_features + cont_features]


# In[ ]:


del weather_train, weather_test, building_metadata
del y
del train, test, tt
gc.collect()


# ### LGBM predict

# In[ ]:


from tqdm import tqdm

step_size = 100000
res = []
i = 0
for j in tqdm(range(int(np.ceil(x.shape[0]/step_size)))):
    r = np.zeros(x.iloc[i:i+step_size].shape[0])
    for model in models:
        r += np.expm1(model.predict(x.iloc[i:i+step_size], num_iteration=model.best_iteration)) / len(models)
    res = np.append(res,r)
    i += step_size
    


# ### Submission

# In[ ]:


submission = pd.read_csv("../input/ashrae-energy-prediction/sample_submission.csv")
submission['meter_reading'] = res

check = submission.loc[submission['meter_reading']<0, 'meter_reading']
check.head()

submission.loc[submission['meter_reading']<0, 'meter_reading'] = 0


# In[ ]:


submission.to_csv('ASHRAE-LGBM-T2-16.csv', index=False)

