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
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import gc
import psutil     
import random
import datetime
import warnings
warnings.filterwarnings('ignore')
        
# Any results you write to the current directory are saved as output.

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer


# In[ ]:


building_metadata = pd.read_csv("../input/ashrae-energy-prediction/building_metadata.csv")
sample_submission = pd.read_csv("../input/ashrae-energy-prediction/sample_submission.csv")
test = pd.read_csv("../input/ashrae-energy-prediction/test.csv")
train = pd.read_csv("../input/ashrae-energy-prediction/train.csv")
weather_test = pd.read_csv("../input/ashrae-energy-prediction/weather_test.csv")
weather_train = pd.read_csv("../input/ashrae-energy-prediction/weather_train.csv")

train.timestamp = pd.to_datetime(train.timestamp)
test.timestamp = pd.to_datetime(test.timestamp)


# In[ ]:


## Function to reduce the DF size
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


def display_missing(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    
    f, ax = plt.subplots(figsize=(15, 6))
    plt.xticks(rotation='90')
    sns.barplot(x=missing_data.index, y=missing_data['Percent'])
    plt.xlabel('Features', fontsize=15)
    plt.ylabel('Percent', fontsize=15)
    plt.title('Percent of missing values by feature', fontsize=15)
    
    missing_data.head()
    return missing_data

def randomize_na(df):
    for col in df.columns:
        data = df[col]
        mask = data.isnull()
        samples = random.choices( data[~mask].values , k = mask.sum() )
        data[mask] = samples
    return df


## handling missing hours from
## https://www.kaggle.com/aitude/ashrae-missing-weather-data-handling
def populate_missing_hours(weather_df):
    time_format = "%Y-%m-%d %H:%M:%S"
    start_date = datetime.datetime.strptime(weather_df['timestamp'].min(),time_format)
    end_date = datetime.datetime.strptime(weather_df['timestamp'].max(),time_format)
    total_hours = int(((end_date - start_date).total_seconds() + 3600) / 3600)
    hours_list = [(end_date - datetime.timedelta(hours=x)).strftime(time_format) for x in range(total_hours)]

    missing_hours = []
    for site_id in range(16):
        site_hours = np.array(weather_df[weather_df['site_id'] == site_id]['timestamp'])
        new_rows = pd.DataFrame(np.setdiff1d(hours_list,site_hours),columns=['timestamp'])
        new_rows['site_id'] = site_id
        weather_df = pd.concat([weather_df,new_rows])

    weather_df = weather_df.reset_index(drop=True)    
    return weather_df

## refactored original in https://www.kaggle.com/aitude/ashrae-missing-weather-data-handling
def process_weather_data(df, update_columns, use_fillna):
    df = populate_missing_hours(df)
    df.timestamp = pd.to_datetime(df.timestamp)
    df['dayofweek'] = df.timestamp.dt.dayofweek
    df['month'] = df.timestamp.dt.month 
    df['week'] = df.timestamp.dt.week 
    df['day'] = df.timestamp.dt.day
    
    df = df.set_index(['site_id','day','month']) ## to speed things up

    for column, fillna in zip(update_columns, use_fillna):
        #print(f'column: {column}, fillna: {fillna}')
        updated = pd.DataFrame(df.groupby(['site_id','day','month'])[column].mean(), columns=[column])
        #if (fillna == True):
        #    updated = updated.fillna(method='ffill')
        df.update(updated, overwrite=False)
    
    df = df.reset_index()
    df.drop(['day','week','month'], axis=1, inplace=True)
    return df 
    


# In[ ]:


train = reduce_mem_usage(train)
test = reduce_mem_usage(test)
weather_train = reduce_mem_usage(weather_train)
weather_test = reduce_mem_usage(weather_test)
building_metadata = reduce_mem_usage(building_metadata)


# ## building_metadata

# In[ ]:


display_missing(building_metadata)


# In[ ]:


from sklearn import preprocessing

building_metadata = randomize_na(building_metadata) 
building_metadata['floor_count'] = building_metadata['floor_count'].astype(np.int16)
building_metadata['year_built'] = building_metadata['year_built'].astype(np.int16)
building_metadata['year_built'] = building_metadata['year_built'] - 1900

## remove outliers in SF
building_metadata['square_feet'] = building_metadata['square_feet'].apply(lambda x: x if x <= 600000 else 600000)
building_metadata['square_feet'] = building_metadata['square_feet'].apply(lambda x: np.log1p(x))

primary_use_encoder = preprocessing.LabelEncoder()
building_metadata['primary_use'] = primary_use_encoder.fit_transform(building_metadata.primary_use)

## remove outliers in floor count
building_metadata['floor_count'] = building_metadata['floor_count'].apply(lambda x: 20 if x > 20 else x)
gc.collect()


# ## weather_train

# In[ ]:


display_missing(weather_train)


# In[ ]:


update_lolumns = ['air_temperature','cloud_coverage', 'dew_temperature', 'sea_level_pressure', 'wind_direction', 'wind_speed', 'precip_depth_1_hr']   
use_fillna = [False, True, False, True, False, False, True]
    
weather_train_df = process_weather_data(weather_train, update_lolumns, use_fillna)
weather_test_df = process_weather_data(weather_test, update_lolumns, use_fillna)
gc.collect()


# ## reduce memory a little

# In[ ]:


to_drop = ['sea_level_pressure','wind_direction','wind_speed']

del weather_train_df[to_drop]
del weather_test_df[to_drop]
           
gc.collect()


# ## merge datasets

# In[ ]:


def merge_datasets(building, weather, data, is_test=False):
    df = data.merge(building, left_on='building_id',right_on='building_id',how='left')
    df = df.merge(weather, how='left', left_on=['site_id','timestamp'],right_on=['site_id','timestamp'])
    
    if is_test == True:
        row_id = df['row_id']
        df.drop(['row_id', 'timestamp'], axis=1, inplace=True)    
        return df, row_id

    ## this is train
    target = np.log1p(data['meter_reading'])  
    df.drop(['meter_reading','timestamp'], axis=1, inplace=True)    
    return df, target

train_df, target = merge_datasets(building_metadata, weather_train_df, train)
test_df, row_id = merge_datasets(building_metadata, weather_test_df, test, True)


# In[ ]:


del train, test
del building_metadata, weather_test, weather_train
del weather_train_df, weather_test_df
gc.collect()


# ## hyperparms from: https://www.kaggle.com/aitude/ashrae-hyperparameter-tuning

# In[ ]:


from sklearn.model_selection import KFold
import lightgbm as lgb

params = {
     'num_iterations':200,
     'boosting_type': 'gbdt',
     'objective': 'regression',
     'metric': 'rmse',
     'num_leaves' : 1000,
     'learning_rate': 0.07,
     'feature_fraction': 0.89,
     'bagging_fraction': 0.97,
     'lambda_l1' : 3,
     'lambda_l2' : 5,
     'max_depth' : 11    
}

categorical_features = ["building_id", "site_id", "meter", "primary_use", "dayofweek"]

evals_results = []  # to record eval results for plotting
models = []

kf = KFold(n_splits=3)
for train_index,test_index in kf.split(train_df):
    X_train, y_train = train_df.loc[train_index], target.loc[train_index]
    X_test, y_test = train_df.loc[test_index], target.loc[test_index]
    
    d_training = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features, free_raw_data=False)
    d_test = lgb.Dataset(X_test, label=y_test,categorical_feature=categorical_features, free_raw_data=False)
    
    evals_result = {}
    model = lgb.train(params, train_set=d_training, valid_sets=[d_training,d_test], 
                      verbose_eval=25, early_stopping_rounds=50, evals_result = evals_result)
    models.append(model)
    evals_results.append(evals_result)
    
    del X_train, y_train, X_test, y_test, d_training, d_test
    gc.collect()


# In[ ]:


for model, evals_result in zip(models, evals_results):
    f, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize=(15, 6))
    lgb.plot_importance(model, ax=ax1)
    lgb.plot_metric(evals_result, metric='rmse', ax=ax2)

plt.show()


# ## Prediction

# In[ ]:


del train_df, target, evals_results
gc.collect()

results = []
for model in models:
    if  results == []:
        results = np.expm1(model.predict(test_df, num_iteration=model.best_iteration)) / len(models)
    else:
        results += np.expm1(model.predict(test_df, num_iteration=model.best_iteration)) / len(models)
    del model
    gc.collect()


# ## Submission

# In[ ]:


del test_df, models
gc.collect()


# In[ ]:


submission = pd.DataFrame({"row_id": row_id, "meter_reading": np.clip(results, 0, a_max=None)})
del row_id,results
gc.collect()
submission.to_csv("submission.csv", index=False)


# In[ ]:


submission.head(10)


# In[ ]:


submission.describe()


# In[ ]:




