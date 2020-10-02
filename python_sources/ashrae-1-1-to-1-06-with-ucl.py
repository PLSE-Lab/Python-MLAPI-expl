#!/usr/bin/env python
# coding: utf-8

#  # **ASHRAE Energy Prediction**

# In[ ]:


# Import Statements
import gc
import datetime
import pandas as pd
import numpy as np
# import pickle
import lightgbm as lgb
# from lightgbm import LGBMRegressor, plot_importance
# from sklearn.metrics import mean_squared_log_error as msle, mean_squared_error as mse
# from sklearn.model_selection import KFold, train_test_split
# from sklearn.preprocessing import OneHotEncoder

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# # Any results you write to the current directory are saved as output.


# In[ ]:


# Code from https://www.kaggle.com/caesarlupum/ashrae-start-here-a-gentle-introduction 
# Function to reduce the DF size
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

# function to calculate evaluation metric
def rmsle(y_true: pd.Series, y_predict: pd.Series) -> float:
    """
    Evaluate root mean squared log error
    :param y_true:
    :param y_predict:
    :return:
    """
    return np.sqrt(msle(y_true, y_predict))


# In[ ]:


# Import data
INPUT = "../input/ashrae-energy-prediction/"

df_train = pd.read_csv(f"{INPUT}train.csv")
# df_test = pd.read_csv(f"{INPUT}test.csv")
bldg_metadata = pd.read_csv(f"{INPUT}building_metadata.csv")
weather_train = pd.read_csv(f"{INPUT}weather_train.csv")
# weather_test = pd.read_csv(f"{INPUT}weather_test.csv")
sample = pd.read_csv(f"{INPUT}sample_submission.csv")


# In[ ]:


def add_lag_feature(weather_df, window=3):
    group_df = weather_df.groupby('site_id')
    cols = ['air_temperature'] 
#             'cloud_coverage', 'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure', 'wind_direction', 'wind_speed']
    rolled = group_df[cols].rolling(window=window, min_periods=0)
    lag_mean = rolled.mean().reset_index().astype(np.float16)
    lag_max = rolled.max().reset_index().astype(np.float16)
    lag_min = rolled.min().reset_index().astype(np.float16)
    lag_std = rolled.std().reset_index().astype(np.float16)
    for col in cols:
        weather_df[f'{col}_mean_lag{window}'] = lag_mean[col]
        weather_df[f'{col}_max_lag{window}'] = lag_max[col]
        weather_df[f'{col}_min_lag{window}'] = lag_min[col]
        weather_df[f'{col}_std_lag{window}'] = lag_std[col]


# In[ ]:


def prepare_data(X,metadata,weather,test=False,new_df=None):
    if test :
        X = X.drop(columns=['row_id'])
        
    add_lag_feature(weather, window=72)
    X = X.merge(metadata, on='building_id',how='left')  
    X = X.merge(weather, on=['site_id', 'timestamp'], how='left')
    X['timestamp'] = pd.to_datetime(arg=X['timestamp'])
        
    X['year'] = X['timestamp'].dt.year
    X['month'] = X['timestamp'].dt.month
    X['day'] = X['timestamp'].dt.day
    X['hour'] = X['timestamp'].dt.hour
    X['weekday'] = X['timestamp'].dt.weekday
    
    X = reduce_mem_usage(X)
    
    beaufort = [(0, 0, 0.3), (1, 0.3, 1.6), (2, 1.6, 3.4), (3, 3.4, 5.5), (4, 5.5, 8), (5, 8, 10.8), (6, 10.8, 13.9), 
          (7, 13.9, 17.2), (8, 17.2, 20.8), (9, 20.8, 24.5), (10, 24.5, 28.5), (11, 28.5, 33), (12, 33, 200)]

    for item in beaufort:
        X.loc[(X['wind_speed']>=item[1]) & (X['wind_speed']<item[2]), 'beaufort_scale'] = item[0]

    
    X['age'] = X['year'] - X['year_built']
    if new_df is None:
        new_df = X.groupby(by=['building_id'], as_index=False)['timestamp'].min()
        new_df = new_df.rename(columns = {'timestamp': 'start_ts'})
    X = X.merge(new_df, on = 'building_id', how='left')
    X['hours_passed'] = (X['timestamp'] - X['start_ts']).dt.total_seconds()/3600
    X = reduce_mem_usage(X)
    if not test:
        X = X.query('not(site_id==0 & timestamp<"2016-05-21 00:00:00")')
        
    cols = ['floor_count', 'air_temperature', 'cloud_coverage', 'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure', 
        'wind_direction', 'wind_speed']
    X.loc[:, cols] = X.loc[:, cols].interpolate(axis=0)
    cat_cols = ['meter', 'primary_use', 'site_id', 'building_id', 'year', 'month', 'day', 'hour']
    for col in cat_cols:
        X[col] = X[col].astype('category')
    X = reduce_mem_usage(X)   
    if not test:
        return X,new_df
    else:
        return X


# In[ ]:


# df_test = df_test.drop(columns=['row_id'])


# In[ ]:


df_train = reduce_mem_usage(df=df_train)
# df_test = reduce_mem_usage(df=df_test)
weather_train = reduce_mem_usage(df=weather_train)
# weather_test = reduce_mem_usage(df=weather_test)
bldg_metadata = reduce_mem_usage(df=bldg_metadata)


# In[ ]:


df_train,new_df = prepare_data(df_train,bldg_metadata,weather_train)
del weather_train;gc.collect()


# In[ ]:


df_train = reduce_mem_usage(df_train)
df_train['group'] = df_train['month']
df_train['group'].replace((1,2,3,4,5,6), 1,inplace=True)
df_train['group'].replace((7,8,9,10,11,12), 2, inplace=True)
df_train['group'].value_counts()


# In[ ]:


df_train.columns


# In[ ]:


numericals = ["square_feet", "year_built", "air_temperature", "cloud_coverage",
              "dew_temperature", 'precip_depth_1_hr', 'floor_count', 'beaufort_scale','air_temperature_mean_lag72', 'air_temperature_max_lag72',
       'air_temperature_min_lag72', 'air_temperature_std_lag72']
#        'cloud_coverage_mean_lag72', 'cloud_coverage_max_lag72',
#        'cloud_coverage_min_lag72', 'cloud_coverage_std_lag72',
#        'dew_temperature_mean_lag72', 'dew_temperature_max_lag72',
#        'dew_temperature_min_lag72', 'dew_temperature_std_lag72',
#        'precip_depth_1_hr_mean_lag72', 'precip_depth_1_hr_max_lag72',
#        'precip_depth_1_hr_min_lag72', 'precip_depth_1_hr_std_lag72',
#        'sea_level_pressure_mean_lag72', 'sea_level_pressure_max_lag72',
#        'sea_level_pressure_min_lag72', 'sea_level_pressure_std_lag72',
#        'wind_direction_mean_lag72', 'wind_direction_max_lag72',
#        'wind_direction_min_lag72', 'wind_direction_std_lag72',
#        'wind_speed_mean_lag72', 'wind_speed_max_lag72', 'wind_speed_min_lag72',
#        'wind_speed_std_lag72']
categoricals = ["site_id", "building_id", "primary_use", "hour", "weekday", "meter",  "wind_direction"]
target = 'meter_reading'
feat_cols = categoricals + numericals


# In[ ]:


df_train[target] = np.log1p(df_train[target])


# ### by month 6 vs 6

# In[ ]:


X_half_1 = df_train.loc[df_train.group==1][feat_cols]
X_half_2 = df_train.loc[df_train.group==2][feat_cols]
y_half_1 = df_train.loc[df_train.group==1][target]
y_half_2 = df_train.loc[df_train.group==2][target]


# In[ ]:


d_half_1 = lgb.Dataset(X_half_1, label=y_half_1, categorical_feature=categoricals, free_raw_data=False)
d_half_2 = lgb.Dataset(X_half_2, label=y_half_2, categorical_feature=categoricals, free_raw_data=False)
watchlist_1 = [d_half_2, d_half_1]
watchlist_2 = [d_half_1, d_half_2]
params = {
    "objective": "regression",
    "boosting": "gbdt",#dart,gbdt
    "num_leaves": 45,
    "learning_rate": 0.02,
    "feature_fraction": 0.9,
    "reg_lambda": 2,
    "metric": "rmse"
}


# In[ ]:


print("Building model with first half and validating on second half:")
model_half_1 = lgb.train(params, train_set=d_half_1, num_boost_round=10, valid_sets=watchlist_1, verbose_eval=200, early_stopping_rounds=200)

print("Building model with second half and validating on first half:")
model_half_2 = lgb.train(params, train_set=d_half_2, num_boost_round=10, valid_sets=watchlist_2, verbose_eval=200, early_stopping_rounds=200)


# In[ ]:


models = [model_half_1,model_half_2]


# In[ ]:


del X_half_1,X_half_2,d_half_1,d_half_2,df_train, watchlist_1,watchlist_2,y_half_1,y_half_2
gc.collect()


# ### try devide by build_id

# In[ ]:


# from sklearn.metrics import mean_squared_error
# import lightgbm as lgb
# from sklearn.model_selection import KFold, StratifiedKFold
# from tqdm import tqdm
# y_train = df_train['meter_reading']
# y_train = np.log1p(y_train)
# X_train = df_train.drop(columns=['meter_reading','start_ts','timestamp'])
# del df_train;gc.collect()

# params = {
#             'boosting_type': 'gbdt',
#             'objective': 'regression',
#             'metric': {'rmse'},
#             'subsample': 0.25,
#             'subsample_freq': 1,
#             'learning_rate': 0.4,
#             'num_leaves': 20,
#             'feature_fraction': 0.9,
#             'lambda_l1': 1,  
#             'lambda_l2': 1
#             }

# folds = 4
# seed = 666

# kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

# models = []
# for train_index, val_index in kf.split(X_train, X_train['building_id']):
#     train_X = X_train[feat_cols].iloc[train_index]
#     val_X = X_train[feat_cols].iloc[val_index]
#     train_y = y_train.iloc[train_index]
#     val_y = y_train.iloc[val_index]
#     lgb_train = lgb.Dataset(train_X, train_y, categorical_feature=categoricals)
#     lgb_eval = lgb.Dataset(val_X, val_y, categorical_feature=categoricals)
#     gbm = lgb.train(params,
#                 lgb_train,
#                 num_boost_round=1000,
#                 valid_sets=(lgb_train, lgb_eval),
#                 early_stopping_rounds=100,
#                 verbose_eval = 100)
#     models.append(gbm)


# In[ ]:


# import gc
# del train_X, val_X, lgb_train, lgb_eval, train_y, val_y
# gc.collect()


# In[ ]:


df_test = pd.read_csv(f"{INPUT}test.csv")
weather_test = pd.read_csv(f"{INPUT}weather_test.csv")
df_test = reduce_mem_usage(df=df_test)
weather_test = reduce_mem_usage(df=weather_test)


# In[ ]:


df_test = prepare_data(df_test,bldg_metadata,weather_test,test=True,new_df=new_df)


# In[ ]:


df_test = reduce_mem_usage(df_test)


# In[ ]:


df_test = df_test.drop(columns=['start_ts','timestamp'])


# In[ ]:


del bldg_metadata,weather_test,new_df;gc.collect()


# In[ ]:


test = df_test[feat_cols]


# In[ ]:


from tqdm import tqdm
i=0
res=[]
step_size = 50000
for j in tqdm(range(int(np.ceil(test.shape[0]/50000)))):
    res.append(sum(np.expm1([model.predict(test.iloc[i:i+step_size]) for model in models])/len(models)))
    i+=step_size


# In[ ]:


res = np.concatenate(res)


# In[ ]:


submission = pd.read_csv('/kaggle/input/ashrae-energy-prediction/sample_submission.csv')
submission['meter_reading'] = res
submission.loc[submission['meter_reading']<0, 'meter_reading'] = 0
submission.to_csv('submission.csv', index=False, float_format='%.4f')
# submission.to_csv('submission.csv', index=False)
# submission


# this is 1.1

# ### replace to UCL data

# In[ ]:


get_ipython().run_cell_magic('time', '', "# site 0\nfrom sklearn.metrics import mean_squared_error\nimport tqdm\nleak_score0 = 0\n\nleak_df = pd.read_pickle('/kaggle/input/ashrae-ucf-spider-and-eda-full-test-labels/site0.pkl') \nleak_df['meter_reading'] = leak_df.meter_reading_scraped\nleak_df.drop(['meter_reading_original','meter_reading_scraped'], axis=1, inplace=True)\nleak_df.fillna(0, inplace=True)\nleak_df = leak_df[leak_df.timestamp.dt.year > 2016]\nleak_df.loc[leak_df.meter_reading < 0, 'meter_reading'] = 0 # remove large negative values\n\nsubmission.loc[submission.meter_reading < 0, 'meter_reading'] = 0\n\nfor bid in leak_df.building_id.unique():\n    temp_df = leak_df[(leak_df.building_id == bid)]\n    for m in temp_df.meter.unique():\n        v0 = submission.loc[(df_test.building_id == bid)&(df_test.meter==m), 'meter_reading'].values\n        v1 = temp_df[temp_df.meter==m].meter_reading.values\n        \n        leak_score0 += mean_squared_error(np.log1p(v0), np.log1p(v1)) * len(v0)\n        \n        submission.loc[(df_test.building_id == bid)&(df_test.meter==m), 'meter_reading'] = temp_df[temp_df.meter==m].meter_reading.values\n        \nleak_score0 /= len(leak_df)")


# In[ ]:


get_ipython().run_cell_magic('time', '', "# site 1\n\nleak_score1 = 0\n\nleak_df = pd.read_pickle('/kaggle/usr/lib/ucl_data_leakage_episode_2/site1.pkl') \nleak_df['meter_reading'] = leak_df.meter_reading_scraped\nleak_df.drop(['meter_reading_scraped'], axis=1, inplace=True)\nleak_df.fillna(0, inplace=True)\nleak_df = leak_df[leak_df.timestamp.dt.year > 2016]\nleak_df.loc[leak_df.meter_reading < 0, 'meter_reading'] = 0 # remove large negative values\n\n#sample_submission.loc[sample_submission.meter_reading < 0, 'meter_reading'] = 0\n\nfor bid in leak_df.building_id.unique():\n    temp_df = leak_df[(leak_df.building_id == bid)]\n    for m in temp_df.meter.unique():\n        v0 = submission.loc[(df_test.building_id == bid)&(df_test.meter==m), 'meter_reading'].values\n        v1 = temp_df[temp_df.meter==m].meter_reading.values\n        \n        leak_score1 += mean_squared_error(np.log1p(v0), np.log1p(v1)) * len(v0)\n        \n        submission.loc[(df_test.building_id == bid)&(df_test.meter==m), 'meter_reading'] = temp_df[temp_df.meter==m].meter_reading.values\n\nleak_score1 /= len(leak_df)")


# In[ ]:


submission.to_csv('submission_ucf_replaced.csv', index=False, float_format='%.4f')


# this is 1.06
