#!/usr/bin/env python
# coding: utf-8

# Full credit to cHa0s and anyone else for the half/half code template. For this kernal I am mostly interested in some basic feature engineering over the top of this. Unfortunately none of them improved things, but wanted to share some of the things that were tried. Let me know if you have any thoughts! 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import random
import gc

from sklearn.preprocessing import LabelEncoder

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
path_data = "/kaggle/input/ashrae-energy-prediction/"
path_train = path_data + "train.csv"
path_test = path_data + "test.csv"
path_building = path_data + "building_metadata.csv"
path_weather_train = path_data + "weather_train.csv"
path_weather_test = path_data + "weather_test.csv"

plt.style.use("seaborn")
sns.set(font_scale=1)

myfavouritenumber = 0
seed = myfavouritenumber
random.seed(seed)


# In[ ]:


## Memory optimization

# Original code from https://www.kaggle.com/gemartin/load-data-reduce-memory-usage by @gemartin
# Modified to support timestamp type, categorical type
# Modified to add option to use float16

from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_categorical_dtype

def reduce_mem_usage(df, use_float16=False):
    """
    Iterate through all the columns of a dataframe and modify the data type to reduce memory usage.        
    """
    
    start_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))
    
    for col in df.columns:
        if is_datetime(df[col]) or is_categorical_dtype(df[col]):
            continue
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype("category")

    end_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))
    
    return df


# In[ ]:


df_train = pd.read_csv(path_train)
df_train = reduce_mem_usage(df_train, use_float16=True)
df_train.head()


# In[ ]:


building = pd.read_csv(path_building)
building = reduce_mem_usage(building, use_float16=True)
building.head()


# In[ ]:


le = LabelEncoder()
building.primary_use = le.fit_transform(building.primary_use)
weather_train = pd.read_csv(path_weather_train)
weather_train = reduce_mem_usage(weather_train, use_float16=True)
building.head()


# In[ ]:



weather_train.head()


# In[ ]:


#weather forecast test:
#what is the expected max temperature for today?
# weather_train.timestamp = pd.to_datetime(weather_train.timestamp, format="%Y-%m-%d %H:%M:%S")
# weather_train['day'] = weather_train['timestamp'].dt.day
# weather_train["month"] = weather_train.timestamp.dt.month
# weather_forecast_df = weather_train.groupby(['site_id','day'])['air_temperature','cloud_coverage','dew_temperature','precip_depth_1_hr','sea_level_pressure','wind_direction','wind_speed'].max()


# In[ ]:


#weather_forecast_df.head()

Merge features with data
# In[ ]:


#education holidays, e.g. spring break. Some sites may not be in USA but lets see how this goes.
edu_holidays = ['2019-10-07','2018-10-08','2017-10-09','2016-10-10','2015-10-12',
'2019-10-08','2018-10-09','2017-10-10','2016-10-11','2015-10-13',
'2019-10-09','2018-10-10','2017-10-11','2016-10-12','2015-10-14',
'2019-10-10','2018-10-11','2017-10-12','2016-10-13','2015-10-15',
'2019-10-11','2018-10-12','2017-10-13','2016-10-14','2015-10-16',
'2019-10-12','2018-10-13','2017-10-14','2016-10-15','2015-10-17',
'2019-10-13','2018-10-14','2017-10-15','2016-10-16','2015-10-18',
'2019-10-14','2018-10-15','2017-10-16','2016-10-17','2015-10-19',
'2019-12-20','2018-12-21','2017-12-22','2016-12-23','2015-12-25',
'2019-12-21','2018-12-22','2017-12-23','2016-12-24','2015-12-26',
'2019-12-22','2018-12-23','2017-12-24','2016-12-25','2015-12-27',
'2019-12-23','2018-12-24','2017-12-25','2016-12-26','2015-12-28',
'2019-12-24','2018-12-25','2017-12-26','2016-12-27','2015-12-29',
'2019-12-25','2018-12-26','2017-12-27','2016-12-28','2015-12-30',
'2019-12-26','2018-12-27','2017-12-28','2016-12-29','2015-12-31',
'2019-12-27','2018-12-28','2017-12-29','2016-12-30','2016-01-01',
'2019-12-28','2018-12-29','2017-12-30','2016-12-31','2016-01-02',
'2019-12-29','2018-12-30','2017-12-31','2017-01-01','2016-01-03',
'2019-12-30','2018-12-31','2018-01-01','2017-01-02','2016-01-04',
'2019-12-31','2019-01-01','2018-01-02','2017-01-03','2016-01-05',
'2020-01-01','2019-01-02','2018-01-03','2017-01-04','2016-01-06',
'2020-02-17','2019-02-18','2018-02-19','2017-02-20','2016-02-22',
'2020-02-18','2019-02-19','2018-02-20','2017-02-21','2016-02-23',
'2020-02-19','2019-02-20','2018-02-21','2017-02-22','2016-02-24',
'2020-02-20','2019-02-21','2018-02-22','2017-02-23','2016-02-25',
'2020-04-20','2019-04-22','2018-04-23','2017-04-24','2016-04-25',
'2020-04-21','2019-04-23','2018-04-24','2017-04-25','2016-04-26',
'2020-04-22','2019-04-24','2018-04-25','2017-04-26','2016-04-27',
'2020-04-23','2019-04-25','2018-04-26','2017-04-27','2016-04-28'
]


# In[ ]:


holidays = ['2016-01-01', '2016-01-18', '2016-02-15', '2016-05-30', '2016-07-04',
                '2016-09-05', '2016-10-10', '2016-11-11', '2016-11-24', '2016-12-26',
                '2017-01-01', '2017-01-16', '2017-02-20', '2017-05-29', '2017-07-04',
                '2017-09-04', '2017-10-09', '2017-11-10', '2017-11-23', '2017-12-25',
                '2018-01-01', '2018-01-15', '2018-02-19', '2018-05-28', '2018-07-04',
                '2018-09-03', '2018-10-08', '2018-11-12', '2018-11-22', '2018-12-25',
                '2019-01-01']


# In[ ]:


edu_holidays_total = list(set(edu_holidays + holidays))


# In[ ]:


edu_holidays_total#


# In[ ]:


def prepare_data(X, building_data, weather_data, test=False):
    """
    Preparing final dataset with all features.
    """
    
    X = X.merge(building_data, on="building_id", how="left")
    X = X.merge(weather_data, on=["site_id", "timestamp"], how="left")
    
    X.timestamp = pd.to_datetime(X.timestamp, format="%Y-%m-%d %H:%M:%S")
    X.square_feet = np.log1p(X.square_feet)
    
    if not test:
        X.sort_values("timestamp", inplace=True)
        X.reset_index(drop=True, inplace=True)
    
    gc.collect()
    
#     holidays = ["2016-01-01", "2016-01-18", "2016-02-15", "2016-05-30", "2016-07-04",
#                 "2016-09-05", "2016-10-10", "2016-11-11", "2016-11-24", "2016-12-26",
#                 "2017-01-01", "2017-01-16", "2017-02-20", "2017-05-29", "2017-07-04",
#                 "2017-09-04", "2017-10-09", "2017-11-10", "2017-11-23", "2017-12-25",
#                 "2018-01-01", "2018-01-15", "2018-02-19", "2018-05-28", "2018-07-04",
#                 "2018-09-03", "2018-10-08", "2018-11-12", "2018-11-22", "2018-12-25",
#                 "2019-01-01"]
    

    
    X["hour"] = X.timestamp.dt.hour
    X["weekday"] = X.timestamp.dt.weekday
    X["is_holiday"] = np.where(X.primary_use == 0,(X.timestamp.dt.date.astype("str").isin(edu_holidays_total)).astype(int),(X.timestamp.dt.date.astype("str").isin(holidays)).astype(int))
    
    #Other features
    #edu_holidays_total = list(set(edu_holidays + holidays))
    #X["is_holiday_edu"] = (X.timestamp.dt.date.astype("str").isin(edu_holidays_total)).astype(int)
    
    
    X["month"] = X.timestamp.dt.month
    X['year'] = X.timestamp.dt.year
        #building age
    X['building_age'] = X['year'] - X['year_built']
    
    #square feet to number of floors
    X['square_feet_to_floor_count'] = np.where(X['floor_count']!= 0, X['square_feet'] / X['floor_count'], X['square_feet'])
    
    #Business hours
    X['is_wider_bus_hours'] = np.where((X["hour"]>=7)&(X["hour"]<=19),1,0)
#     X['is_core_bus_hours'] = np.where((X["hour"]>=9)&(X["hour"]<=17),1,0)
    #Weekend
    X['is_weekend'] = np.where((X["weekday"]>=1)&(X["weekday"]<=5),0,1)
    #Season of year
    X['season'] = (np.where(X["month"].isin([12,1,2]),1,
                   np.where(X["month"].isin([3,4,5]),2,         
                   np.where(X["month"].isin([6,7,8]),3,          
                   np.where(X["month"].isin([9,10,11]),4,0)))))
    #Business Quarter
#     X['business_quarter'] = (np.where(X["month"].isin([1,2,3]),1,
#                    np.where(X["month"].isin([4,5,6]),2,         
#                    np.where(X["month"].isin([7,8,9]),3,          
#                    np.where(X["month"].isin([10,11,12]),4,0)))))
    #Daylight Savings
    X['daylight_savings'] = (np.where(X["month"].isin([3,4,5,6,7,8,9,10]),1,0))
    #Lunch breaks
#     X['lunch_break'] = (np.where(X["hour"].isin([12]),1,0))  

    X['air_to_dew_temperature'] = X['air_temperature'] / X['dew_temperature']
    
    X = reduce_mem_usage(X, use_float16=True)
    
    drop_features = ["timestamp", "sea_level_pressure", "wind_direction", "wind_speed", 'year']

    X.drop(drop_features, axis=1, inplace=True)

    if test:
        row_ids = X.row_id
        X.drop("row_id", axis=1, inplace=True)
        return X, row_ids
    else:
        y = np.log1p(X.meter_reading)
        #X.drop("meter_reading", axis=1, inplace=True)
        return X, y


# In[ ]:


X_train, y_train = prepare_data(df_train, building, weather_train)
del df_train, weather_train

gc.collect()


# In[ ]:


X_train.head()


# In[ ]:


X_train.groupby('month')['building_id'].count()


# In[ ]:


#Target Encoding
df_group = X_train.groupby('building_id')['meter_reading']
building_mean = df_group.mean().astype(np.float16)
building_median = df_group.median().astype(np.float16)
building_min = df_group.min().astype(np.float16)
building_max = df_group.max().astype(np.float16)
building_std = df_group.std().astype(np.float16)


# In[ ]:


#Target Encoding Weather
weather_group = X_train.groupby('site_id')['air_temperature']
weather_mean = weather_group.mean().astype(np.float16)
weather_median = weather_group.median().astype(np.float16)
weather_min = weather_group.min().astype(np.float16)
weather_max = weather_group.max().astype(np.float16)
weather_std = weather_group.std().astype(np.float16)


# In[ ]:


building_mean.head()


# In[ ]:




def target_encoding(df):
    df['building_mean'] = df['building_id'].map(building_mean)
    df['building_median'] = df['building_id'].map(building_median)
    df['building_min'] = df['building_id'].map(building_min)
    df['building_max'] = df['building_id'].map(building_max)
    df['building_range'] = df['building_max'] - df['building_min']
    df['building_std'] = df['building_id'].map(building_std)
    #efficiency per sqft
    df['efficiency_per_sqft'] = df['square_feet'] / df['building_median']
    df['weather_mean'] = df['site_id'].map(weather_mean)
    df['weather_median'] = df['site_id'].map(weather_median)
    df['weather_min'] = df['site_id'].map(weather_min)
    df['weather_max'] = df['site_id'].map(weather_max)
    df['weather_range'] = df['weather_max'] - df['weather_min']
    df['weather_std'] = df['site_id'].map(weather_std)
    #outliers
    df['is_weather_outlier_2sd'] = (np.where(df['air_temperature'] > (df['weather_mean'] + 2 * df['weather_std']),1,
                                        np.where(df['air_temperature'] < (df['weather_mean'] - 2 * df['weather_std']),1,0)))
    df['is_weather_outlier_3sd'] = (np.where(df['air_temperature'] > (df['weather_mean'] + 3 * df['weather_std']),1,
                                        np.where(df['air_temperature'] < (df['weather_mean'] - 3 * df['weather_std']),1,0)))     
    df = reduce_mem_usage(df, use_float16=True)
    return df


# In[ ]:


X_train = target_encoding(X_train)


# In[ ]:


X_train.head()


# In[ ]:


X_train.groupby('is_weather_outlier_2sd')['building_id'].count()


# In[ ]:


building_std.head()


# In[ ]:





# In[ ]:


X_train = X_train.drop("meter_reading", axis=1)


# In[ ]:


X_train.head()


# In[ ]:


X_half_1 = X_train[:int(X_train.shape[0] / 2)]
X_half_2 = X_train[int(X_train.shape[0] / 2):]

y_half_1 = y_train[:int(X_train.shape[0] / 2)]
y_half_2 = y_train[int(X_train.shape[0] / 2):]

categorical_features = (["building_id", "site_id", "meter", "primary_use", "hour", "weekday",'is_wider_bus_hours',
                         'is_weekend','season','daylight_savings','month'])

d_half_1 = lgb.Dataset(X_half_1, label=y_half_1, categorical_feature=categorical_features, free_raw_data=False)
d_half_2 = lgb.Dataset(X_half_2, label=y_half_2, categorical_feature=categorical_features, free_raw_data=False)

del X_half_1, X_half_2, y_half_1, y_half_2

watchlist_1 = [d_half_1, d_half_2]
watchlist_2 = [d_half_2, d_half_1]

params = {
    "objective": "regression",
    "boosting": "gbdt",
    "num_leaves": 40,
    "learning_rate": 0.05,
    "feature_fraction": 0.85,
    "reg_lambda": 2,
    "metric": "rmse"
}

print("Building model with first half and validating on second half:")
model_half_1 = lgb.train(params, train_set=d_half_1, num_boost_round=1000, valid_sets=watchlist_1, verbose_eval=200, early_stopping_rounds=200)

print("Building model with second half and validating on first half:")
model_half_2 = lgb.train(params, train_set=d_half_2, num_boost_round=1000, valid_sets=watchlist_2, verbose_eval=200, early_stopping_rounds=200)


# ** Feature Importance **
# 
# Plotting the feature importance from LGBM.

# In[ ]:


df_fimp_1 = pd.DataFrame()
df_fimp_1["feature"] = X_train.columns.values
df_fimp_1["importance"] = model_half_1.feature_importance()
df_fimp_1["half"] = 1

df_fimp_2 = pd.DataFrame()
df_fimp_2["feature"] = X_train.columns.values
df_fimp_2["importance"] = model_half_2.feature_importance()
df_fimp_2["half"] = 2

df_fimp = pd.concat([df_fimp_1, df_fimp_2], axis=0)

plt.figure(figsize=(14, 7))
sns.barplot(x="importance", y="feature", data=df_fimp.sort_values(by="importance", ascending=False))
plt.title("LightGBM Feature Importance")
plt.tight_layout()


# Preparing test data 
# 
# Preparing test data with same features as train data.

# In[ ]:


del X_train, d_half_1, d_half_2
gc.collect()


# In[ ]:





# In[ ]:


# import gc
# import os
# import random

# import lightgbm as lgb
# import numpy as np
# import pandas as pd
# import seaborn as sns

# from matplotlib import pyplot as plt
# from sklearn.metrics import mean_squared_error
# from sklearn.preprocessing import LabelEncoder

df_test = pd.read_csv(path_test)
weather_test = pd.read_csv(path_weather_test)

df_test = reduce_mem_usage(df_test)
weather_test = reduce_mem_usage(weather_test)

X_test, row_ids = prepare_data(df_test, building, weather_test, test=True)

X_test = target_encoding(X_test)
X_test = reduce_mem_usage(X_test)


# In[ ]:


del df_test, building, weather_test
gc.collect()


# In[ ]:


del df_group,building_mean,building_median,building_min,building_max,building_std
gc.collect()


# In[ ]:



models = [model_half_1,model_half_2]

from tqdm import tqdm
i=0
pred=[]
step_size = 50000
for j in tqdm(range(int(np.ceil(X_test.shape[0]/50000)))):
    pred.append(np.expm1(sum([model.predict(X_test.iloc[i:i+step_size], num_iteration=model.best_iteration) for model in models])/2))
    i+=step_size



# In[ ]:




pred = np.concatenate(pred)


# In[ ]:


# pred = np.expm1(model_half_1.predict(X_test, num_iteration=model_half_1.best_iteration)) / 2

del model_half_1
gc.collect()


# In[ ]:


# pred += np.expm1(model_half_2.predict(X_test, num_iteration=model_half_2.best_iteration)) / 2
    
del model_half_2
gc.collect()


# In[ ]:


submission = pd.DataFrame({"row_id": row_ids, "meter_reading": np.clip(pred, 0, a_max=None)})
submission.to_csv("submission.csv", index=False)
print("DONE")


# In[ ]:


submission.to_pickle('GEPIII_V2_FE_20191117.pkl')

