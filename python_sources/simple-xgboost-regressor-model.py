#!/usr/bin/env python
# coding: utf-8

# ## <p style="color:MediumSeaGreen;">XGBOOST Regressor</p>
# 
# 
# 
# 
# <p style="color:rgb(60, 60, 60);">In this kernel XGB Regressor is implemented.<br>
# Due to memory reason rather than appending model i exported them with pickle and imported to predict.<br>
# 
# 
# 

# In[ ]:


import gc
import os
import random

import lightgbm as lgb
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from sklearn.model_selection import GroupKFold

import xgboost as xgb
from xgboost import plot_importance, plot_tree
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import pickle

path_data = "/kaggle/input/ashrae-energy-prediction/"
path_train = path_data + "train.csv"
path_test = path_data + "test.csv"
path_building = path_data + "building_metadata.csv"
path_weather_train = path_data + "weather_train.csv"
path_weather_test = path_data + "weather_test.csv"


seed = 2019
random.seed(seed)
plt.style.use('fivethirtyeight')


# ## <p style="color:MediumSeaGreen;">Memory reducer function</p>
# 

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


# ## <p style="color:MediumSeaGreen;">Prepare Data :}</p>
# There are two files with features that need to be merged with the data. One is building metadata that has information on the buildings and the other is weather data that has information on the weather.   
# 
# Note that the only features created are hour, weekday and is_holiday!<br>
# 
# FE code ref:-https://www.kaggle.com/rohanrao/ashrae-half-and-half<br></p>

# In[ ]:


def prepare_data(X, building_data, weather_data, test=False):
    """
    Preparing final dataset with all features.
    """
    
    X = X.merge(building_data, on="building_id", how="left")
    X = X.merge(weather_data, on=["site_id", "timestamp"], how="left")
    
    #X.sort_values("timestamp")
    #X.reset_index(drop=True)
    
    gc.collect()
    
    holidays = ["2016-01-01", "2016-01-18", "2016-02-15", "2016-05-30", "2016-07-04",
                "2016-09-05", "2016-10-10", "2016-11-11", "2016-11-24", "2016-12-26",
                "2017-01-02", "2017-01-16", "2017-02-20", "2017-05-29", "2017-07-04",
                "2017-09-04", "2017-10-09", "2017-11-10", "2017-11-23", "2017-12-25",
                "2018-01-01", "2018-01-15", "2018-02-19", "2018-05-28", "2018-07-04",
                "2018-09-03", "2018-10-08", "2018-11-12", "2018-11-22", "2018-12-25",
                "2019-01-01"]
    
    X.timestamp = pd.to_datetime(X.timestamp, format="%Y-%m-%d %H:%M:%S")
    X.square_feet = np.log1p(X.square_feet)
    
    X["hour"] = X.timestamp.dt.hour
    X["month"]=X.timestamp.dt.month
    X["weekday"] = X.timestamp.dt.weekday
    X["is_holiday"] = (X.timestamp.isin(holidays)).astype(int)
    
    drop_features = ["timestamp", "sea_level_pressure", "wind_direction", "wind_speed"]

    X.drop(drop_features, axis=1, inplace=True)

    if test:
        row_ids = X.row_id
        X.drop("row_id", axis=1, inplace=True)
        return X, row_ids
    else:
        y = np.log1p(X.meter_reading)
        X.drop("meter_reading", axis=1, inplace=True)
        return X, y


# In[ ]:


#TRAIN MAKER

def TRAINMAKER():
    #get csv
    df_train = pd.read_csv(path_train)
    building = pd.read_csv(path_building)
    #labelencode it
    le = LabelEncoder()
    building.primary_use = le.fit_transform(building.primary_use)
    weather_train = pd.read_csv(path_weather_train)
    #reduce memory
    df_train = reduce_mem_usage(df_train, use_float16=True)
    building = reduce_mem_usage(building, use_float16=True)
    weather_train = reduce_mem_usage(weather_train, use_float16=True)
    #make train set
    X_train, y_train = prepare_data(df_train, building, weather_train)
    
    print('helo')
    del df_train, weather_train,building
    gc.collect()
    
    return X_train,y_train


# In[ ]:


#TEST MAKER
def TESTMAKER():
    df_test = pd.read_csv(path_test)
    df_test = reduce_mem_usage(df_test)

    weather_test = pd.read_csv(path_weather_test)
    weather_test = reduce_mem_usage(weather_test)
    
    building = pd.read_csv(path_building)
    building = reduce_mem_usage(building, use_float16=True)
    
    le = LabelEncoder()
    building.primary_use = le.fit_transform(building.primary_use)
    
    X_test, row_ids = prepare_data(df_test, building, weather_test, test=True)
    del df_test, building, weather_test
    gc.collect()
    return X_test,row_ids


# In[ ]:


X_train,y_train=TRAINMAKER()


# In[ ]:


X_train.shape


# In[ ]:


rows_to_drop = pd.read_csv("../input/ashrae-simple-data-cleanup-lb-1-08-no-leaks-v3/rows_to_drop.csv")

bad_rows = rows_to_drop.values.tolist() # This is a list of lists
# Flatten the bad_rows list
flattened_bad_rows = [val for sublist in bad_rows for val in sublist]
X_train.drop(flattened_bad_rows, axis=0, inplace=True)

# Do the same to the target variable
y_train = y_train.reindex_like(X_train)

X_train.shape, y_train.shape


# ## <p style="color:MediumSeaGreen;">Training XGB With GroupKFold (Month) :</p>

# In[ ]:


cols=list(X_train.columns)
models = []
skf = GroupKFold(n_splits=6)
a=0


# In[ ]:


for i, (idxT, idxV) in enumerate( skf.split(X_train, y_train, groups=X_train['month']) ):
    month = X_train.iloc[idxV]['month'].iloc[0]
    print('Fold',i,'withholding month',month)
    print(' rows of train =',len(idxT),'rows of holdout =',len(idxV))
    oof = []
    reg =  xgb.XGBRegressor(
                  n_estimators=6000,
                  max_depth=12,
                  num_boost_round=500,
                  learning_rate=0.03,
                  subsample=0.8,
                  colsample_bytree=0.4,
                  missing=np.nan,
                  objective ='reg:squarederror',
                  tree_method='gpu_hist'
                  )
    h = reg.fit(X_train[cols].iloc[idxT], y_train.iloc[idxT], 
            eval_set=[(X_train[cols].iloc[idxV],y_train.iloc[idxV])],
            verbose=1000, early_stopping_rounds=500)

    oof = reg.predict(X_train[cols].iloc[idxV])
    #preds += reg.predict_proba(X_test[cols])[:,1]/skf.n_splits
    print('#'*20)
    print ('OOF CV=',mean_squared_error(y_train.iloc[idxV],oof))
    print('#'*20)
   # models.append(reg)
    pickle.dump(reg, open("thunder{}.pickle.dat".format(a), "wb"))
    a=a+1
    del h, reg, oof
    x=gc.collect()
   


# In[ ]:


del X_train,y_train
gc.collect()


# In[ ]:


X_test,row_ids=TESTMAKER()


# In[ ]:


i=0
a=0
res=[]
models=[]
folds=6
step_size = 50000
for a in range(0,6):
    models.append(pickle.load(open("thunder{}.pickle.dat".format(a), "rb")))


# ## <p style="color:MediumSeaGreen;">Feature Importance</p>

# In[ ]:


for a in models:
    plot_importance(a)


# ## <p style="color:MediumSeaGreen;font_size:100px">Predicting Chunks</p>
# prediction code ref:-https://www.kaggle.com/corochann/ashrae-training-lgbm-by-meter-type<br>

# In[ ]:


for j in tqdm(range(int(np.ceil(X_test.shape[0]/50000)))):
    res.append(np.expm1(sum([model.predict(X_test.iloc[i:i+step_size]) for model in models])/folds))
    i+=step_size


# In[ ]:


res = np.concatenate(res)


# In[ ]:


submission = pd.read_csv('/kaggle/input/ashrae-energy-prediction/sample_submission.csv')
submission['meter_reading'] = res
submission.loc[submission['meter_reading']<0, 'meter_reading'] = 0
submission.to_csv('submission.csv', index=False)
submission.sample(10)


# <p style="color:orange;">UPVOTE if you like<p> 
