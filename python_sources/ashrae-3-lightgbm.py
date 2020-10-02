#!/usr/bin/env python
# coding: utf-8

#  # **ASHRAE Energy Prediction**

# In[ ]:


# Import Statements

import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from lightgbm import LGBMRegressor, plot_importance
from sklearn.metrics import mean_squared_log_error as msle, mean_squared_error as mse
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
from subprocess import check_output
import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# # Any results you write to the current directory are saved as output.
pd.set_option('display.max_columns', 100)


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


print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


# Import data
INPUT = "../input/ashrae-energy-prediction/"
LEAKED_INPUT = "../input/ucb-data-leakage-site-4-81-buildings/"

df_train = pd.read_csv(f"{INPUT}train.csv")
df_test = pd.read_csv(f"{INPUT}test.csv")
bldg_metadata = pd.read_csv(f"{INPUT}building_metadata.csv")
weather_train = pd.read_csv(f"{INPUT}weather_train.csv")
weather_test = pd.read_csv(f"{INPUT}weather_test.csv")
sample = pd.read_csv(f"{INPUT}sample_submission.csv")
df_leaked = pd.read_csv(f"{LEAKED_INPUT}site4.csv")


# In[ ]:





# In[ ]:


df_test = df_test.drop(columns=['row_id'])
df_train = reduce_mem_usage(df=df_train)
df_test = reduce_mem_usage(df=df_test)
weather_train = reduce_mem_usage(df=weather_train)
weather_test = reduce_mem_usage(df=weather_test)
df_leaked = reduce_mem_usage(df=df_leaked)


# In[ ]:


df_train = df_train.merge(bldg_metadata, on='building_id', how='left')
df_test = df_test.merge(bldg_metadata, on='building_id', how='left')
df_train = df_train.merge(weather_train, on=['site_id', 'timestamp'], how='left')
df_test = df_test.merge(weather_test, on=['site_id', 'timestamp'], how='left')


# In[ ]:


df_leaked.head


# In[ ]:


sample.loc[df_test[df_test['site_id']==4].index, 'meter'] = df_leaked['meter_reading_scraped']


# In[ ]:


import gc
del weather_train, weather_test, bldg_metadata
gc.collect()


# In[ ]:


sample.to_csv('submission.csv')


# ## Data Cleaning

# References:
# * https://www.kaggle.com/caesarlupum/ashrae-start-here-a-gentle-introduction
# * https://www.kaggle.com/rohanrao/ashrae-half-and-half
