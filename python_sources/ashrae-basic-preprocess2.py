#!/usr/bin/env python
# coding: utf-8

# ASHRAE - GREAT ENERGY PREDICTOR III 
# 
# Sources: 
# - https://www.kaggle.com/kyakovlev/ashrae-data-minification
# - https://www.kaggle.com/hmendonca/starter-eda-and-feature-selection-ashrae3
# - https://www.kaggle.com/chmaxx/ashrae-eda-and-visualization-wip
# 

# In[ ]:


# General imports
import numpy as np
import pandas as pd
import os, warnings, gc, math

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

warnings.filterwarnings('ignore')
NROWS = None


# In[ ]:


# Memory reducer function
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


get_ipython().run_cell_magic('time', '', "# Data load\ntrain_df = pd.read_csv('../input/ashrae-energy-prediction/train.csv', nrows = NROWS)\ntest_df = pd.read_csv('../input/ashrae-energy-prediction/test.csv', nrows = NROWS)\n\nbuilding_df = pd.read_csv('../input/ashrae-energy-prediction/building_metadata.csv', nrows = NROWS)\n\ntrain_weather_df = pd.read_csv('../input/ashrae-energy-prediction/weather_train.csv', nrows = NROWS)\ntest_weather_df = pd.read_csv('../input/ashrae-energy-prediction/weather_test.csv', nrows = NROWS)")


# ## Date transformations

# In[ ]:


get_ipython().run_cell_magic('time', '', "# Date convertions\nfor df in [train_df, test_df, train_weather_df, test_weather_df]:\n    \n    df['timestamp'] = pd.to_datetime(df['timestamp'])\n    \n# for df in [train_df, test_df, train_weather_df, test_weather_df]:\nfor df in [train_df, test_df]:\n    df['hour'] = np.uint8(df['timestamp'].dt.hour)\n    df['day'] = np.uint8(df['timestamp'].dt.day)\n    df['weekday'] = np.uint8(df['timestamp'].dt.weekday)\n    df['month'] = np.uint8(df['timestamp'].dt.month)\n    df['year'] = np.uint8(df['timestamp'].dt.year-2000)\n    \n# Categorical convertions")


# ## Building tranformations

# In[ ]:


get_ipython().run_cell_magic('time', '', '# Fill NA\nbuilding_df.rename(columns={"square_feet": "log_square_feet"}, inplace=True)\nbuilding_df[\'log_square_feet\'] = np.float16(np.log(building_df[\'log_square_feet\']))\nbuilding_df[\'year_built\'] = np.uint8(building_df[\'year_built\']-1900)\nbuilding_df[\'floor_count\'] = np.uint8(building_df[\'floor_count\'])\n\n# Enconding\nfrom sklearn.preprocessing import LabelEncoder\nle = LabelEncoder()\nbuilding_df[\'primary_use\'] = building_df[\'primary_use\'].astype(str)\nbuilding_df[\'primary_use\'] = le.fit_transform(building_df[\'primary_use\']).astype(np.int8)')


# ## Weather NA Imputation

# In[ ]:


get_ipython().run_cell_magic('time', '', "from sklearn.experimental import enable_iterative_imputer\nfrom sklearn.impute import IterativeImputer\n\nfor df in [train_weather_df, test_weather_df]:\n    cols = list(df.columns)\n    cols.remove('timestamp')\n    imp = IterativeImputer(random_state=42)\n    temp = imp.fit_transform(df[cols])\n    df[cols] = pd.DataFrame(temp, columns = cols)")


# ## Memory optimization

# In[ ]:


get_ipython().run_cell_magic('time', '', 'for df in [train_df, test_df, building_df, train_weather_df, test_weather_df]:\n    original = df.copy()\n    df = reduce_mem_usage(df)')


# ## Merging building into train dataset
# Temp_df combined with merge it's a faster and smarter whey to do this operation

# In[ ]:


get_ipython().run_cell_magic('time', '', "temp_df = train_df[['building_id']]\ntemp_df = temp_df.merge(building_df, on=['building_id'], how='left')\ndel temp_df['building_id']\ntrain_df = pd.concat([train_df, temp_df], axis=1)\ndel temp_df")


# ## Merging building into test dataset

# In[ ]:


get_ipython().run_cell_magic('time', '', "temp_df = test_df[['building_id']]\ntemp_df = temp_df.merge(building_df, on=['building_id'], how='left')\ndel temp_df['building_id']\ntest_df = pd.concat([test_df, temp_df], axis=1)\ndel temp_df")


# ## Merging weather into train dataset

# In[ ]:


get_ipython().run_cell_magic('time', '', "temp_df = train_df[['site_id','timestamp']]\ntemp_df = temp_df.merge(train_weather_df, on=['site_id','timestamp'], how='left')\ndel temp_df['site_id'], temp_df['timestamp']\ntrain_df = pd.concat([train_df, temp_df], axis=1)\ndel temp_df")


# ## Merging weather into test dataset

# In[ ]:


get_ipython().run_cell_magic('time', '', "temp_df = test_df[['site_id','timestamp']]\ntemp_df = temp_df.merge(test_weather_df, on=['site_id','timestamp'], how='left')\ndel temp_df['site_id'], temp_df['timestamp']\ntest_df = pd.concat([test_df, temp_df], axis=1)\n")


# ## Cleaning Memory

# In[ ]:


get_ipython().run_cell_magic('time', '', 'del train_weather_df, test_weather_df, temp_df\ngc.collect()')


# In[ ]:


get_ipython().run_cell_magic('time', '', "for m in train_df.meter.unique():\n#     train_df[train_df.meter == m].to_parquet('train'+ str(m) + '.parquet')\n#     test_df[test_df.meter == m].to_parquet('test'+ str(m) + '.parquet')\n    train_df[train_df.meter == m].to_pickle('train'+ str(m) + '.pkl')\n    test_df[test_df.meter == m].to_pickle('test'+ str(m) + '.pkl')")


# In[ ]:




