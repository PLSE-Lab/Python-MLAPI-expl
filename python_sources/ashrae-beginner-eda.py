#!/usr/bin/env python
# coding: utf-8

# # Ashrae - Beginner EDA
# 
# This is my first competition that I decided to do it right: Do some EDA and feature engineering before any model submission. 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')

import seaborn as sb


# ### Cleansing, merging and shrinking data

# In[ ]:


df_train = pd.read_csv('../input/ashrae-energy-prediction/train.csv')
df_test = pd.read_csv('../input/ashrae-energy-prediction/test.csv')
df_building = pd.read_csv('../input/ashrae-energy-prediction/building_metadata.csv')
df_w_train = pd.read_csv('../input/ashrae-energy-prediction/weather_train.csv')
df_w_test = pd.read_csv('../input/ashrae-energy-prediction/weather_test.csv')


# In[ ]:


def reduce_df_mem_usage(df, verbose=True):
    '''
    Function that reduces memory cosumption of pandas dataframe reducing number of bits for floats and numbers

    Parameters
    ----------
    df : pandas.data_frame 
        The dataframe to be shrinked
    verbose: boolean, optional
        if function should print the memory reduction
    
    Returns
    ----------
    df : pandas.data_frame 
        The shrinked dataframe
    '''

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

def generate_date_parts_features(df, timestamp_col='timestamp'):
    df['dt_timestamp'] = pd.to_datetime(df[timestamp_col], format='%Y-%m-%d %H:%M:%S')
    df['dt_weekday'] = df['dt_timestamp'].dt.weekday
    df['dt_quarter'] = df['dt_timestamp'].dt.quarter
    df['dt_year'] = df['dt_timestamp'].dt.year
    df['dt_month'] = df['dt_timestamp'].dt.month
    df['dt_day'] = df['dt_timestamp'].dt.day
    df['dt_dayofyear'] = df['dt_timestamp'].dt.dayofyear
    df['dt_weekofyear'] = df['dt_timestamp'].dt.weekofyear
    df['dt_hour'] = df['dt_timestamp'].dt.hour
    return df    


# In[ ]:


df_train = reduce_df_mem_usage(df_train)
df_test = reduce_df_mem_usage(df_test)
df_w_train = reduce_df_mem_usage(df_w_train)
df_w_test = reduce_df_mem_usage(df_w_test)


# In[ ]:


df_train = generate_date_parts_features(df_train)
df_test = generate_date_parts_features(df_test)


# In[ ]:


n_train = len(df_train)
n_test = len(df_test)
print(n_train, n_test)


# In[ ]:


df_train = pd.merge(df_train, df_building, left_on='building_id', right_on='building_id', how='left')
df_test = pd.merge(df_test, df_building, left_on='building_id', right_on='building_id', how='left')
df_train = pd.merge(df_train, df_w_train, left_on=['site_id', 'timestamp'], right_on=['site_id', 'timestamp'], how='left')
df_test = pd.merge(df_test, df_w_test, left_on=['site_id', 'timestamp'], right_on=['site_id', 'timestamp'], how='left')
df_train = reduce_df_mem_usage(df_train)
df_test = reduce_df_mem_usage(df_test)


# In[ ]:


df_train.head()


# In[ ]:


df_train.tail()


# In[ ]:


df_test.head()


# In[ ]:


df_test.tail()


# In[ ]:


df_train.dt_timestamp.describe()


# In[ ]:


df_test.dt_timestamp.describe()


# ### Basic EDA

# In[ ]:


pd.options.display.float_format = '{:,.3f}'.format
df_train.meter_reading.describe()


# In[ ]:


## looks like an error outlier. Let's check top 5 values
df_train.meter_reading.sort_values().tail(5)


# Let's investigate
# 

# In[ ]:


df_train.iloc[8905140]


# In[ ]:


df_train[df_train.building_id==1099].groupby(['building_id','meter'])['meter_reading'].agg(['max', 'min', 'mean', 'std'])


# Actualy depending on meter type values oscilate too much

# ### Investigate target variable

# In[ ]:


## Plot some buildings consumption
from IPython.display import display

random_buildings = df_building.sample(10)['building_id'].values
for b_id in random_buildings:
    sb.lineplot(x='dt_timestamp', y='meter_reading', data=df_train[df_train.building_id==b_id], hue='meter' ).set_title('Building_id = %s' % str(b_id))
    plt.show()


# It is clear that some meters are inversely proportional to others and that there is some interpolated data

# In[ ]:


## Some buildings have no data for some period. For instance Building_id == 1442. Let's investigate
sb.lineplot(x='dt_timestamp', y='meter_reading', data=df_train[df_train.building_id==1442], hue='meter' ).set_title('Building_id =1442')


# In[ ]:


df_building_1442 = df_train[((df_train.building_id==1442)&(df_train.dt_timestamp > '2016-02-10')&(df_train.dt_timestamp < '2016-03-30'))]
plt.figure(figsize=(16, 6))
sb.lineplot(x='dt_timestamp', y='meter_reading', data=df_building_1442[df_building_1442.meter==0]).set_title('Building_id =1442 between Feb and Mar')


# In[ ]:


## Target was clearly interpolated. Let's try to see why
df_building_1442[df_building_1442.meter==0].head(50)


# Actually seaborn interpolated the missing data. Some series has missing values. In 1442 Building example above, there's only values for a few days

# ### Build a simple model to check feature importance

# In[ ]:


from sklearn.ensemble import RandomForestRegressor

X = df_train.copy()

y = df_train['meter_reading']
X = X.drop(['meter_reading','timestamp', 'dt_timestamp'], axis=1)

X = X.fillna(-999)

# Label encoder for categorical features
for c in X.columns[X.dtypes == 'object']:
    X[c] = X[c].factorize()[0]
    
rf = RandomForestRegressor(max_depth=5)
rf.fit(X,y)
plt.plot(rf.feature_importances_)
plt.xticks(np.arange(X.shape[1]), X.columns.tolist(), rotation=90);


# * ### Investigate Nulls

# In[ ]:


## Number of NaN on each column
df_train.isnull().sum(axis=0)


# In[ ]:


df_test.isnull().sum(axis=0)


# In[ ]:


## check if there are buildings in test set not present on train data
train_buildings = df_train.building_id.unique()
test_buildings = df_test.building_id.unique()
print(set(train_buildings) == set(test_buildings))


# ### Some thoughts
# 
# * Investigate Holidays (which country?)
# * Probably we cannot treat each meter series independently, because they are correlated
# * Investigate site_id variable.
# * Several missing data in some dates for some buildings. Interpolate?
# * Several nan in weather sensors. Interpolate?
# * Generate aggregate metrics for weather sensors. What the best windows size? Hyperparameter?

# In[ ]:




