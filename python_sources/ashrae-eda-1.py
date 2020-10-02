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

# Any results you write to the current directory are saved as output.

# matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
import matplotlib.patches as patches

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from tqdm import tqdm


# In[ ]:


get_ipython().run_cell_magic('time', '', 'root = \'../input/ashrae-energy-prediction/\'\ntrain_df = pd.read_csv(root + \'train.csv\')\ntrain_df["timestamp"] = pd.to_datetime(train_df["timestamp"], format=\'%Y-%m-%d %H:%M:%S\')\n\nweather_train_df = pd.read_csv(root + \'weather_train.csv\')\nweather_train_df["timestamp"] = pd.to_datetime(weather_train_df["timestamp"], format=\'%Y-%m-%d %H:%M:%S\')\ntest_df = pd.read_csv(root + \'test.csv\')\ntest_df["timestamp"] = pd.to_datetime(test_df["timestamp"], format=\'%Y-%m-%d %H:%M:%S\')\nweather_test_df = pd.read_csv(root + \'weather_test.csv\')\nweather_test_df["timestamp"] = pd.to_datetime(weather_test_df["timestamp"], format=\'%Y-%m-%d %H:%M:%S\')\nbuilding_meta_df = pd.read_csv(root + \'building_metadata.csv\')\nsample_submission = pd.read_csv(root + \'sample_submission.csv\')')


# In[ ]:


print('Size of train_df data', train_df.shape)
print('Size of weather_train_df data', weather_train_df.shape)
print('Size of weather_test_df data', weather_test_df.shape)
print('Size of building_meta_df data', building_meta_df.shape)


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


# In[ ]:


## Reducing memory
train_df = reduce_mem_usage(train_df)
test_df = reduce_mem_usage(test_df)

weather_train_df = reduce_mem_usage(weather_train_df)
weather_test_df = reduce_mem_usage(weather_test_df)
building_meta_df = reduce_mem_usage(building_meta_df)


# In[ ]:


train_df.head()


# In[ ]:


train_df.info()


# In[ ]:


train_df.describe()


# In[ ]:


plt.figure(figsize = (16,5))
plt.plot(train_df['meter_reading'])


# In[ ]:


train_df.isnull().sum()


# In[ ]:


# No. of unique values in building id
train_df.building_id.nunique()


# In[ ]:


date_meter_reading = train_df.groupby(['timestamp'])['meter_reading'].sum()
plt.figure(figsize = (16,5))
plt.xlabel("timestamp")
plt.ylabel("meter_reading")
plt.plot(date_meter_reading)


# In[ ]:


date_mtype_reading = train_df.groupby(['timestamp', 'meter'])['meter_reading'].agg(['min','max','median','sum'])
date_mtype_reading.columns = ['meter_reading_' + x for x in date_mtype_reading.columns]
date_mtype_reading.reset_index(inplace=True)
date_mtype_reading['Date'] = date_mtype_reading.timestamp.dt.date
date_mtype_reading.head()


# In[ ]:


plt.figure(figsize = (16,5))
sns.lineplot(x="Date", 
             y="meter_reading_sum", 
             hue="meter", 
             data=date_mtype_reading, 
             palette=sns.color_palette('coolwarm', n_colors=4))


# {0: electricity, 1: chilledwater, 2: steam, 3: hotwater}
# What is the difference between steam and hotwater, in terms of usage in building and for meter reading?

# In[ ]:


plt.figure(figsize = (16,5))
sns.lineplot(x="Date", 
             y="meter_reading_max", 
             hue="meter", 
             data=date_mtype_reading, 
             palette=sns.color_palette('hls', n_colors=4))


# Most of the spikes are caused because of the maximum meter readings for each building. Lets try to plot median, which will give a clear picture

# In[ ]:


plt.figure(figsize = (16,5))
sns.lineplot(x="Date", 
             y="meter_reading_median", 
             hue="meter", 
             data=date_mtype_reading, 
             palette=sns.color_palette('husl', n_colors=4))


# Find what's the difference between values(variance or std) in meter_reading for a particular date. Although, median gives a clear picture, still want to confirm the numbers. OR plot the meter reading(scatter plot) for few dates and see the numbers. If the SD is less, we are good, else, this approach doesn't yield correct numbers.

# In[ ]:


building_meta_df.head()


# In[ ]:


building_meta_df.shape


# In[ ]:


building_meta_df.info()


# In[ ]:


building_meta_df.isnull().sum()


# In[ ]:


building_meta_df.describe()


# mean() of year_built is inf because sum is inf. Since, we have changed the dtype of columns to save memory usage, the sum() is coming out of range of float16.

# In[ ]:


# Types of building in the data
building_meta_df['primary_use'].unique()


# **UNIVARIATE ANALYSIS**

# In[ ]:


plt.figure(figsize = (16,5))
sns.distplot(building_meta_df.square_feet)


# In[ ]:


# Year built is the year building is opened, as per the kaggle information
building_meta_df_temp = building_meta_df[pd.notnull(building_meta_df.year_built)]
plt.figure(figsize = (16,5))
sns.distplot(building_meta_df_temp.year_built)


# In[ ]:


building_meta_df_temp = building_meta_df[pd.notnull(building_meta_df.floor_count)]
plt.figure(figsize = (16,5))
sns.distplot(building_meta_df_temp.floor_count)


# In[ ]:


plt.figure(figsize = (16,7))
use_plot = sns.countplot(building_meta_df_temp.primary_use)


# In[ ]:


# Check no. of buildings for each building type
building_meta_df_temp.primary_use.value_counts()


# In[ ]:


# Year built is the year building is opened, as per the kaggle information
building_meta_df_temp = building_meta_df[pd.notnull(building_meta_df.year_built)]
plt.figure(figsize = (16,5))
sns.distplot(building_meta_df_temp[building_meta_df_temp.primary_use == "Education"].year_built)


# In[ ]:


plt.figure(figsize = (16,5))
sns.distplot(building_meta_df_temp[building_meta_df_temp.primary_use == "Entertainment/public assembly"].year_built)


# In[ ]:


plt.figure(figsize = (16,5))
sns.distplot(building_meta_df_temp[building_meta_df_temp.primary_use == "Public services"].year_built)


# In[ ]:


plt.figure(figsize = (16,5))
sns.distplot(building_meta_df_temp[building_meta_df_temp.primary_use == "Office"].year_built)


# In[ ]:


plt.figure(figsize = (16,5))
sns.distplot(building_meta_df_temp[building_meta_df_temp.primary_use == "Lodging/residential"].year_built)


# The boom in the real estate industry(especially for resindential purposes) that came during 2000s, is shown by the graph.

# In[ ]:


plt.figure(figsize = (16,5))
sns.scatterplot(x="square_feet", y="floor_count", data=building_meta_df)


# Most of the buildings below 2,00,000 square feet have floor count within range of 10. Can be used to impute missing values.

# In[ ]:


building_meta_df.corr()


# No strong co-relations between any of the independent variables in building_meta dataframe.

# In[ ]:


building_type_df = building_meta_df.groupby(['primary_use'])['square_feet','year_built', 'floor_count'].agg(['mean','max','min'])
building_type_df.head()


# In[ ]:


building_type_df.columns = [building_type_df.columns[i][0] + '_' + building_type_df.columns[i][1] for i in range(0,len(building_type_df.columns))]


# In[ ]:


# Fill index of 'Food sales and service'
building_type_df.fillna(1, inplace=True)


# In[ ]:


for i in range(0,len(building_meta_df)):
    if(pd.isna(building_meta_df['year_built'][i])):
        building_meta_df.loc[i,'year_built'] = building_type_df.loc[building_meta_df['primary_use'][i], 'year_built_mean']
    if(pd.isna(building_meta_df['floor_count'][i])):
        building_meta_df.loc[i,'floor_count'] = building_type_df.loc[building_meta_df['primary_use'][i], 'floor_count_mean']


# In[ ]:


# Check null values
building_meta_df.isnull().sum()


# In[ ]:


weather_train_df.head()


# In[ ]:


weather_train_df.info()


# In[ ]:


weather_train_df.describe()


# In[ ]:


# Get unique count of site id's in the data
weather_train_df.site_id.unique()


# In[ ]:


weather_train_df.isnull().sum()


# In[ ]:


# To check occurence of missing values
plt.figure(figsize=(16,5))
sns.heatmap(weather_train_df.isnull(), cbar=False)


# In[ ]:


# To check occurence of missing values
plt.figure(figsize=(16,5))
sns.heatmap(weather_test_df.isnull(), cbar=False)


# In[ ]:


weather_train_df.corr()


# In[ ]:


# Get a date column from timestamp column
weather_train_date_df = weather_train_df.copy()
weather_train_date_df['Date'] = weather_train_df['timestamp'].dt.date


# In[ ]:


weather_train_df_temp = weather_train_df[pd.notnull(weather_train_df.air_temperature)]
plt.figure(figsize = (16,5))
sns.distplot(weather_train_df_temp.air_temperature)


# In[ ]:


plt.figure(figsize = (16,5))
sns.lineplot(x="Date", 
             y="air_temperature", 
             data=weather_train_date_df, 
             palette=sns.color_palette('husl', n_colors=4))


# This graph is in line with train_df data. Because, as the air temperature increased, meter reading for chilled water got increase, while as the temperature decrease, meter reading for hot water and specially steam increased.
# Winter Season - 2016-01 to 2016-05 & 2016-10 to 2017-01
# summer - 2016-05 to 2016-10
# 

# In[ ]:


weather_train_df_temp = weather_train_df[pd.notnull(weather_train_df.cloud_coverage)]
plt.figure(figsize = (16,5))
sns.distplot(weather_train_df_temp.cloud_coverage)


# In[ ]:


plt.figure(figsize = (16,5))
sns.lineplot(x="Date", 
             y="cloud_coverage", 
             data=weather_train_date_df, 
             palette=sns.color_palette('hls', n_colors=4))


# Cloud coverage is almost constant with spikes ranging from 2 to 3 except for a few cases. Means, it might be partly sunny always, given the weather conditions.
# FEW = Few (1 to 2 oktas); SCT = Scattered (3 to 4 oktas)

# In[ ]:


weather_train_df_temp = weather_train_df[pd.notnull(weather_train_df.dew_temperature)]
plt.figure(figsize = (16,5))
sns.distplot(weather_train_df_temp.dew_temperature)


# In[ ]:


plt.figure(figsize = (16,5))
sns.lineplot(x="Date", 
             y="dew_temperature", 
             data=weather_train_date_df, 
             palette=sns.color_palette('hls', n_colors=4))


# Information: The dewpoint temperature is the temperature at which the air can no longer "hold" all of the water vapor which is mixed with it, and some of the water vapor must condense into liquid water. The dew point is always lower than (or equal to) the air temperature.
# 
# If the air temperature cools to the dew point, or if the dew point rises to equal the air temperature, then dew, fog or clouds begin to form. At this point where the dew point temperature equals the air temperature, the relative humidity is 100%.
# 
# Take away: Can include relative humidity column from air temperature and dew temperature.

# In[ ]:


len(weather_train_df.precip_depth_1_hr.unique())


# In[ ]:


weather_train_df_temp = weather_train_df[pd.notnull(weather_train_df.precip_depth_1_hr)]
plt.figure(figsize = (16,5))
sns.distplot(weather_train_df_temp.precip_depth_1_hr)


# In[ ]:


plt.figure(figsize = (16,5))
sns.lineplot(x="Date", 
             y="precip_depth_1_hr", 
             data=weather_train_date_df, 
             palette=sns.color_palette('hls', n_colors=4))


# The spikes might indicate that rain fall happened. Can look into the picture by plotting values of relative humidity. Should show same behaviour.

# In[ ]:


weather_train_df_temp = weather_train_df[pd.notnull(weather_train_df.wind_direction)]
plt.figure(figsize = (16,5))
sns.distplot(weather_train_df_temp.wind_direction)


# In[ ]:


plt.figure(figsize = (16,5))
sns.lineplot(x="Date", 
             y="wind_direction", 
             data=weather_train_date_df, 
             palette=sns.color_palette('hls', n_colors=4))


# In[ ]:


weather_train_df_temp = weather_train_df[pd.notnull(weather_train_df.wind_speed)]
plt.figure(figsize = (16,5))
sns.distplot(weather_train_df_temp.wind_speed)


# In[ ]:


plt.figure(figsize = (16,5))
sns.lineplot(x="Date", 
             y="wind_speed", 
             data=weather_train_date_df, 
             palette=sns.color_palette('hls', n_colors=4))


# In[ ]:


# Fill the missing values using linear interpolation method in forward direction
weather_train_df.interpolate(method='linear', limit_direction='forward', inplace=True)


# In[ ]:


# Fill the missing values using linear interpolation method in backward direction to fill left-over values
weather_train_df.interpolate(method='linear', limit_direction='backward', inplace=True)


# In[ ]:


# Join the train and building dataframes
train_building_df = train_df.merge(building_meta_df, on=['building_id'], how='left')
train_building_df.shape


# In[ ]:


train_building_df.head()


# In[ ]:


train_building_df.info()


# In[ ]:


train_building_df.isnull().sum()


# In[ ]:


train_final_df = train_building_df.merge(weather_train_df, on=['timestamp', 'site_id'], how='left')
train_final_df.shape


# In[ ]:


train_final_df.head()


# In[ ]:


train_final_df.isnull().sum()


# Check for which timestamp and site id, values are not present in weather_data but present in training data

# In[ ]:


train_final_df.fillna(0, inplace=True)


# In[ ]:


train_final_df_temp = train_final_df.groupby('primary_use')['meter_reading'].agg(sum)
train_final_df_temp.sort_values(ascending=False)


# Since, no. of building which are high in count, have highest energy consumption, which shows a linear relationship with meter reading. And, also, since we are building a tree based model, label encoding is the best option for encoding.

# # Feature engineering
# 1) relative humidity = dewtemperature/airtemperature
# 2) building size = squarefee*floorcount

# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_final_df['primary_use'] = le.fit_transform(train_final_df['primary_use'])
train_final_df['primary_use'].value_counts()


# Can change the label encoding, as it changed the orientation, need to think about it.

# In[ ]:


train_final_df['month'] = train_final_df.timestamp.dt.month
train_final_df['hour'] = train_final_df.timestamp.dt.hour


# In[ ]:


# Take log of target variable to calculate rmse score for RMSLE
train_final_df['meter_reading'] = np.log(train_final_df['meter_reading'] + 1)


# In[ ]:


categoricals = ["site_id", "building_id", "primary_use", "meter",  "cloud_coverage"]
target = train_final_df.pop('meter_reading')


# In[ ]:


feat_cols = list(train_final_df.columns)
feat_cols.remove('timestamp')
feat_cols


# In[ ]:


from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold

params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': {'rmse'},
            'subsample': 0.25,
            'subsample_freq': 1,
            'learning_rate': 0.4,
            'num_leaves': 20,
            'feature_fraction': 0.9,
            'lambda_l1': 1,  
            'lambda_l2': 1
            }

folds = 4
seed = 666

kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

models = []

for train_index, val_index in kf.split(train_final_df, train_final_df['building_id']):
    train_X = train_final_df[feat_cols].iloc[train_index]
    val_X = train_final_df[feat_cols].iloc[val_index]
    train_y = target.iloc[train_index]
    val_y = target.iloc[val_index]
    lgb_train = lgb.Dataset(train_X, train_y, categorical_feature=categoricals)
    lgb_eval = lgb.Dataset(val_X, val_y, categorical_feature=categoricals)
    gbm = lgb.train(params,
                lgb_train,
                num_boost_round=500,
                valid_sets=(lgb_train, lgb_eval),
                early_stopping_rounds=100,
                verbose_eval = 100)
    models.append(gbm)


# In[ ]:


import pickle
model_dir = "/kaggle/working/models/"
if(os.path.exists(model_dir) == False):
    os.mkdir(model_dir)
else:
    i=1
    for model in models:
        pkl_filename = model_dir +  'model_'+ str(i) + '.pkl'
        print(pkl_filename)
        with open(pkl_filename, 'wb') as file:
            pickle.dump(model, file)
            i = i+1


# In[ ]:


for model in models:
    lgb.plot_importance(model)
    plt.show()


# In[ ]:


# Removing unnecessary dataframes
import gc
del train_final_df, train_X, val_X, lgb_train, lgb_eval, train_y, val_y, target
gc.collect()


# In[ ]:


# Join the train and building dataframes
test_building_df = test_df.merge(building_meta_df, on=['building_id'], how='left')
test_building_df.shape


# In[ ]:


test_building_df.isnull().sum()


# In[ ]:


# Fill the missing values using linear interpolation method in forward direction
weather_test_df.interpolate(method='linear', limit_direction='forward', inplace=True)
# Fill the missing values using linear interpolation method in backward direction
weather_test_df.interpolate(method='linear', limit_direction='forward', inplace=True)
test_final_df = test_building_df.merge(weather_test_df, on=['timestamp', 'site_id'], how='left')
test_final_df.shape


# In[ ]:


test_final_df.isnull().sum()


# Plot-> weather_train_df and weather_test_df after interpolation to see, if the graphs are changed or not. Should not change.
# Plot the distribution of all variables in both train and test. Get insights.
# Understand results properly, and do feature engineering. 
# What to do with outliers? Some buildings have very high meter value.

# In[ ]:


# Fill null values with 0
test_final_df.fillna(0, inplace=True)
test_final_df['primary_use'] = le.fit_transform(test_final_df['primary_use'])


# In[ ]:


test_final_df["timestamp"] = pd.to_datetime(test_final_df["timestamp"], format='%Y-%m-%d %H:%M:%S')
test_final_df['month'] = test_final_df.timestamp.dt.month
test_final_df['hour'] = test_final_df.timestamp.dt.hour
test_final_df = test_final_df[feat_cols]


# In[ ]:


# Predictions
# Summing up all the model predictions, divided by folds, and then taking exponent followed by subtraction from 1.
i=0
res=[]
step_size = 50000
for j in tqdm(range(int(np.ceil(test_final_df.shape[0]/50000)))):
    res.append(np.expm1(sum([model.predict(test_final_df.iloc[i:i+step_size]) for model in models])/folds))
    i+=step_size


# In[ ]:


res = np.concatenate(res)


# In[ ]:


submission = pd.read_csv('/kaggle/input/ashrae-energy-prediction/sample_submission.csv')
submission['meter_reading'] = res
submission.loc[submission['meter_reading']<0, 'meter_reading'] = 0
submission.to_csv('submission.csv', index=False)

