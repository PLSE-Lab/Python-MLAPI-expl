#!/usr/bin/env python
# coding: utf-8

# ### ASHRAE - Great Energy Predictor III

# In[ ]:


import numpy as np # linear algebra
import pandas as pd 
import os, gc
import random
import datetime

from tqdm import tqdm_notebook as tqdm #progress tool bar

# matplotlib and seaborn for plotting
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, mean_squared_log_error
import lightgbm as lgb
import shap


# In[ ]:


for dirname,blank,filenames in os.walk('../input/ashrae-energy-prediction'):
    print(dirname,blank,filenames)
    for file in filenames:
        print(os.path.join(dirname,file))


#  **Loading data**

# In[ ]:


get_ipython().run_cell_magic('time', '', "path='../input/ashrae-energy-prediction'\nunimportant_cols = []\ntarget = 'meter_reading'\n#function to load data\ndef load_data(source='train', path=path):\n    assert source in ['train', 'test']\n    df_building = pd.read_csv(f'{path}/building_metadata.csv', \n                              dtype={'building_id':np.uint16, 'site_id':np.uint8})\n    df_weather  = pd.read_csv(f'{path}/weather_{source}.csv', parse_dates=['timestamp'],\n                                                           dtype={'site_id':np.uint8, 'air_temperature':np.float16,\n                                                                  'cloud_coverage':np.float16, 'dew_temperature':np.float16,\n                                                                  'precip_depth_1_hr':np.float16},\n                                                           usecols=lambda c: c not in unimportant_cols)\n    df = pd.read_csv(f'{path}/{source}.csv', \n                     dtype={'building_id':np.uint16, 'meter':np.uint8}, \n                     parse_dates=['timestamp'])\n\n    return df_building,df_weather,df")


# In[ ]:


## a very simple Function to reduce the DF size 
def reduce_mem_usage(df, verbose=True):
    numerics = ['int8','int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':# comparing string
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


get_ipython().run_cell_magic('time', '', "# load and display some samples\ndf_building,df_weather,df_train = load_data('train')\ndf_building_train=reduce_mem_usage(df_building)\ndf_weather_train=reduce_mem_usage(df_weather)\ndf_train=reduce_mem_usage(df_train)\ngc.collect()")


# ### Test data

# In[ ]:


get_ipython().run_cell_magic('time', '', "# load and display some samples\ndf_building_test,df_weather_test,df_test = load_data('test')\ndf_building_test=reduce_mem_usage(df_building_test)\ndf_weather_test=reduce_mem_usage(df_weather_test)\ndf_test=reduce_mem_usage(df_test)")


# ### Aligning timestamp 
# #### Ref: https://www.kaggle.com/frednavruzov/aligning-temperature-timestamp<br>
# ###### Align timestamps
# Timestap data is not in their local time. As energy consumptions are related to the local time, an alighment is nescessary before using timestamp. 
# 
# The credit goes to [this kernel](https://www.kaggle.com/nz0722/aligned-timestamp-lgbm-by-meter-type) for the idea. Refer it for more details and explanation about below code.

# Aligning timestamp process:
# 1. concating the weather data

# In[ ]:


weather = pd.concat([df_weather_train,df_weather_test],ignore_index=True)


# 1. The hottest time of the day is around 2 p.m. Heat continues building up after noon, when the sun is highest in the sky, as long as more heat is arriving at the earth than leaving. By 2 p.m. or so, the sun is low enough in the sky for outgoing heat to be greater than incoming.

# In[ ]:


weather_key = ['site_id', 'timestamp']
#small data requirement for timestamp alignment (alginment is on the basis of air temprature which is highest at 3:00PM or 15:00 hrs)
temp_skeleton = weather[weather_key + ['air_temperature']].drop_duplicates(subset=weather_key).sort_values(by=weather_key).copy()


# ranking the temprature of particular date w.r.t air temprature for each site_id

# In[ ]:


# calculate ranks of hourly temperatures within date/site_id chunks (extra feature is added on temprory dataset)
temp_skeleton['temp_rank'] = temp_skeleton.groupby(['site_id', temp_skeleton.timestamp.dt.date])['air_temperature'].rank('average')
#calculate avg ranking of temprature including other searches


# In[ ]:


# create a dataframe of site_ids (0-16) x mean hour rank of temperature within day (0-23)(columns)
df_2d = temp_skeleton.groupby(['site_id', temp_skeleton.timestamp.dt.hour])['temp_rank'].mean().unstack(level=1)
# Subtract the columnID of temperature peak by 14, getting the timestamp alignment gap.
site_ids_offsets = pd.Series(df_2d.values.argmax(axis=1) - 14)
site_ids_offsets.index.name = 'site_id'


# In[ ]:


#aligning timestamp using above result
def timestamp_align(df):
    df['offset'] = df.site_id.map(site_ids_offsets)
    df['timestamp_aligned'] = (df.timestamp - pd.to_timedelta(df.offset, unit='H'))
    df['timestamp'] = df['timestamp_aligned']
    del df['timestamp_aligned']
    del df['offset']
    return df


# In[ ]:


df_weather_train_aligned=timestamp_align(df_weather_train)
df_weather_test_aligned=timestamp_align(df_weather_test)


# In[ ]:


def merging_data(df,df_building,df_weather):    
    df = df.merge(df_building, on='building_id', how='left')
    df = df.merge(df_weather, on=['site_id', 'timestamp'], how='left')
    del df_building
    del df_weather
    return df


# In[ ]:


df_train_aligned=merging_data(df_train,df_building,df_weather_train_aligned)
df_test_aligned=merging_data(df_test,df_building_test,df_weather_test_aligned)
print(f'shape of traindata before alignment: {df_train.shape} and shape of test data before alignment: {df_test.shape}')
print(f'shape of traindata after alignment: {df_train_aligned.shape} and shape of test data after alignment: {df_test_aligned.shape}')


# In[ ]:


#removing unwanted columns
del df_test_aligned['row_id']


# In[ ]:


df_train_aligned=reduce_mem_usage(df_train_aligned)
df_test_aligned=reduce_mem_usage(df_test_aligned)


# In[ ]:


print(f'memory used in merged train data {df_train_aligned.info(verbose=False)} and memory test data:{df_test_aligned.info(verbose=False)} ')


# #### Removing weired data on site_id==0
# there is already so much discussion on this issue so there is no need to explain 
# https://www.kaggle.com/corochann/ashrae-training-lgbm-by-meter-type

# In[ ]:


def plot_date_usage(train_df,site_id,meter,building_id):
    train_temp_df=train_df[train_df['site_id']==site_id]
    train_temp_df=train_temp_df[train_df['meter']==meter]
    train_temp_df = train_temp_df[train_temp_df['building_id'] == building_id]   
    train_temp_df['date']=train_temp_df['timestamp'].dt.date
    train_temp_df_meter = train_temp_df.groupby('date')['meter_reading'].sum()
    train_temp_df_meter = train_temp_df_meter.to_frame().reset_index()
    plt.plot(train_temp_df_meter['date'],train_temp_df_meter['meter_reading'])
    plt.xlabel('date')
    plt.ylabel('meter_reading_transform')
    plt.show()


# In[ ]:


plot_date_usage(df_train_aligned,0,0,0)


# In[ ]:


#removing weirder data from site_id=0; All electricity meter is 0 until May 20 for site_id == 0 and meter=0
#building 0 to 104 lies on site_id 0 only
df_train_aligned=df_train_aligned.query('not(building_id <= 104 & meter == 0 & timestamp <= "2016-05-20")')


# In[ ]:


print(f'shape of traindata before alignment: {df_train.shape} and shape of test data before alignment: {df_test.shape}')
print(f'shape of traindata after alignment: {df_train_aligned.shape} and shape of test data after alignment: {df_test_aligned.shape}')


# ### feature extraction
# 
# * Hour of day.
# * Business hours or not.(not applicable due to different type of building)
# * Weekend or not.
# * Season of the year.
# * Public holiday or not.+weekend

# In[ ]:


#PREPROCESSING TIMESTAMP IN TRAIN AND TEST
import holidays
def timestamp_preprocess(df):
    df['date']=df['timestamp'].dt.date
    df['hour']=df['timestamp'].dt.hour#hour of day
    df['day']=df['timestamp'].dt.day
    df['weekday']=df['timestamp'].dt.weekday
    df['month']=df['timestamp'].dt.month
    import holidays
    us_holidays =holidays.US()
    df['holiday']=df['date'].apply(lambda x: us_holidays.get(x))
    df['holiday']=df['holiday'].apply(lambda x:0 if x==None else 1)
    df['holiday'][df.weekday == 6]=1#sun
    df['holiday'][df.weekday == 5]=1 #sat  
    df['square_feet']=np.float16(np.log(df['square_feet']))#normalising floorspace
    del df['floor_count']
    del df['year_built']
    del df['cloud_coverage']
    del df['weekday']
    return df


# In[ ]:


get_ipython().run_cell_magic('time', '', "#preprocessing train data\ndf_train_preprocess=reduce_mem_usage(timestamp_preprocess(df_train_aligned))\ndf_train_preprocess['meter_reading_transform'] = np.log1p(df_train_preprocess['meter_reading']).astype(np.float32)")


# In[ ]:


get_ipython().run_cell_magic('time', '', 'gc.collect()\n#preprocessing test data\ndf_test_preprocess=reduce_mem_usage(timestamp_preprocess(df_test_aligned))')


# In[ ]:


#removing redundant columns
gc.collect()
del df_train_preprocess['meter_reading']


# In[ ]:


import matplotlib.pyplot as plt
for feature in ['air_temperature','dew_temperature','wind_speed','precip_depth_1_hr']:
    sns.distplot(df_train_preprocess[feature], hist=False)
    plt.show(sns)


# #### filling missing value

# In[ ]:


# Import label encoder 
from sklearn import preprocessing 
# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 
# Encode labels in column 'species'. 
df_train_preprocess['primary_use']= label_encoder.fit_transform(df_train_preprocess['primary_use'])
df_test_preprocess['primary_use']=label_encoder.fit_transform(df_test_preprocess['primary_use'])


# In[ ]:


df_train_preprocess=reduce_mem_usage(df_train_preprocess)
df_test_preprocess=reduce_mem_usage(df_test_preprocess)


# In[ ]:


#removing unwanted columns
def remove_redundant_cols(df):
    unwanted_columns=['wind_direction','wind_speed','sea_level_pressure']
    for col in unwanted_columns:
        del df[col]
    return df  


# In[ ]:


get_ipython().run_cell_magic('time', '', "df_test_preprocess=remove_redundant_cols(df_test_preprocess)\ndf_train_preprocess=remove_redundant_cols(df_train_preprocess)\nprint(f'memory usage df_train_preprocess:{df_train_preprocess.info(verbose=False)} and memory usage of  df_test_preprocess: {df_test_preprocess.info(verbose=False)}')\ngc.collect()")


# In[ ]:


list_missing_columns=['air_temperature','dew_temperature','precip_depth_1_hr']
def fill_na(df):
    for value in list_missing_columns:
        df[value] = df[value].fillna(df.groupby('primary_use')[value].transform('mean'))
    return df


# In[ ]:


df_train_preprocess=fill_na(df_train_preprocess)
df_test_preprocess=fill_na(df_test_preprocess)
df_train_preprocess=df_train_preprocess.reset_index(drop=True)


# In[ ]:


df_train_preprocess.tail()


# ### timeseries feature (mean,median,lag,deviation)

# In[ ]:


#generating rolling mean,std deviation, max,min,actual_value
def rolling_feature(df):
    df['air_temperature_mean'] = df['air_temperature'].rolling(window=7,center=False).mean()
    df['dew_temperature_mean'] = df['dew_temperature'].rolling(window=7,center=False).mean()
    df['precip_depth_1_hr_mean'] = df['precip_depth_1_hr'].rolling(window=7,center=False).mean()
    df['air_temperature_std'] = df['air_temperature'].rolling(window=7,center=False).std()
    df['dew_temperature_std'] = df['dew_temperature'].rolling(window=7,center=False).std()
    df['precip_depth_1_hr_std'] = df['precip_depth_1_hr'].rolling(window=7,center=False).std()
    df['air_temperature_max'] = df['air_temperature'].rolling(window=7,center=False).max()
    df['dew_temperature_max'] = df['dew_temperature'].rolling(window=7,center=False).max()
    df['precip_depth_1_hr_max'] = df['precip_depth_1_hr'].rolling(window=7,center=False).max()
    df['air_temperature_min'] = df['air_temperature'].rolling(window=7,center=False).min()
    df['dew_temperature_min'] = df['dew_temperature'].rolling(window=7,center=False).min()
    df['precip_depth_1_hr_min'] = df['precip_depth_1_hr'].rolling(window=7,center=False).min()
    df["air_temperature_mean"].fillna( method ='bfill', inplace = True) 
    df["dew_temperature_mean"].fillna( method ='bfill', inplace = True) 
    df["precip_depth_1_hr_mean"].fillna( method ='bfill', inplace = True) 
    df["air_temperature_std"].fillna( method ='bfill', inplace = True) 
    df["dew_temperature_std"].fillna( method ='bfill', inplace = True) 
    df["precip_depth_1_hr_std"].fillna( method ='bfill', inplace = True)
    df["air_temperature_min"].fillna( method ='bfill', inplace = True) 
    df["dew_temperature_min"].fillna( method ='bfill', inplace = True) 
    df["precip_depth_1_hr_min"].fillna( method ='bfill', inplace = True) 
    df["air_temperature_max"].fillna( method ='bfill', inplace = True) 
    df["dew_temperature_max"].fillna( method ='bfill', inplace = True) 
    df["precip_depth_1_hr_max"].fillna( method ='bfill', inplace = True)
    
    return df


# In[ ]:


df_train_preprocess=reduce_mem_usage(rolling_feature(df_train_preprocess))
df_test_preprocess=reduce_mem_usage(rolling_feature(df_test_preprocess))


# In[ ]:


df_train_preprocess.tail(10)


# ### baseline model

# In[ ]:


# # force the model to use the weather data instead of dates, to avoid overfitting to the past history
features = [col for col in df_train_preprocess.columns if col not in ['timestamp','date','meter_reading_transform', 'year', 'month', 'day']]


# In[ ]:


features


# In[ ]:


folds = 4
seed = 42
target='meter_reading_transform'
kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
models = []
oof_pred = np.zeros(df_train_preprocess.shape[0])  # out of fold predictions

## stratify data by building_id
for i, (tr_idx, val_idx) in tqdm(enumerate(kf.split(df_train_preprocess, df_train_preprocess['meter'])), total=folds):
    def fit_regressor(tr_idx, val_idx): # memory closure
        tr_x, tr_y = df_train_preprocess[features].iloc[tr_idx],  df_train_preprocess[target].iloc[tr_idx]
        vl_x, vl_y = df_train_preprocess[features].iloc[val_idx], df_train_preprocess[target].iloc[val_idx]
        print({'fold':i, 'train size':len(tr_x), 'eval size':len(vl_x)})

        tr_data = lgb.Dataset(tr_x, label=tr_y)
        vl_data = lgb.Dataset(vl_x, label=vl_y)  
        clf = lgb.LGBMRegressor(n_estimators=1000,
                                learning_rate=0.4,
                                feature_fraction=0.9,
                                subsample=0.25,  # batches of 25% of the data
                                subsample_freq=1,
                                num_leaves=20,
                                lambda_l1=1,  # regularisation
                                lambda_l2=1,
                                metric='rmse')
        clf.fit(tr_x, tr_y,
                eval_set=[(vl_x, vl_y)],
#                 early_stopping_rounds=50,
                verbose=200)
        # out of fold predictions
        valid_prediticion = clf.predict(vl_x, num_iteration=clf.best_iteration_)
        oof_loss = np.sqrt(mean_squared_error(vl_y, valid_prediticion)) # target is already in log scale
        print(f'Fold:{i} RMSLE: {oof_loss:.4f}')
        return clf, valid_prediticion

    clf, oof_pred[val_idx] = fit_regressor(tr_idx, val_idx)
    models.append(clf)
    
gc.collect()


# ## inference base line model

# In[ ]:


oof_loss = np.sqrt(mean_squared_error(df_train_preprocess[target], oof_pred)) # target is already in log scale
print(f'OOF RMSLE: {oof_loss:.4f}')


# #### Feature importance

# In[ ]:


_ = lgb.plot_importance(models[0], importance_type='gain')


# #### submission

# In[ ]:


# split test data into batches
set_size = len(df_test_preprocess)
iterations = 100
batch_size = set_size // iterations

print(set_size, iterations, batch_size)
assert set_size == iterations * batch_size


# In[ ]:


len (models)


# In[ ]:


meter_reading = []
for i in tqdm(range(iterations)):
    pos = i*batch_size
    fold_preds = [np.expm1(model.predict(df_test_preprocess[features].iloc[pos : pos+batch_size])) for model in models]
    meter_reading.extend(np.mean(fold_preds, axis=0))

print(len(meter_reading))
assert len(meter_reading) == set_size


# In[ ]:


submission = pd.read_csv(f'{path}/sample_submission.csv')
submission['meter_reading'] = np.clip(meter_reading, a_min=0, a_max=None) # clip min at zero


# In[ ]:


submission.to_csv('submission.csv', index=False)
# submission.head(9)

