#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# General imports
import numpy as np
import pandas as pd
import os, gc, sys, warnings, random, math, psutil, pickle

from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')


# In[ ]:


########################### Helpers
#################################################################################
## Seeder
# :seed to make all processes deterministic     # type: int
def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    
## Simple "Memory profilers" to see memory usage
def get_memory_usage():
    return np.round(psutil.Process(os.getpid()).memory_info()[0]/2.**30, 2) 
        
def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

## Memory Reducer
# :df pandas dataframe to reduce size             # type: pd.DataFrame()
# :verbose                                        # type: bool
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        if col!=TARGET:
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


########################### Vars
#################################################################################
SEED = 42
LOCAl_TEST = False
seed_everything(SEED)
TARGET = 'meter_reading'


# In[ ]:


########################### DATA LOAD
#################################################################################
print('Load Data')
train_df = pd.read_pickle('../input/ashrae-data-minification/train.pkl')
test_df = pd.read_pickle('../input/ashrae-data-minification/test.pkl')

building_df = pd.read_pickle('../input/ashrae-data-minification/building_metadata.pkl')

train_weather_df = pd.read_pickle('../input/ashrae-data-minification/weather_train.pkl')
test_weather_df = pd.read_pickle('../input/ashrae-data-minification/weather_test.pkl')


# In[ ]:


########################### Remove 0 meter readings for site_id==0
#################################################################################
df = building_df[building_df['site_id']==0]

train_df['drop'] = np.where(train_df['DT_D']<=140, 1, 0)
train_df['drop'] = np.where(train_df['building_id'].isin(df['building_id']), train_df['drop'], 0)

train_df = train_df[train_df['drop']==0].reset_index(drop=True)

del df, train_df['drop']


# In[ ]:


########################### Building DF merge through concat 
#################################################################################
# Benefits of concat:
## Faster for huge datasets (columns number)
## No dtype change for dataset
## Consume less memmory 

temp_df = train_df[['building_id']]
temp_df = temp_df.merge(building_df, on=['building_id'], how='left')
del temp_df['building_id']
train_df = pd.concat([train_df, temp_df], axis=1)

temp_df = test_df[['building_id']]
temp_df = temp_df.merge(building_df, on=['building_id'], how='left')
del temp_df['building_id']
test_df = pd.concat([test_df, temp_df], axis=1)

del building_df, temp_df


# In[ ]:


########################### Weather DF merge over concat (to not lose type)
#################################################################################
# Benefits of concat:
## Faster for huge datasets (columns number)
## No dtype change for dataset
## Consume less memmory 

temp_df = train_df[['site_id','timestamp']]
temp_df = temp_df.merge(train_weather_df, on=['site_id','timestamp'], how='left')
del temp_df['site_id'], temp_df['timestamp']
train_df = pd.concat([train_df, temp_df], axis=1)

temp_df = test_df[['site_id','timestamp']]
temp_df = temp_df.merge(test_weather_df, on=['site_id','timestamp'], how='left')
del temp_df['site_id'], temp_df['timestamp']
test_df = pd.concat([test_df, temp_df], axis=1)

del train_weather_df, test_weather_df, temp_df


# In[ ]:


########################### Delete some columns
#################################################################################
del test_df['row_id']

i_cols = [
         'timestamp',
         'DT_D',
         'DT_day_month',
         'DT_week_month',
        ]

for col in i_cols:
    try:
        del train_df[col], test_df[col]
    except:
        pass


# In[ ]:


########################### Smooth readings
#################################################################################
train_df['s_uid'] = train_df['site_id'].astype(str) +'_'+                    train_df['DT_M'].astype(str) +'_'+                    train_df['meter'].astype(str) +'_'+                    train_df['primary_use'].astype(str)

temp_df = train_df.groupby(['s_uid'])[TARGET].apply(lambda x: int(np.percentile(x,99)))
temp_df = temp_df.to_dict()

train_df['s_uid'] = train_df['s_uid'].map(temp_df)
train_df[TARGET] = np.where(train_df[TARGET]>train_df['s_uid'], train_df['s_uid'], train_df[TARGET])

del train_df['s_uid'], temp_df


# In[ ]:


########################### Encode Meter
#################################################################################
# Building and site id
for enc_col in ['building_id', 'site_id']:
    temp_df = train_df.groupby([enc_col])['meter'].agg(['unique'])
    temp_df['unique'] = temp_df['unique'].apply(lambda x: '_'.join(str(x))).astype(str)

    le = LabelEncoder()
    temp_df['unique'] = le.fit_transform(temp_df['unique']).astype(np.int8)
    temp_df = temp_df['unique'].to_dict()

    train_df[enc_col+'_uid_enc'] = train_df[enc_col].map(temp_df)
    test_df[enc_col+'_uid_enc'] = test_df[enc_col].map(temp_df)
    
    # Nunique
    temp_dict = train_df.groupby([enc_col])['meter'].agg(['nunique'])['nunique'].to_dict()
    train_df[enc_col+'-m_nunique'] = train_df[enc_col].map(temp_dict).astype(np.int8)
    test_df[enc_col+'-m_nunique'] = test_df[enc_col].map(temp_dict).astype(np.int8)

del temp_df, temp_dict


# In[ ]:


########################### Daily temperature
#################################################################################
for df in [train_df, test_df]:
    df['DT_w_hour'] = np.where((df['DT_hour']>5)&(df['DT_hour']<13),1,0)
    df['DT_w_hour'] = np.where((df['DT_hour']>12)&(df['DT_hour']<19),2,df['DT_w_hour'])
    df['DT_w_hour'] = np.where((df['DT_hour']>18),3,df['DT_w_hour'])

    df['DT_w_temp'] = df.groupby(['site_id','DT_W','DT_w_hour'])['air_temperature'].transform('mean')
    df['DT_w_dew_temp'] = df.groupby(['site_id','DT_W','DT_w_hour'])['dew_temperature'].transform('mean')

i_cols = [
         'DT_w_hour',
        ]

for col in i_cols:
    del train_df[col], test_df[col]


# In[ ]:


########################### Reduce memory usage
#################################################################################
train_df = reduce_mem_usage(train_df)
test_df = reduce_mem_usage(test_df)


# In[ ]:


########################### Features
#################################################################################
remove_columns = [TARGET]
features_columns = [col for col in list(train_df) if col not in remove_columns]

categorical_features = [
        'building_id',
        'site_id',
        'primary_use',
        'DT_M',
        'floor_count',
        'building_id_uid_enc', 
        'site_id_uid_enc',
]


# In[ ]:


########################### Store test_df to HDD and cleanup
#################################################################################
test_df[features_columns].to_pickle('test_df.pkl')

df = 0
temp_df = 0
temp_dict = 0
i_cols = 0
col = 0

del test_df
del df, temp_df, temp_dict
del col, i_cols
gc.collect()


# In[ ]:


########################### Check memory usage
#################################################################################
for name, size in sorted(((name, sys.getsizeof(value)) for name,value in locals().items()),
                         key= lambda x: -x[1])[:10]:
    print("{:>30}: {:>8}".format(name,sizeof_fmt(size)))
print('Memory in Gb', get_memory_usage())


# In[ ]:


########################### Catboost Model
from catboost import CatBoostRegressor

model_filename = 'catboost'
models = []

cat_params = {
        'n_estimators': 2000,
        'learning_rate': 0.1,
        'eval_metric': 'RMSE',
        'loss_function': 'RMSE',
        'random_seed': SEED,
        'metric_period': 10,
        'task_type': 'GPU',
        'depth': 8,
    }

estimator = CatBoostRegressor(**cat_params)
estimator.fit(
            train_df[features_columns], np.log1p(train_df[TARGET]),
            cat_features=categorical_features,
            verbose=True)

estimator.save_model(model_filename + '.bin')
models.append(model_filename + '.bin')

del estimator
gc.collect()


# In[ ]:


########################### Predict
#################################################################################
if not LOCAl_TEST:
   
    # delete train_df
    del train_df

    # Read test file
    test_df = pd.read_pickle('test_df.pkl')
    
    # Remove test_df from hdd
    os.system('rm test_df.pkl')
 
    # Read submission file
    submission = pd.read_csv('../input/ashrae-energy-prediction/sample_submission.csv')

    # Remove row_id for a while
    del submission['row_id']
    
    for model_path in models:
        print('Predictions for', model_path)
        
        if 'catboost' in model_path:
            estimator = CatBoostRegressor()
            estimator.load_model(model_path)
        else:
            estimator = pickle.load(open(model_path, 'rb'))

        predictions = []
        batch_size = 500000
        for batch in range(int(len(test_df)/batch_size)+1):
            print('Predicting batch:', batch)
            predictions += list(np.expm1(estimator.predict(test_df[features_columns].iloc[batch*batch_size:(batch+1)*batch_size])))
            
        submission['meter_reading'] += predictions
        
    # Average over models
    submission['meter_reading'] /= len(models)
    
    # Delete test_df
    del test_df
     
    # Fix negative values
    submission['meter_reading'] = submission['meter_reading'].clip(0,None)

    # Restore row_id
    submission['row_id'] = submission.index
    
    ########################### Check
    print(submission.iloc[:20])
    print(submission['meter_reading'].describe())


# In[ ]:


########################### Export
#################################################################################
if not LOCAl_TEST:
    submission.to_csv('submission.csv', index=False)

