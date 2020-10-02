#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# General imports
import numpy as np
import pandas as pd
import os, gc, sys, warnings, random, math, psutil, pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold

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
N_SPLITS = 2
LOCAl_TEST = False
USE_EXTERNAL_MODELS = True
EXTERNAL_PATH = '../input/ashrae-lgbm-external-models/'

if USE_EXTERNAL_MODELS:
    external_models = []
    for i in range(N_SPLITS):
        external_models.append(EXTERNAL_PATH + 'lgbm__fold_' + str(i)  + '.bin')
    BATCH_SIZE = 500000
else:
    BATCH_SIZE = 2000000
    
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


########################### Model params
import lightgbm as lgb
lgb_params = {
                    'objective':'regression',
                    'boosting_type':'gbdt',
                    'metric':'rmse',
                    'n_jobs':-1,
                    'learning_rate':0.05,
                    'num_leaves': 2**8,
                    'max_depth':-1,
                    'tree_learner':'serial',
                    'colsample_bytree': 0.9,
                    'subsample_freq':1,
                    'subsample':0.5,
                    'n_estimators':2000,
                    'max_bin':255,
                    'verbose':-1,
                    'seed': SEED,
                    'early_stopping_rounds':100, 
                } 


# In[ ]:


########################### Features
#################################################################################
remove_columns = [TARGET]
features_columns = [col for col in list(train_df) if col not in remove_columns]

i_cols = [
        'building_id',
        'site_id',
        'primary_use',
        'DT_M',
        'floor_count',
        'building_id_uid_enc', 
        'site_id_uid_enc',
]

for col in i_cols:
    train_df[col] = train_df[col].astype('category')
    test_df[col] = test_df[col].astype('category')


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


########################### Split groups
#################################################################################
split_groups = train_df['building_id'].astype(str) +'_'+ train_df['DT_M'].astype(str)
le = LabelEncoder()
split_groups = le.fit_transform(split_groups).astype(np.int16)


# In[ ]:


########################### Model
#################################################################################

if not USE_EXTERNAL_MODELS:
    # Models saving
    model_filename = 'lgbm'
    models = []

    folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    for i, (idxT, idxV) in enumerate(folds.split(train_df, split_groups)):
        print('Fold', i)

        tr_data = lgb.Dataset(train_df.iloc[idxT][features_columns], label=np.log1p(train_df[TARGET][idxT]))
        vl_data = lgb.Dataset(train_df.iloc[idxV][features_columns], label=np.log1p(train_df[TARGET][idxV]))

        estimator = lgb.train(
                    lgb_params,
                    tr_data,
                    valid_sets = [tr_data,vl_data],
                    verbose_eval = 100,
                )

        pickle.dump(estimator, open(model_filename + '__fold_' + str(i)  + '.bin', 'wb'))
        models.append(model_filename + '__fold_' + str(i)  + '.bin')

    if not LOCAl_TEST:
        del tr_data, train_df, split_groups
        gc.collect()
    
else:
    models = external_models
    del train_df, split_groups


# In[ ]:


########################### Predict
#################################################################################
if not LOCAl_TEST:
    
    # Load test_df from hdd
    test_df = pd.read_pickle('test_df.pkl')
    
    # Remove test_df from hdd
    os.system('rm test_df.pkl')
    
    # Read submission file
    submission = pd.read_csv('../input/ashrae-energy-prediction/sample_submission.csv')

    # Remove row_id for a while
    del submission['row_id']
    
    for model_path in models:
        print('Predictions for', model_path)
        estimator = pickle.load(open(model_path, 'rb'))

        predictions = []
        for batch in range(int(len(test_df)/BATCH_SIZE)+1):
            print('Predicting batch:', batch)
            predictions += list(np.expm1(estimator.predict(test_df[features_columns].iloc[batch*BATCH_SIZE:(batch+1)*BATCH_SIZE])))
            
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

