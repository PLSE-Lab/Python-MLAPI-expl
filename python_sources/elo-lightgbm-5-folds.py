#!/usr/bin/env python
# coding: utf-8

# The general workflow is:
# 
# 1. Fill NaN with most common value for that column 
# 2. Take the historical / new files, aggregate each column using min, mean, max, standard deviation, count. (including categorical variables which proved to be important)
# 3. A good chunk of the train dataset doesn't have rows for the new files but lightgbm can handle this.  I use a StratifiedKFold of 5 folds to help smooth out predictions for test set.
# 4. Important note for CV vs LB which I learned in the Kaggle forums: drop the columns that don't have a similar distribution of values between the train and test files.  This will cause your predictions on test to be well-prepared by the train data set.
# 4. The features that stand out the most are the ones that represent how much they paid and when they committed those transactions.  I've tried layering on more features as this competition has gone on and some of them have added value to my score. 
# 5.  There's a LOT of crazy outliers (10x standard deviations -> -31 value) that obviously blow out the end RMSE.  If I train just on the non-outliers, my score is in the mid 1's (1.4 ish), but since we have to predict these outliers too, it gets blown out to 3.735.  Whoever can best predict outliers will win this competition, plain and simple.  I tried out doing a classifier on whether we could predict if an outlier exists, but it didn't help my end RMSE (even though the AUC was ~.8).  Curious if anyone had success adding a classifier on the -31 outliers.
# 
# In addition a challenge for this dataset was the size of the data using 16GB of RAM on Kaggle environment.  Just importing the data takes >1 minute.  I used some tactics like pandas .sample() to more efficiently move the data through the pipeline.

# In[ ]:


import numpy as np
import pandas as pd
pd.options.display.max_rows = 999

import datetime
import gc


# In[ ]:


#credit to Ashish Gupta for sharing this function: https://www.kaggle.com/roydatascience/elo-stack-interactions-on-categorical-variables
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


historical_transactions = reduce_mem_usage(pd.read_csv('../input/historical_transactions.csv'))
train = reduce_mem_usage(pd.read_csv('../input/train.csv'))
test = reduce_mem_usage(pd.read_csv('../input/test.csv'))


# # historical_transactions = 29mm rows / 325K card_ids
# # new_merchant_transactions = 1.96mm rows
# # train = 201K card_ids
# # test = 123K card_ids
# 
# # 573  / 2,207 outliers don't have row in new_merchant_transactions
# # 21,931 / 201K card_ids in train don't have row in new_merchant_transactions
# 

# In[ ]:


transx = pd.merge(
    historical_transactions,
    train['card_id'].to_frame(),
    on = 'card_id',
    how = 'inner'
)

test_transx = pd.merge(
    test['card_id'].to_frame(),
    historical_transactions,
    on = 'card_id',
    how = 'inner'
)

del historical_transactions
gc.collect()


# # historical_transactions clean up

# In[ ]:


transx['nan_merchant_id'] = 0
transx.loc[transx.merchant_id.isnull(), 'nan_merchant_id'] = 1
transx.loc[transx.merchant_id.isnull(), 'merchant_id'] = 'M_ID_00a6ca8a8a'

transx['nan_category_3'] = 0
transx.loc[transx.category_3.isnull(), 'nan_category_3'] = 1
transx.loc[transx.category_3.isnull(), 'category_3'] = 'A'

transx['nan_category_2'] = 0
transx.loc[transx.category_2.isnull(), 'nan_category_2'] = 1
transx.loc[transx.category_2.isnull(), 'category_2'] = 1.0

transx['category_1'] = transx['category_1'].map({'Y': 1, 'N': 0})
transx['authorized_flag'] = transx['authorized_flag'].map({'Y': 1, 'N': 0})
transx['category_3'] = transx['category_3'].map({'A': 0, 'B': 1, 'C': 2})

transx['exec_date'] = pd.to_datetime(transx['purchase_date'], format = '%Y%m%d %H:%M:%S')
transx['month'] = pd.DatetimeIndex(transx['exec_date']).month
transx['year'] = pd.DatetimeIndex(transx['exec_date']).year
transx['day'] = pd.DatetimeIndex(transx['exec_date']).day
transx['day_of_year'] = pd.DatetimeIndex(transx['exec_date']).dayofyear
transx['day_of_week'] = pd.DatetimeIndex(transx['exec_date']).dayofweek
transx['is_month_start'] = (pd.DatetimeIndex(transx['exec_date']).is_month_start).astype(int)
transx['is_month_end'] = (pd.DatetimeIndex(transx['exec_date']).is_month_end).astype(int)
transx['is_weekend'] = (pd.DatetimeIndex(transx['exec_date']).dayofweek >= 5).astype(int)
transx['is_weekday'] = (pd.DatetimeIndex(transx['exec_date']).dayofweek < 5).astype(int)
transx['weekday'] = pd.DatetimeIndex(transx['exec_date']).weekday
transx['week_of_year'] = pd.DatetimeIndex(transx['exec_date']).weekofyear
transx['days_since_purchase'] = (datetime.datetime.today() - transx['exec_date']).dt.days
transx['quarter'] = pd.DatetimeIndex(transx['exec_date']).quarter
transx['hour'] = pd.DatetimeIndex(transx['exec_date']).hour
transx['months_since_purchase'] = (((datetime.datetime.today() - transx['exec_date']).dt.days) / 30) + transx['month_lag']
transx['duration'] = transx['purchase_amount'] * transx['months_since_purchase']


# In[ ]:


test_transx['nan_merchant_id'] = 0
test_transx.loc[test_transx.merchant_id.isnull(), 'nan_merchant_id'] = 1
test_transx.loc[test_transx.merchant_id.isnull(), 'merchant_id'] = 'M_ID_00a6ca8a8a'

test_transx['nan_category_3'] = 0
test_transx.loc[test_transx.category_3.isnull(), 'nan_category_3'] = 1
test_transx.loc[test_transx.category_3.isnull(), 'category_3'] = 'A'

test_transx['nan_category_2'] = 0
test_transx.loc[test_transx.category_2.isnull(), 'nan_category_2'] = 1
test_transx.loc[test_transx.category_2.isnull(), 'category_2'] = 1.0

test_transx['category_1'] = test_transx['category_1'].map({'Y': 1, 'N': 0})
test_transx['authorized_flag'] = test_transx['authorized_flag'].map({'Y': 1, 'N': 0})
test_transx['category_3'] = test_transx['category_3'].map({'A': 0, 'B': 1, 'C': 2})

test_transx['exec_date'] = pd.to_datetime(test_transx['purchase_date'], format = '%Y%m%d %H:%M:%S')
test_transx['month'] = pd.DatetimeIndex(test_transx['exec_date']).month
test_transx['year'] = pd.DatetimeIndex(test_transx['exec_date']).year
test_transx['day'] = pd.DatetimeIndex(test_transx['exec_date']).day
test_transx['day_of_year'] = pd.DatetimeIndex(test_transx['exec_date']).dayofyear
test_transx['day_of_week'] = pd.DatetimeIndex(test_transx['exec_date']).dayofweek
test_transx['is_month_start'] = (pd.DatetimeIndex(test_transx['exec_date']).is_month_start).astype(int)
test_transx['is_month_end'] = (pd.DatetimeIndex(test_transx['exec_date']).is_month_end).astype(int)
test_transx['is_weekend'] = (pd.DatetimeIndex(test_transx['exec_date']).dayofweek >= 5).astype(int)
test_transx['is_weekday'] = (pd.DatetimeIndex(test_transx['exec_date']).dayofweek < 5).astype(int)
test_transx['weekday'] = pd.DatetimeIndex(test_transx['exec_date']).weekday
test_transx['week_of_year'] = pd.DatetimeIndex(test_transx['exec_date']).weekofyear
test_transx['days_since_purchase'] = (datetime.datetime.today() - test_transx['exec_date']).dt.days
test_transx['quarter'] = pd.DatetimeIndex(test_transx['exec_date']).quarter
test_transx['hour'] = pd.DatetimeIndex(test_transx['exec_date']).hour
test_transx['months_since_purchase'] = (((datetime.datetime.today() - test_transx['exec_date']).dt.days) / 30) + test_transx['month_lag']
test_transx['duration'] = test_transx['purchase_amount'] * test_transx['months_since_purchase']


# In[ ]:


agg_inputs = {
    'card_id' : ['nunique', 'size'],
    'exec_date' : ['min', 'max'],
    'city_id' : ['nunique'], 
    'installments' : ['mean', 'max', 'min', 'var', 'std', 'sum'],
    'merchant_category_id' : ['nunique'], 
    'month_lag' : ['mean', 'max', 'min', 'var', 'std', 'sum'],
    'purchase_amount' : ['mean', 'max', 'min', 'var', 'std', 'sum'], 
    'category_2': ['nunique', 'mean'],
    'state_id' : ['nunique'],
    'subsector_id' : ['nunique'], 
    'nan_merchant_id' : ['nunique', 'mean', 'sum'], 
    'nan_category_3': ['nunique', 'mean', 'sum'], 
    'nan_category_2': ['nunique', 'mean', 'sum'],
    'authorized_flag': ['sum', 'mean'],
    'category_1' : ['nunique', 'mean', 'sum'],
    'category_3': ['nunique', 'mean', 'sum'],
    'month': ['mean', 'max', 'min', 'var', 'std', 'nunique'], 
    'year': ['mean', 'max', 'min', 'var', 'std', 'nunique'], 
    'day': ['mean', 'max', 'min', 'var', 'std', 'nunique'], 
    'weekday': ['mean', 'max', 'min', 'var', 'std', 'nunique'], 
    'week_of_year': ['mean', 'max', 'min', 'var', 'std', 'nunique'],
    'day_of_year': ['mean', 'max', 'min', 'var', 'std', 'nunique'],
    'day_of_week': ['mean', 'max', 'min', 'var', 'std', 'nunique'],
    'months_since_purchase' : ['mean', 'max', 'min', 'var', 'std', 'sum'],
    'quarter' : ['mean', 'max', 'min', 'var', 'std', 'nunique'],
    'hour' : ['mean', 'max', 'min', 'var', 'std', 'nunique'],
    'is_month_start': ['mean', 'sum'],
    'is_month_end': ['mean', 'sum'],
    'is_weekend': ['mean', 'sum'],
    'is_weekday': ['mean', 'sum'],
    'duration' : ['mean', 'max', 'min', 'var', 'std', 'sum']
}


# In[ ]:


transx_staging = transx[
    [
        'card_id', 
        'exec_date',
        'city_id', 
        'installments',
        'merchant_category_id', 
        'month_lag',
        'purchase_amount', 
        'category_2', 
        'state_id',
        'subsector_id', 
        'nan_merchant_id', 
        'nan_category_3', 
        'nan_category_2',
        'authorized_flag',
        'category_1',
        'category_3', 
        'month', 
        'year', 
        'day', 
        'weekday', 
        'week_of_year',
        'day_of_year',
        'day_of_week',
        'days_since_purchase',
        'is_month_start',
        'is_month_end',
        'is_weekend',
        'is_weekday',
        'quarter',
        'hour',
        'months_since_purchase',
        'duration'
    ]
]

del transx
gc.collect()

transx_staging = transx_staging     .groupby('card_id')     .agg(agg_inputs)     .reset_index()

transx_staging.columns = [
    'h_t_' + '_'.join(col).strip() 
        for col in transx_staging.columns.values
]

transx_staging = transx_staging.rename(columns={transx_staging.columns[0] : 'card_id'})


# In[ ]:


test_transx_staging = test_transx[
    [
        'card_id',
        'exec_date',
        'city_id',
        'installments',
        'merchant_category_id', 
        'month_lag',
        'purchase_amount', 
        'category_2', 
        'state_id',
        'subsector_id', 
        'nan_merchant_id', 
        'nan_category_3', 
        'nan_category_2',
        'authorized_flag',
        'category_1',
        'category_3', 
        'month', 
        'year', 
        'day', 
        'weekday', 
        'week_of_year',
        'day_of_year',
        'day_of_week',
        'days_since_purchase',
        'is_month_start',
        'is_month_end',
        'is_weekend',
        'is_weekday',
        'quarter',
        'hour',
        'months_since_purchase',
        'duration'
    ]
]

del test_transx
gc.collect()

test_transx_staging = test_transx_staging     .groupby('card_id')     .agg(agg_inputs)     .reset_index()

test_transx_staging.columns = [
    'h_t_' + '_'.join(col).strip() 
        for col in test_transx_staging.columns.values
]

test_transx_staging = test_transx_staging.rename(columns={test_transx_staging.columns[0] : 'card_id'})


# In[ ]:


train = pd.merge(
    train,
    transx_staging,
    on = 'card_id',
    how = 'left'
)

train['first_purch'] = pd.to_datetime(train['first_active_month'], format = '%Y%m%d %H:%M:%S')
train['first_month'] = pd.DatetimeIndex(train['first_purch']).month
train['first_year'] = pd.DatetimeIndex(train['first_purch']).year
train['days'] = pd.DatetimeIndex(train['first_purch']).day
train['first_quarter'] = pd.DatetimeIndex(train['first_purch']).quarter
train['first_week'] = pd.DatetimeIndex(train['first_purch']).weekofyear
train['first_day_of_week'] = pd.DatetimeIndex(train['first_purch']).dayofweek

train['days_feature1'] = train['days'] * train['feature_1']
train['days_feature2'] = train['days'] * train['feature_2']
train['days_feature3'] = train['days'] * train['feature_3']

test = pd.merge(
    test,
    test_transx_staging,
    on = 'card_id',
    how = 'left'
)

test['first_purch'] = pd.to_datetime(test['first_active_month'], format = '%Y%m%d %H:%M:%S')
test['first_month'] = pd.DatetimeIndex(test['first_purch']).month
test['first_year'] = pd.DatetimeIndex(test['first_purch']).year
test['days'] = pd.DatetimeIndex(test['first_purch']).day
test['first_quarter'] = pd.DatetimeIndex(test['first_purch']).quarter
test['first_week'] = pd.DatetimeIndex(test['first_purch']).weekofyear
test['first_day_of_week'] = pd.DatetimeIndex(test['first_purch']).dayofweek

test['days_feature1'] = test['days'] * train['feature_1']
test['days_feature2'] = test['days'] * train['feature_2']
test['days_feature3'] = test['days'] * train['feature_3']

del [
    test_transx_staging,
    transx_staging
]

gc.collect()


# In[ ]:


new_merchant_transactions = reduce_mem_usage(pd.read_csv('../input/new_merchant_transactions.csv'))

new_transx = pd.merge(
    new_merchant_transactions,
    train['card_id'].to_frame(),
    on = 'card_id',
    how = 'inner'
)

test_new_transx = pd.merge(
    test['card_id'].to_frame(),
    new_merchant_transactions,
    on = 'card_id',
    how = 'inner'
)

del new_merchant_transactions
gc.collect()


# # new_merchant_transactions clean up

# In[ ]:


new_transx['nan_merchant_id'] = 0
new_transx.loc[new_transx.merchant_id.isnull(), 'nan_merchant_id'] = 1
new_transx.loc[new_transx.merchant_id.isnull(), 'merchant_id'] = 'M_ID_00a6ca8a8a'

new_transx['nan_category_3'] = 0
new_transx.loc[new_transx.category_3.isnull(), 'nan_category_3'] = 1
new_transx.loc[new_transx.category_3.isnull(), 'category_3'] = 'A'

new_transx['nan_category_2'] = 0
new_transx.loc[new_transx.category_2.isnull(), 'nan_category_2'] = 1
new_transx.loc[new_transx.category_2.isnull(), 'category_2'] = 1.0

new_transx['category_1'] = new_transx['category_1'].map({'Y': 1, 'N': 0})
new_transx['authorized_flag'] = new_transx['authorized_flag'].map({'Y': 1, 'N': 0})
new_transx['category_3'] = new_transx['category_3'].map({'A': 0, 'B': 1, 'C': 2})

new_transx['exec_date'] = pd.to_datetime(new_transx['purchase_date'], format = '%Y%m%d %H:%M:%S')
new_transx['month'] = pd.DatetimeIndex(new_transx['exec_date']).month
new_transx['year'] = pd.DatetimeIndex(new_transx['exec_date']).year
new_transx['day'] = pd.DatetimeIndex(new_transx['exec_date']).day
new_transx['day_of_year'] = pd.DatetimeIndex(new_transx['exec_date']).dayofyear
new_transx['day_of_week'] = pd.DatetimeIndex(new_transx['exec_date']).dayofweek
new_transx['is_month_start'] = (pd.DatetimeIndex(new_transx['exec_date']).is_month_start).astype(int)
new_transx['is_month_end'] = (pd.DatetimeIndex(new_transx['exec_date']).is_month_end).astype(int)
new_transx['is_weekend'] = (pd.DatetimeIndex(new_transx['exec_date']).dayofweek >= 5).astype(int)
new_transx['is_weekday'] = (pd.DatetimeIndex(new_transx['exec_date']).dayofweek < 5).astype(int)
new_transx['weekday'] = pd.DatetimeIndex(new_transx['exec_date']).weekday
new_transx['week_of_year'] = pd.DatetimeIndex(new_transx['exec_date']).weekofyear
new_transx['quarter'] = pd.DatetimeIndex(new_transx['exec_date']).quarter
new_transx['hour'] = pd.DatetimeIndex(new_transx['exec_date']).hour
new_transx['days_since_purchase'] = (datetime.datetime.today() - new_transx['exec_date']).dt.days
new_transx['months_since_purchase'] = (((datetime.datetime.today() - new_transx['exec_date']).dt.days) / 30) + new_transx['month_lag']
new_transx['duration'] = new_transx['purchase_amount'] * new_transx['months_since_purchase']


# In[ ]:


test_new_transx['nan_merchant_id'] = 0
test_new_transx.loc[test_new_transx.merchant_id.isnull(), 'nan_merchant_id'] = 1
test_new_transx.loc[test_new_transx.merchant_id.isnull(), 'merchant_id'] = 'M_ID_00a6ca8a8a'

test_new_transx['nan_category_3'] = 0
test_new_transx.loc[test_new_transx.category_3.isnull(), 'nan_category_3'] = 1
test_new_transx.loc[test_new_transx.category_3.isnull(), 'category_3'] = 'A'

test_new_transx['nan_category_2'] = 0
test_new_transx.loc[test_new_transx.category_2.isnull(), 'nan_category_2'] = 1
test_new_transx.loc[test_new_transx.category_2.isnull(), 'category_2'] = 1.0

test_new_transx['exec_date'] = pd.to_datetime(test_new_transx['purchase_date'], format = '%Y%m%d %H:%M:%S')
test_new_transx['month'] = pd.DatetimeIndex(test_new_transx['exec_date']).month
test_new_transx['year'] = pd.DatetimeIndex(test_new_transx['exec_date']).year
test_new_transx['day'] = pd.DatetimeIndex(test_new_transx['exec_date']).day
test_new_transx['day_of_year'] = pd.DatetimeIndex(test_new_transx['exec_date']).dayofyear
test_new_transx['day_of_week'] = pd.DatetimeIndex(test_new_transx['exec_date']).dayofweek
test_new_transx['is_month_start'] = (pd.DatetimeIndex(test_new_transx['exec_date']).is_month_start).astype(int)
test_new_transx['is_month_end'] = (pd.DatetimeIndex(test_new_transx['exec_date']).is_month_end).astype(int)
test_new_transx['is_weekend'] = (pd.DatetimeIndex(test_new_transx['exec_date']).dayofweek >= 5).astype(int)
test_new_transx['is_weekday'] = (pd.DatetimeIndex(test_new_transx['exec_date']).dayofweek < 5).astype(int)
test_new_transx['weekday'] = pd.DatetimeIndex(test_new_transx['exec_date']).weekday
test_new_transx['week_of_year'] = pd.DatetimeIndex(test_new_transx['exec_date']).weekofyear
test_new_transx['quarter'] = pd.DatetimeIndex(test_new_transx['exec_date']).quarter
test_new_transx['hour'] = pd.DatetimeIndex(test_new_transx['exec_date']).hour
test_new_transx['days_since_purchase'] = (datetime.datetime.today() - test_new_transx['exec_date']).dt.days
test_new_transx['category_1'] = test_new_transx['category_1'].map({'Y': 1, 'N': 0})
test_new_transx['authorized_flag'] = test_new_transx['authorized_flag'].map({'Y': 1, 'N': 0})
test_new_transx['category_3'] = test_new_transx['category_3'].map({'A': 0, 'B': 1, 'C': 2})
test_new_transx['months_since_purchase'] = (((datetime.datetime.today() - test_new_transx['exec_date']).dt.days) / 30) + test_new_transx['month_lag']
test_new_transx['duration'] = test_new_transx['purchase_amount'] * test_new_transx['months_since_purchase']


# # Aggregate historical / new files

# In[ ]:


new_transx_staging = new_transx[
    [
        'card_id', 
        'exec_date',
        'city_id', 
        'installments',
        'merchant_category_id', 
        'month_lag',
        'purchase_amount', 
        'category_2', 
        'state_id',
        'subsector_id', 
        'nan_merchant_id', 
        'nan_category_3', 
        'nan_category_2',
        'authorized_flag',
        'category_1',
        'category_3', 
        'month', 
        'year', 
        'day', 
        'weekday', 
        'week_of_year',
        'day_of_year',
        'day_of_week',
        'days_since_purchase',
        'is_month_start',
        'is_month_end',
        'is_weekend',
        'is_weekday',
        'quarter',
        'hour',
        'months_since_purchase',
        'duration'
    ]
].reset_index(drop = True)

del new_transx
gc.collect()

new_transx_staging = new_transx_staging     .groupby('card_id')     .agg(agg_inputs)     .reset_index()

new_transx_staging.columns = [
    'new_' + '_'.join(col).strip() 
        for col in new_transx_staging.columns.values
]

new_transx_staging = new_transx_staging.rename(columns={new_transx_staging.columns[0] : 'card_id'})


# In[ ]:


test_new_transx_staging = test_new_transx[
    [
        'card_id', 
        'exec_date',
        'city_id', 
        'installments',
        'merchant_category_id', 
        'month_lag',
        'purchase_amount', 
        'category_2', 
        'state_id',
        'subsector_id', 
        'nan_merchant_id', 
        'nan_category_3', 
        'nan_category_2',
        'authorized_flag',
        'category_1',
        'category_3', 
        'month', 
        'year', 
        'day', 
        'weekday', 
        'week_of_year',
        'day_of_year',
        'day_of_week',
        'days_since_purchase',
        'is_month_start',
        'is_month_end',
        'is_weekend',
        'is_weekday',
        'quarter',
        'hour',
        'months_since_purchase',
        'duration'
    ]
].reset_index(drop = True)

del test_new_transx
gc.collect()

test_new_transx_staging = test_new_transx_staging     .groupby('card_id')     .agg(agg_inputs)     .reset_index()

test_new_transx_staging.columns = [
    'new_' + '_'.join(col).strip() 
        for col in test_new_transx_staging.columns.values
]

test_new_transx_staging = test_new_transx_staging.rename(columns={test_new_transx_staging.columns[0] : 'card_id'})


# # Bring it all together

# In[ ]:


train = pd.merge(
    train,
    new_transx_staging,
    on = 'card_id',
    how = 'left'
)

test = pd.merge(
    test,
    test_new_transx_staging,
    on = 'card_id',
    how = 'left'
)

del [
    test_new_transx_staging,
    new_transx_staging
]

gc.collect()


# # Domain features
# ## DONE Sum historical and new purchase amounts
# ## DONE Average day of purchase
# ## DONE Min / Max day of year
# ## DONE How dense are purchases?
# ## DONE Time of day
# ## DONE How consistent are they with amounts? Standard dev of amounts
# ## Store level score averages
# ## DONE Pct authorized flag = N
# ## DONE First month / year -> get more granular
# ## Month lag?

# In[ ]:


import datetime

train['new_hist_purch_amt_max'] = train['h_t_purchase_amount_max'] + train['new_purchase_amount_max']
test['new_hist_purch_amt_max'] = test['h_t_purchase_amount_max'] + test['new_purchase_amount_max']

train['new_time_elapsed'] = (train['new_exec_date_max'] - train['new_exec_date_min']).dt.days
test['new_time_elapsed'] = (test['new_exec_date_max'] - test['new_exec_date_min']).dt.days
train['h_t_time_elapsed'] = (train['h_t_exec_date_max'] - train['h_t_exec_date_min']).dt.days
test['h_t_time_elapsed'] = (test['h_t_exec_date_max'] - test['h_t_exec_date_min']).dt.days

train['days_since_first_purch'] = (datetime.datetime.today() - train['first_purch']).dt.days
test['days_since_first_purch'] = (datetime.datetime.today() - test['first_purch']).dt.days

train['new_days_since_first_exec'] = (datetime.datetime.today() - train['new_exec_date_min']).dt.days
test['new_days_since_first_exec'] = (datetime.datetime.today() - test['new_exec_date_min']).dt.days
train['h_t_days_since_first_exec'] = (datetime.datetime.today() - train['h_t_exec_date_min']).dt.days
test['h_t_days_since_first_exec'] = (datetime.datetime.today() - test['h_t_exec_date_min']).dt.days

train['new_days_since_last_exec'] = (datetime.datetime.today() - train['new_exec_date_max']).dt.days
test['new_days_since_last_exec'] = (datetime.datetime.today() - test['new_exec_date_max']).dt.days
train['h_t_days_since_last_exec'] = (datetime.datetime.today() - train['h_t_exec_date_max']).dt.days
test['h_t_days_since_last_exec'] = (datetime.datetime.today() - test['h_t_exec_date_max']).dt.days

train['h_t_avg_purch_per_day'] = train['h_t_time_elapsed'] / train['h_t_card_id_size']
test['h_t_avg_purch_per_day'] = test['h_t_time_elapsed'] / test['h_t_card_id_size']
train['new_avg_purch_per_day'] = train['new_time_elapsed'] / train['new_card_id_size']
test['new_avg_purch_per_day'] = test['new_time_elapsed'] / test['new_card_id_size']

train['h_t_days_between_first_purchases'] = (train['h_t_exec_date_min'] - train['first_purch']).dt.days
test['h_t_days_between_first_purchases'] = (test['h_t_exec_date_min'] - test['first_purch']).dt.days
train['new_days_between_first_purchases'] = (train['new_exec_date_min'] - train['first_purch']).dt.days
test['new_days_between_first_purchases'] = (test['new_exec_date_min'] - test['first_purch']).dt.days


# # Run predictions, submit csv

# In[ ]:


blah


# In[ ]:


from sklearn.model_selection import StratifiedKFold, RepeatedKFold
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from math import sqrt

x = train.copy().drop(
    [
        #'target',
        'first_active_month',
        'first_purch',
        'card_id',
        'new_exec_date_min',
        'new_exec_date_max',
        'h_t_exec_date_min',
        'h_t_exec_date_max'
    ],
    axis = 1
)

y = train['target']

x_submit = test.copy().drop(
    [
        'first_active_month',
        'first_purch',
        'card_id',
        'new_exec_date_min',
        'new_exec_date_max',
        'h_t_exec_date_min',
        'h_t_exec_date_max'
    ],
    axis = 1
)

param = {'num_leaves': 31,
         'min_data_in_leaf': 27, 
         'objective':'regression',
         'max_depth': -1,
         'learning_rate': 0.015,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9,
         "bagging_seed": 11,
         "metric": 'rmse',
         "lambda_l1": 0.1,
         "verbosity": -1,
         "nthread": 4,
         "random_state": 4950}

folds = StratifiedKFold(n_splits = 5, shuffle = True)
train_predictions = np.zeros(len(train))
test_predictions = np.zeros(len(test))

for train_index, test_index in folds.split(x, x['feature_1']):
    y = x['target']
    x0 = x.drop('target', axis = 1)
    
    trn_data = lgb.Dataset(x0.iloc[train_index], label=y.iloc[train_index])
    val_data = lgb.Dataset(x0.iloc[test_index], label=y.iloc[test_index])
    
    #x_train, x_test = x0.iloc[train_index], x0.iloc[test_index]
    #y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    num_round = 10000
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=-1, early_stopping_rounds = 200)
    train_predictions[test_index] = clf.predict(x0.iloc[test_index], num_iteration=clf.best_iteration)
    
    #train_predictions[test_index] = lgb_model.predict(x_test)

    test_predictions += clf.predict(x_submit, num_iteration=clf.best_iteration) / folds.n_splits

np.sqrt(mean_squared_error(train_predictions, y))


# In[ ]:


predictions = pd.DataFrame(
    data = {
        'card_id' : test['card_id'],
        'target' : test_predictions
    }
).to_csv('submit.csv', index = False) 

