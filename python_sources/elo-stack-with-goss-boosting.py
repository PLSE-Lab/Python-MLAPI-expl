#!/usr/bin/env python
# coding: utf-8

# This is my revised kernel inspired by Chau Ngoc Huynh's kernel (3.699).
# 
# **New Update : Application of Interaction on Categorical Variables followed by Stacking using Bayesian Ridge on Stratified K Folds**

# In[ ]:


# Import the necessary libraries
import numpy as np
import pandas as pd
import os
import time
import warnings
import gc
gc.collect()
import os
from six.moves import urllib
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn')
from scipy.stats import norm, skew
from sklearn.preprocessing import StandardScaler


# In[ ]:


#Add All the Models Libraries

# Scalers
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion

# Models

from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_log_error,mean_squared_error, r2_score,mean_absolute_error

from sklearn.model_selection import train_test_split #training and testing data split
from sklearn import metrics #accuracy measure
from sklearn.metrics import confusion_matrix #for confusion matrix
from scipy.stats import reciprocal, uniform

from sklearn.model_selection import StratifiedKFold, RepeatedKFold

# Cross-validation
from sklearn.model_selection import KFold #for K-fold cross validation
from sklearn.model_selection import cross_val_score #score evaluation
from sklearn.model_selection import cross_val_predict #prediction
from sklearn.model_selection import cross_validate

# GridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

#Common data processors
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from scipy import sparse


# In[ ]:


# to make this notebook's output stable across runs
np.random.seed(123)
gc.collect()
# To plot pretty figures
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12


# In[ ]:


#Reduce the memory usage - Inspired by Panchajanya Banerjee
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


def one_hot_encoder(df, nan_as_category=True):
    original_columns = df.columns.tolist()

    categorical_columns = list(filter(lambda c: c in ['object'], df.dtypes))
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)

    new_columns = list(filter(lambda c: c not in original_columns, df.columns))
    return df, new_columns


# In[ ]:


train = reduce_mem_usage(pd.read_csv('../input/train.csv',parse_dates=["first_active_month"]))
test = reduce_mem_usage(pd.read_csv('../input/test.csv', parse_dates=["first_active_month"]))


# In[ ]:


train.shape


# In[ ]:


train.info()


# In[ ]:


plt.subplot(1, 2, 2)
sns.distplot(train.target, kde=True, fit = norm)
plt.xlabel('Customer Loyality (Skewed)')


# In[ ]:


# Remove the Outliers if any 
train['outliers'] = 0
train.loc[train['target'] < -30, 'outliers'] = 1
train['outliers'].value_counts()

for features in ['feature_1','feature_2','feature_3']:
    order_label = train.groupby([features])['outliers'].mean()
    train[features] = train[features].map(order_label)
    test[features] =  test[features].map(order_label)


# In[ ]:


# Now extract the days and Qtr
train['days'] = (datetime.date(2018, 2, 1) - train['first_active_month'].dt.date).dt.days
train['quarter'] = train['first_active_month'].dt.quarter

feature_cols = ['feature_1', 'feature_2', 'feature_3']
for f in feature_cols:
    train['days_' + f] = train['days'] * train[f]
    train['days_' + f + '_ratio'] = train[f] / train['days']

test['days'] = (datetime.date(2018, 2, 1) - test['first_active_month'].dt.date).dt.days
test['quarter'] = test['first_active_month'].dt.quarter

feature_cols = ['feature_1', 'feature_2', 'feature_3']
for f in feature_cols:
    test['days_' + f] = test['days'] * test[f]
    test['days_' + f + '_ratio'] = test[f] / test['days']

gc.collect()


# Now we will try to extract more features from Transactions Data

# In[ ]:


def aggregate_transaction_hist(trans, prefix):  
        
    agg_func = {
        'purchase_amount' : ['sum','max','min','mean','var','skew'],
        'installments' : ['sum','max','mean','var','skew'],
        'purchase_date' : ['max','min'],
        'month_lag' : ['max','min','mean','var','skew'],
        'month_diff' : ['max','min','mean','var','skew'],
        'weekend' : ['sum', 'mean'],
        'weekday' : ['sum', 'mean'],
        'authorized_flag': ['sum', 'mean'],
        'category_1': ['sum','mean', 'max','min'],
        'card_id' : ['size','count'],
        'month': ['nunique', 'mean', 'min', 'max'],
        'hour': ['nunique', 'mean', 'min', 'max'],
        'weekofyear': ['nunique', 'mean', 'min', 'max'],
        'day': ['nunique', 'mean', 'min', 'max'],
        'subsector_id': ['nunique'],
        'merchant_category_id' : ['nunique'],
        'price' :['sum','mean','max','min','var'],
        'duration' : ['mean','min','max','var','skew'],
        'amount_month_ratio':['mean','min','max','var','skew']
        
    }
    
    agg_trans = trans.groupby(['card_id']).agg(agg_func)
    agg_trans.columns = [prefix + '_'.join(col).strip() 
                           for col in agg_trans.columns.values]
    agg_trans.reset_index(inplace=True)
    
    df = (trans.groupby('card_id')
          .size()
          .reset_index(name='{}transactions_count'.format(prefix)))
    
    agg_trans = pd.merge(df, agg_trans, on='card_id', how='left')
    
    return agg_trans


# In[ ]:


transactions = reduce_mem_usage(pd.read_csv('../input/historical_transactions.csv'))
gc.collect()


# In[ ]:


#impute missing values - This is now excluded.
transactions['category_2'] = transactions['category_2'].fillna(1.0,inplace=True)
transactions['category_3'] = transactions['category_3'].fillna('A',inplace=True)
transactions['merchant_id'] = transactions['merchant_id'].fillna('M_ID_00a6ca8a8a',inplace=True)
transactions['installments'].replace(-1, np.nan,inplace=True)
transactions['installments'].replace(999, np.nan,inplace=True)
transactions['purchase_amount'] = transactions['purchase_amount'].apply(lambda x: min(x, 0.8))

#Feature Engineering - Adding new features inspired by Chau's first kernel
transactions['authorized_flag'] = transactions['authorized_flag'].map({'Y': 1, 'N': 0})
transactions['category_1'] = transactions['category_1'].map({'Y': 1, 'N': 0})
transactions['category_3'] = transactions['category_3'].map({'A':0, 'B':1, 'C':2})

transactions['purchase_date'] = pd.to_datetime(transactions['purchase_date'])
transactions['weekofyear'] = transactions['purchase_date'].dt.weekofyear
transactions['month'] = transactions['purchase_date'].dt.month
transactions['day'] = transactions['purchase_date'].dt.day
transactions['weekday'] = transactions.purchase_date.dt.weekday
transactions['weekend'] = (transactions.purchase_date.dt.weekday >=5).astype(int)
transactions['hour'] = transactions['purchase_date'].dt.hour 
transactions['month_diff'] = ((datetime.datetime.today() - transactions['purchase_date']).dt.days)//30
transactions['month_diff'] += transactions['month_lag']

# additional features
transactions['duration'] = transactions['purchase_amount']*transactions['month_diff']
transactions['amount_month_ratio'] = transactions['purchase_amount']/transactions['month_diff']
transactions['price'] = transactions['purchase_amount'] / transactions['installments']

gc.collect()


# In[ ]:


agg_func = {
        'mean': ['mean'],
    }
for col in ['category_2','category_3']:
    transactions[col+'_mean'] = transactions['purchase_amount'].groupby(transactions[col]).agg('mean')
    transactions[col+'_max'] = transactions['purchase_amount'].groupby(transactions[col]).agg('max')
    transactions[col+'_min'] = transactions['purchase_amount'].groupby(transactions[col]).agg('min')
    transactions[col+'_sum'] = transactions['purchase_amount'].groupby(transactions[col]).agg('sum')
    agg_func[col+'_mean'] = ['mean']
    
gc.collect()


# In[ ]:


merge_trans = aggregate_transaction_hist(transactions, prefix='hist_')
del transactions
gc.collect()
train = pd.merge(train, merge_trans, on='card_id',how='left')
test = pd.merge(test, merge_trans, on='card_id',how='left')
del merge_trans
gc.collect()


# In[ ]:


#Feature Engineering - Adding new features inspired by Chau's first kernel
train['hist_purchase_date_max'] = pd.to_datetime(train['hist_purchase_date_max'])
train['hist_purchase_date_min'] = pd.to_datetime(train['hist_purchase_date_min'])
train['hist_purchase_date_diff'] = (train['hist_purchase_date_max'] - train['hist_purchase_date_min']).dt.days
train['hist_purchase_date_average'] = train['hist_purchase_date_diff']/train['hist_card_id_size']
train['hist_purchase_date_uptonow'] = (datetime.datetime.today() - train['hist_purchase_date_max']).dt.days
train['hist_purchase_date_uptomin'] = (datetime.datetime.today() - train['hist_purchase_date_min']).dt.days
train['hist_first_buy'] = (train['hist_purchase_date_min'] - train['first_active_month']).dt.days
train['hist_last_buy'] = (train['hist_purchase_date_max'] - train['first_active_month']).dt.days

for feature in ['hist_purchase_date_max','hist_purchase_date_min']:
    train[feature] = train[feature].astype(np.int64) * 1e-9
gc.collect()


# In[ ]:


#Feature Engineering - Adding new features inspired by Chau's first kernel
test['hist_purchase_date_max'] = pd.to_datetime(test['hist_purchase_date_max'])
test['hist_purchase_date_min'] = pd.to_datetime(test['hist_purchase_date_min'])
test['hist_purchase_date_diff'] = (test['hist_purchase_date_max'] - test['hist_purchase_date_min']).dt.days
test['hist_purchase_date_average'] = test['hist_purchase_date_diff']/test['hist_card_id_size']
test['hist_purchase_date_uptonow'] = (datetime.datetime.today() - test['hist_purchase_date_max']).dt.days
test['hist_purchase_date_uptomin'] = (datetime.datetime.today() - test['hist_purchase_date_min']).dt.days

test['hist_first_buy'] = (test['hist_purchase_date_min'] - test['first_active_month']).dt.days
test['hist_last_buy'] = (test['hist_purchase_date_max'] - test['first_active_month']).dt.days

for feature in ['hist_purchase_date_max','hist_purchase_date_min']:
    test[feature] = test[feature].astype(np.int64) * 1e-9

gc.collect()


# In[ ]:


train.head(10)


# In[ ]:


# Taking Reference from Other Kernels
def aggregate_transaction_new(trans, prefix):  
        
    agg_func = {
        'purchase_amount' : ['sum','max','min','mean','var','skew'],
        'installments' : ['sum','max','mean','var','skew'],
        'purchase_date' : ['max','min'],
        'month_lag' : ['max','min','mean','var','skew'],
        'month_diff' : ['max','min','mean','var','skew'],
        'weekend' : ['sum', 'mean'],
        'weekday' : ['sum', 'mean'],
        'authorized_flag': ['sum', 'mean'],
        'category_1': ['sum','mean', 'max','min'],
        'card_id' : ['size','count'],
        'month': ['nunique', 'mean', 'min', 'max'],
        'hour': ['nunique', 'mean', 'min', 'max'],
        'weekofyear': ['nunique', 'mean', 'min', 'max'],
        'day': ['nunique', 'mean', 'min', 'max'],
        'subsector_id': ['nunique'],
        'merchant_category_id' : ['nunique'],
        'price' :['sum','mean','max','min','var'],
        'duration' : ['mean','min','max','var','skew'],
        'amount_month_ratio':['mean','min','max','var','skew']
    }
    
    agg_trans = trans.groupby(['card_id']).agg(agg_func)
    agg_trans.columns = [prefix + '_'.join(col).strip() 
                           for col in agg_trans.columns.values]
    agg_trans.reset_index(inplace=True)
    
    df = (trans.groupby('card_id')
          .size()
          .reset_index(name='{}transactions_count'.format(prefix)))
    
    agg_trans = pd.merge(df, agg_trans, on='card_id', how='left')
    
    return agg_trans


# In[ ]:


# Now extract the data from the new transactions
new_transactions = reduce_mem_usage(pd.read_csv('../input/new_merchant_transactions.csv'))


# In[ ]:


#impute missing values
new_transactions['category_2'] = new_transactions['category_2'].fillna(1.0,inplace=True)
new_transactions['category_3'] = new_transactions['category_3'].fillna('A',inplace=True)
new_transactions['merchant_id'] = new_transactions['merchant_id'].fillna('M_ID_00a6ca8a8a',inplace=True)
new_transactions['installments'].replace(-1, np.nan,inplace=True)
new_transactions['installments'].replace(999, np.nan,inplace=True)
new_transactions['purchase_amount'] = new_transactions['purchase_amount'].apply(lambda x: min(x, 0.8))

#Feature Engineering - Adding new features inspired by Chau's first kernel
new_transactions['authorized_flag'] = new_transactions['authorized_flag'].map({'Y': 1, 'N': 0})
new_transactions['category_1'] = new_transactions['category_1'].map({'Y': 1, 'N': 0})
new_transactions['category_3'] = new_transactions['category_3'].map({'A':0, 'B':1, 'C':2}) 

new_transactions['purchase_date'] = pd.to_datetime(new_transactions['purchase_date'])
new_transactions['month'] = new_transactions['purchase_date'].dt.month
new_transactions['weekofyear'] = new_transactions['purchase_date'].dt.weekofyear
new_transactions['day'] = new_transactions['purchase_date'].dt.day
new_transactions['weekday'] = new_transactions.purchase_date.dt.weekday
new_transactions['weekend'] = (new_transactions.purchase_date.dt.weekday >=5).astype(int)
new_transactions['hour'] = new_transactions['purchase_date'].dt.hour 
new_transactions['month_diff'] = ((datetime.datetime.today() - new_transactions['purchase_date']).dt.days)//30
new_transactions['month_diff'] += new_transactions['month_lag']

gc.collect()

# additional features
new_transactions['duration'] = new_transactions['purchase_amount']*new_transactions['month_diff']
new_transactions['amount_month_ratio'] = new_transactions['purchase_amount']/new_transactions['month_diff']
new_transactions['price'] = new_transactions['purchase_amount'] / new_transactions['installments']

aggs = {
        'mean': ['mean'],
    }

for col in ['category_2','category_3']:
    new_transactions[col+'_mean'] = new_transactions['purchase_amount'].groupby(new_transactions[col]).agg('mean')
    new_transactions[col+'_max'] = new_transactions['purchase_amount'].groupby(new_transactions[col]).agg('max')
    new_transactions[col+'_min'] = new_transactions['purchase_amount'].groupby(new_transactions[col]).agg('min')
    new_transactions[col+'_var'] = new_transactions['purchase_amount'].groupby(new_transactions[col]).agg('var')
    aggs[col+'_mean'] = ['mean']

gc.collect()


# In[ ]:


merge_new = aggregate_transaction_new(new_transactions, prefix='new_')
del new_transactions
gc.collect()

train = pd.merge(train, merge_new, on='card_id',how='left')
test = pd.merge(test, merge_new, on='card_id',how='left')
del merge_new

gc.collect()


# In[ ]:


#Feature Engineering - Adding new features inspired by Chau's first kernel
train['new_purchase_date_max'] = pd.to_datetime(train['new_purchase_date_max'])
train['new_purchase_date_min'] = pd.to_datetime(train['new_purchase_date_min'])
train['new_purchase_date_diff'] = (train['new_purchase_date_max'] - train['new_purchase_date_min']).dt.days
train['new_purchase_date_average'] = train['new_purchase_date_diff']/train['new_card_id_size']
train['new_purchase_date_uptonow'] = (datetime.datetime.today() - train['new_purchase_date_max']).dt.days
train['new_purchase_date_uptomin'] = (datetime.datetime.today() - train['new_purchase_date_min']).dt.days
train['new_first_buy'] = (train['new_purchase_date_min'] - train['first_active_month']).dt.days
train['new_last_buy'] = (train['new_purchase_date_max'] - train['first_active_month']).dt.days
for feature in ['new_purchase_date_max','new_purchase_date_min']:
    train[feature] = train[feature].astype(np.int64) * 1e-9

#Feature Engineering - Adding new features inspired by Chau's first kernel
test['new_purchase_date_max'] = pd.to_datetime(test['new_purchase_date_max'])
test['new_purchase_date_min'] = pd.to_datetime(test['new_purchase_date_min'])
test['new_purchase_date_diff'] = (test['new_purchase_date_max'] - test['new_purchase_date_min']).dt.days
test['new_purchase_date_average'] = test['new_purchase_date_diff']/test['new_card_id_size']
test['new_purchase_date_uptonow'] = (datetime.datetime.today() - test['new_purchase_date_max']).dt.days
test['new_purchase_date_uptomin'] = (datetime.datetime.today() - test['new_purchase_date_min']).dt.days
test['new_first_buy'] = (test['new_purchase_date_min'] - test['first_active_month']).dt.days
test['new_last_buy'] = (test['new_purchase_date_max'] - test['first_active_month']).dt.days

for feature in ['new_purchase_date_max','new_purchase_date_min']:
    test[feature] = test[feature].astype(np.int64) * 1e-9
    
gc.collect()


# In[ ]:


#NEW Features referred from https://www.kaggle.com/mfjwr1/simple-lightgbm-without-blending
train['card_id_total'] = train['new_card_id_size']+train['hist_card_id_size']
train['card_id_cnt_total'] = train['new_card_id_count']+train['hist_card_id_count']
train['card_id_cnt_ratio'] = train['new_card_id_count']/train['hist_card_id_count']
train['purchase_amount_total'] = train['new_purchase_amount_sum']+train['hist_purchase_amount_sum']
train['purchase_amount_mean'] = train['new_purchase_amount_mean']+train['hist_purchase_amount_mean']
train['purchase_amount_max'] = train['new_purchase_amount_max']+train['hist_purchase_amount_max']
train['purchase_amount_min'] = train['new_purchase_amount_min']+train['hist_purchase_amount_min']
train['purchase_amount_ratio'] = train['new_purchase_amount_sum']/train['hist_purchase_amount_sum']
train['month_diff_mean'] = train['new_month_diff_mean']+train['hist_month_diff_mean']
train['month_diff_ratio'] = train['new_month_diff_mean']/train['hist_month_diff_mean']
train['month_lag_mean'] = train['new_month_lag_mean']+train['hist_month_lag_mean']
train['month_lag_max'] = train['new_month_lag_max']+train['hist_month_lag_max']
train['month_lag_min'] = train['new_month_lag_min']+train['hist_month_lag_min']
train['category_1_mean'] = train['new_category_1_mean']+train['hist_category_1_mean']
train['installments_total'] = train['new_installments_sum']+train['hist_installments_sum']
train['installments_mean'] = train['new_installments_mean']+train['hist_installments_mean']
train['installments_max'] = train['new_installments_max']+train['hist_installments_max']
train['installments_ratio'] = train['new_installments_sum']/train['hist_installments_sum']
train['price_total'] = train['purchase_amount_total'] / train['installments_total']
train['price_mean'] = train['purchase_amount_mean'] / train['installments_mean']
train['price_max'] = train['purchase_amount_max'] / train['installments_max']
train['duration_mean'] = train['new_duration_mean']+train['hist_duration_mean']
train['duration_min'] = train['new_duration_min']+train['hist_duration_min']
train['duration_max'] = train['new_duration_max']+train['hist_duration_max']
train['amount_month_ratio_mean']=train['new_amount_month_ratio_mean']+train['hist_amount_month_ratio_mean']
train['amount_month_ratio_min']=train['new_amount_month_ratio_min']+train['hist_amount_month_ratio_min']
train['amount_month_ratio_max']=train['new_amount_month_ratio_max']+train['hist_amount_month_ratio_max']
train['new_CLV'] = train['new_card_id_count'] * train['new_purchase_amount_sum'] / train['new_month_diff_mean']
train['hist_CLV'] = train['hist_card_id_count'] * train['hist_purchase_amount_sum'] / train['hist_month_diff_mean']
train['CLV_ratio'] = train['new_CLV'] / train['hist_CLV']

test['card_id_total'] = test['new_card_id_size']+test['hist_card_id_size']
test['card_id_cnt_total'] = test['new_card_id_count']+test['hist_card_id_count']
test['card_id_cnt_ratio'] = test['new_card_id_count']/test['hist_card_id_count']
test['purchase_amount_total'] = test['new_purchase_amount_sum']+test['hist_purchase_amount_sum']
test['purchase_amount_mean'] = test['new_purchase_amount_mean']+test['hist_purchase_amount_mean']
test['purchase_amount_max'] = test['new_purchase_amount_max']+test['hist_purchase_amount_max']
test['purchase_amount_min'] = test['new_purchase_amount_min']+test['hist_purchase_amount_min']
test['purchase_amount_ratio'] = test['new_purchase_amount_sum']/test['hist_purchase_amount_sum']
test['month_diff_mean'] = test['new_month_diff_mean']+test['hist_month_diff_mean']
test['month_diff_ratio'] = test['new_month_diff_mean']/test['hist_month_diff_mean']
test['month_lag_mean'] = test['new_month_lag_mean']+test['hist_month_lag_mean']
test['month_lag_max'] = test['new_month_lag_max']+test['hist_month_lag_max']
test['month_lag_min'] = test['new_month_lag_min']+test['hist_month_lag_min']
test['category_1_mean'] = test['new_category_1_mean']+test['hist_category_1_mean']
test['installments_total'] = test['new_installments_sum']+test['hist_installments_sum']
test['installments_mean'] = test['new_installments_mean']+test['hist_installments_mean']
test['installments_max'] = test['new_installments_max']+test['hist_installments_max']
test['installments_ratio'] = test['new_installments_sum']/test['hist_installments_sum']
test['price_total'] = test['purchase_amount_total'] / test['installments_total']
test['price_mean'] = test['purchase_amount_mean'] / test['installments_mean']
test['price_max'] = test['purchase_amount_max'] / test['installments_max']
test['duration_mean'] = test['new_duration_mean']+test['hist_duration_mean']
test['duration_min'] = test['new_duration_min']+test['hist_duration_min']
test['duration_max'] = test['new_duration_max']+test['hist_duration_max']
test['amount_month_ratio_mean']=test['new_amount_month_ratio_mean']+test['hist_amount_month_ratio_mean']
test['amount_month_ratio_min']=test['new_amount_month_ratio_min']+test['hist_amount_month_ratio_min']
test['amount_month_ratio_max']=test['new_amount_month_ratio_max']+test['hist_amount_month_ratio_max']
test['new_CLV'] = test['new_card_id_count'] * test['new_purchase_amount_sum'] / test['new_month_diff_mean']
test['hist_CLV'] = test['hist_card_id_count'] * test['hist_purchase_amount_sum'] / test['hist_month_diff_mean']
test['CLV_ratio'] = test['new_CLV'] / test['hist_CLV']


# In[ ]:


train.head(10)


# In[ ]:


train = train.drop(['card_id', 'first_active_month'], axis = 1)
test = test.drop(['card_id', 'first_active_month'], axis = 1)


# In[ ]:


# Get the X and Y
df_train_columns = [c for c in train.columns if c not in ['first_active_month', 'target', 'card_id', 'outliers',
                  'hist_purchase_date_max', 'hist_purchase_date_min', 'hist_card_id_size',
                  'new_purchase_date_max', 'new_purchase_date_min', 'new_card_id_size']] 
target = train['target']
del train['target']


# In[ ]:


train.shape


# Apply Light GBM Modelling Technique with Stratified K Folds enumerated on training set and outliers 

# In[ ]:


# Change in Hyper Parameters using kernel : https://www.kaggle.com/mfjwr1/simple-lightgbm-without-blending/output
import lightgbm as lgb

folds = StratifiedKFold(n_splits=5, shuffle = True, random_state=4950)
oof = np.zeros(len(train))
predictions = np.zeros(len(test))
feature_importance_df = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, train['outliers'].values)):
    
    print("fold {}".format(fold_))
    trn_data = lgb.Dataset(train.iloc[trn_idx][df_train_columns], label=target.iloc[trn_idx])
    val_data = lgb.Dataset(train.iloc[val_idx][df_train_columns], label=target.iloc[val_idx])

    param ={
                'task': 'train',
                'boosting': 'goss',
                'objective': 'regression',
                'metric': 'rmse',
                'learning_rate': 0.01,
                'subsample': 0.9855232997390695,
                'max_depth': 7,
                'top_rate': 0.9064148448434349,
                'num_leaves': 63,
                'min_child_weight': 41.9612869171337,
                'other_rate': 0.0721768246018207,
                'reg_alpha': 9.677537745007898,
                'colsample_bytree': 0.5665320670155495,
                'min_split_gain': 9.820197773625843,
                'reg_lambda': 8.2532317400459,
                'min_data_in_leaf': 21,
                'verbose': -1,
                'seed':int(2**fold_),
                'bagging_seed':int(2**fold_),
                'drop_seed':int(2**fold_)
                }
    
    num_round = 10000
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=-1, early_stopping_rounds = 200)
    oof[val_idx] = clf.predict(train.iloc[val_idx][df_train_columns], num_iteration=clf.best_iteration)
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = df_train_columns
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    predictions += clf.predict(test[df_train_columns], num_iteration=clf.best_iteration) / folds.n_splits

np.sqrt(mean_squared_error(oof, target))


# Feature Importance - Stratified K Folds Technique

# In[ ]:


cols = (feature_importance_df[["Feature", "importance"]]
        .groupby("Feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:1000].index)

best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]

plt.figure(figsize=(14,25))
sns.barplot(x="importance",
            y="Feature",
            data=best_features.sort_values(by="importance",
                                           ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.savefig('lgbm_importances.png')


# In[ ]:


features = [c for c in train.columns if c not in ['first_active_month', 'target', 'card_id', 'outliers',
                  'hist_purchase_date_max', 'hist_purchase_date_min', 'hist_card_id_size',
                  'new_purchase_date_max', 'new_purchase_date_min', 'new_card_id_size']]


# Apply Light GBM Modelling Technique with Repeated K Folds enumerated on training set and Target 

# In[ ]:


folds = RepeatedKFold(n_splits=5, n_repeats=2, random_state=4950)
oof_2 = np.zeros(len(train))
predictions_2 = np.zeros(len(test))
feature_importance_df_2 = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)):
    
    print("fold {}".format(fold_))
    trn_data = lgb.Dataset(train.iloc[trn_idx][features], label=target.iloc[trn_idx])
    val_data = lgb.Dataset(train.iloc[val_idx][features], label=target.iloc[val_idx])
    
    param ={
                'task': 'train',
                'boosting': 'goss',
                'objective': 'regression',
                'metric': 'rmse',
                'learning_rate': 0.01,
                'subsample': 0.9855232997390695,
                'max_depth': 7,
                'top_rate': 0.9064148448434349,
                'num_leaves': 63,
                'min_child_weight': 41.9612869171337,
                'other_rate': 0.0721768246018207,
                'reg_alpha': 9.677537745007898,
                'colsample_bytree': 0.5665320670155495,
                'min_split_gain': 9.820197773625843,
                'reg_lambda': 8.2532317400459,
                'min_data_in_leaf': 21,
                'verbose': -1,
                'seed':int(2**fold_),
                'bagging_seed':int(2**fold_),
                'drop_seed':int(2**fold_)
                }
        
    num_round = 10000
    clf_r = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=-1, early_stopping_rounds = 200)
    oof_2[val_idx] = clf_r.predict(train.iloc[val_idx][features], num_iteration=clf_r.best_iteration)
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = features
    fold_importance_df["importance"] = clf_r.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df_2 = pd.concat([feature_importance_df_2, fold_importance_df], axis=0)
    
    predictions_2 += clf_r.predict(test[features], num_iteration=clf_r.best_iteration) / (5 * 2)

print("CV score: {:<8.5f}".format(mean_squared_error(oof_2, target)**0.5))


# Feature Importance - Repeated K Folds

# In[ ]:


cols = (feature_importance_df_2[["Feature", "importance"]]
        .groupby("Feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:1000].index)

best_features = feature_importance_df_2.loc[feature_importance_df_2.Feature.isin(cols)]

plt.figure(figsize=(14,25))
sns.barplot(x="importance",
            y="Feature",
            data=best_features.sort_values(by="importance",
                                           ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.savefig('lgbm_importances.png')


# Stacking the models together.

# In[ ]:


from sklearn.linear_model import BayesianRidge

train_stack = np.vstack([oof,oof_2]).transpose()
test_stack = np.vstack([predictions, predictions_2]).transpose()

folds_stack = RepeatedKFold(n_splits=5, n_repeats=1, random_state=4590)
oof_stack = np.zeros(train_stack.shape[0])
predictions_3 = np.zeros(test_stack.shape[0])

for fold_, (trn_idx, val_idx) in enumerate(folds_stack.split(train_stack,target)):
    print("fold {}".format(fold_))
    trn_data, trn_y = train_stack[trn_idx], target.iloc[trn_idx].values
    val_data, val_y = train_stack[val_idx], target.iloc[val_idx].values
    
    clf_3 = BayesianRidge()
    clf_3.fit(trn_data, trn_y)
    
    oof_stack[val_idx] = clf_3.predict(val_data)
    predictions_3 += clf_3.predict(test_stack) / 5
    
np.sqrt(mean_squared_error(target.values, oof_stack))


# Final Predictions and Submission File.

# In[ ]:


sample_submission = pd.read_csv('../input/sample_submission.csv')
sample_submission['target'] = predictions_3
sample_submission.to_csv('submission_ashish_lgbm.csv', index=False)

