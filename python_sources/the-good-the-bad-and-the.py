#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn import *
from matplotlib import pyplot
get_ipython().run_line_magic('matplotlib', 'inline')
from catboost import CatBoostRegressor
import lightgbm as lgb
import xgboost as xgb
import datetime, time

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

def features1(df):
    df['first_active_month'] = pd.to_datetime(df['first_active_month'])
    df['first_active_year'] = df['first_active_month'].dt.year
    df['first_active_quarter'] = df['first_active_month'].dt.quarter
    df['elapsed_time'] = (datetime.datetime.today() - df['first_active_month']).dt.days
    df['first_active_month'] = df['first_active_month'].dt.month
    df['days_feature1'] = df['elapsed_time'] * df['feature_1']
    df['days_feature2'] = df['elapsed_time'] * df['feature_2']
    df['days_feature3'] = df['elapsed_time'] * df['feature_3']
    df['days_feature1_ratio'] = df['feature_1'] / df['elapsed_time']
    df['days_feature2_ratio'] = df['feature_2'] / df['elapsed_time']
    df['days_feature3_ratio'] = df['feature_3'] / df['elapsed_time']
    return df

train = features1(train)
test = features1(test)
#data_dict = pd.read_excel('../input/Data_Dictionary.xlsx')
train.shape, test.shape


# In[ ]:


data_nmt = pd.read_csv('../input/new_merchant_transactions.csv') #(1_963_031, 14) card_id merchant_id
data_nmt.purchase_date = pd.to_datetime(data_nmt.purchase_date)
data_nmt['year'] = data_nmt.purchase_date.dt.year
data_nmt['month'] = data_nmt.purchase_date.dt.month
data_nmt['category_1'] = data_nmt['category_1'].map({'Y':1, 'N':0}).astype(np.int8)
data_nmt['category_3'] = data_nmt['category_3'].map({'A':2, 'B':1, 'C':0, np.nan: -1}).astype(np.int8)
data_nmt.drop(columns=['authorized_flag', 'purchase_date'], inplace=True)
data_nmt.shape


# In[ ]:


data_hist = pd.read_csv('../input/historical_transactions.csv') #(29_112_361, 14) card_id merchant_id
data_hist['category_1'] = data_hist['category_1'].map({'Y':1, 'N':0}).astype(np.int8)
data_hist['category_3'] = data_hist['category_3'].map({'A':2, 'B':1, 'C':0, np.nan: -1}).astype(np.int8)
data_hist = data_hist[data_hist['authorized_flag']=='Y']
data_hist.purchase_date = pd.to_datetime(data_hist.purchase_date)
data_hist['year'] = data_hist.purchase_date.dt.year
data_hist['month'] = data_hist.purchase_date.dt.month
data_hist.drop(columns=['authorized_flag', 'purchase_date'], inplace=True)
print(data_hist.shape)


# In[ ]:


data_hist = pd.concat((data_hist, data_nmt))
data_hist['price'] = data_hist['purchase_amount'] / data_hist['installments']
print(data_hist.shape)
del data_nmt


# In[ ]:


for c in ['category_1', 'category_2', 'category_3', 'year', 'month']:
    du = pd.get_dummies(data_hist[c], prefix=c)
    du['card_id'] = data_hist['card_id']
    du = du.groupby(['card_id']).sum()
    train = pd.merge(train, du, how='left', on=['card_id'])
    test = pd.merge(test, du, how='left', on=['card_id'])
    data_hist.drop(columns=[c], inplace=True)
train.shape, test.shape, data_hist.shape


# In[ ]:


data_hist_group = data_hist.groupby(['card_id']).agg({
        'city_id': ['nunique'],
        'merchant_category_id': ['nunique'],
        'merchant_id': ['nunique'],
        'month_lag': ['min', 'max'],
        'purchase_amount': ['sum', 'median', 'max', 'min', 'std'],
        'installments': ['sum', 'median', 'max', 'min', 'std'],
        'price': ['sum', 'median', 'max', 'min', 'std'],
        'state_id': ['nunique'],
        'subsector_id': ['nunique']
        }).reset_index()
data_hist_group.columns = [''.join(c) for c in data_hist_group.columns]

train = pd.merge(train, data_hist_group, on='card_id', how='left').fillna(-1)
test = pd.merge(test, data_hist_group, on='card_id', how='left').fillna(-1)
del data_hist_group
data_hist.drop(columns=['city_id', 'merchant_category_id', 'month_lag', 'purchase_amount', 'state_id', 'subsector_id'], inplace=True)
train.shape, test.shape, data_hist.shape


# In[ ]:


merchants = pd.read_csv('../input/merchants.csv') #(334_696, 22) merchant_id
merchants['category_1'] = merchants['category_1'].map({'Y':1, 'N':0}).astype(np.int8)
merchants['category_4'] = merchants['category_4'].map({'Y':1, 'N':0}).astype(np.int8)
merchants['most_recent_sales_range'] = merchants['most_recent_sales_range'].map({'E':4, 'D':3, 'C':2, 'B':1, 'A':0}).astype(np.int8)
merchants['most_recent_purchases_range'] = merchants['most_recent_purchases_range'].map({'E':4, 'D':3, 'C':2, 'B':1, 'A':0}).astype(np.int8)
merchants.drop(columns=['merchant_category_id', 'subsector_id', 'city_id', 'state_id'], inplace=True)
data_hist = pd.merge(data_hist, merchants, how='left', on=['merchant_id'])
del merchants
data_hist.drop(columns=['merchant_id'], inplace=True)
data_hist.shape


# In[ ]:


for c in ['category_1', 'category_2', 'category_4', 'most_recent_sales_range', 'most_recent_purchases_range', 'active_months_lag3', 'active_months_lag6', 'active_months_lag12']:
    du = pd.get_dummies(data_hist[c], prefix='merchants_'+c)
    du['card_id'] = data_hist['card_id']
    du = du.groupby(['card_id']).sum()
    train = pd.merge(train, du, how='left', on=['card_id'])
    test = pd.merge(test, du, how='left', on=['card_id'])
    data_hist.drop(columns=[c], inplace=True)
train.shape, test.shape, data_hist.shape


# In[ ]:


data_hist_group = data_hist.groupby(['card_id']).agg({
        'merchant_group_id': ['nunique'],
        'numerical_1': ['min', 'max', 'mean'],
        'numerical_2': ['min', 'max', 'mean'],
        'avg_sales_lag3': ['min', 'max', 'mean'],
        'avg_purchases_lag3': ['min', 'max', 'mean'],
        'avg_sales_lag6': ['min', 'max', 'mean'],
        'avg_purchases_lag6': ['min', 'max', 'mean'],
        'avg_sales_lag12': ['min', 'max', 'mean'],
        'avg_purchases_lag12': ['min', 'max', 'mean']
        }).reset_index()
data_hist_group.columns = [''.join(c) for c in data_hist_group.columns]

train = pd.merge(train, data_hist_group, on='card_id', how='left').fillna(-1)
test = pd.merge(test, data_hist_group, on='card_id', how='left').fillna(-1)
del data_hist_group
del data_hist
train.shape, test.shape


# In[ ]:


col = [c for c in train.columns if c not in ['card_id', 'target', 'first_active_month', 'first_active_year']]
x1, x2, y1, y2 = model_selection.train_test_split(train[col], train['target'], test_size=0.2, random_state=3)
params = {'learning_rate': 0.02,'max_depth': 10, 'num_leaves': 90, 'application': 'regression', 'boosting': 'gbdt', 'metric': 'rmse', 'seed': 3}
model = lgb.train(params, lgb.Dataset(x1, label=y1), 2500, lgb.Dataset(x2, label=y2), verbose_eval=500, early_stopping_rounds=200)
test['target'] = model.predict(test[col], num_iteration=model.best_iteration)
test[['card_id', 'target']].to_csv('submission.csv', index=False)
lgb.plot_importance(model, importance_type='split', max_num_features=20)


# In[ ]:




