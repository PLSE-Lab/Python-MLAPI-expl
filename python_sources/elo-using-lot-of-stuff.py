#!/usr/bin/env python
# coding: utf-8

#  Packages
#  First, let's load a few useful Python packages. This section will keep growing in subsequent versions of this EDA.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn import model_selection, preprocessing, metrics
import warnings
import datetime
import time
import gc
from tqdm import tqdm
from scipy.stats import describe
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
import xgboost as xgb
warnings.filterwarnings("ignore")

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


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


#Loading Train and Test Data
df_train = pd.read_csv("../input/train.csv", parse_dates=["first_active_month"])
df_test = pd.read_csv("../input/test.csv", parse_dates=["first_active_month"])
print("{} observations and {} features in train set.".format(df_train.shape[0],df_train.shape[1]))
print("{} observations and {} features in test set.".format(df_test.shape[0],df_test.shape[1]))


# In[ ]:


df_train.head()


# In[ ]:


df_test.head()


# In[ ]:


df_train.target.describe()


# In[ ]:


df_train["month"] = df_train["first_active_month"].dt.month
df_test["month"] = df_test["first_active_month"].dt.month
df_train["year"] = df_train["first_active_month"].dt.year
df_test["year"] = df_test["first_active_month"].dt.year
df_train['elapsed_time'] = (datetime.date(2018, 2, 1) - df_train['first_active_month'].dt.date).dt.days
df_test['elapsed_time'] = (datetime.date(2018, 2, 1) - df_test['first_active_month'].dt.date).dt.days
df_train.head()


# In[ ]:


# df_train = pd.get_dummies(df_train, columns=['feature_1', 'feature_2'])
# df_test = pd.get_dummies(df_test, columns=['feature_1', 'feature_2'])
# df_train.head()


# In[ ]:


# df_test.head()


# In[ ]:


df_hist_trans = pd.read_csv('../input/historical_transactions.csv')
df_new_merchant_trans = pd.read_csv('../input/new_merchant_transactions.csv')


# In[ ]:


# df_hist_trans.head()


# In[ ]:


# df_new_merchant_trans.head()


# In[ ]:


def binarize(df):
    for col in ['authorized_flag', 'category_1']:
        df[col] = df[col].map({'Y':1, 'N':0})
    return df


# In[ ]:


df_hist_trans = binarize(df_hist_trans)
df_new_merchant_trans = binarize(df_new_merchant_trans)


# In[ ]:


df_hist_trans['purchase_date'] = pd.to_datetime(df_hist_trans['purchase_date'])
df_hist_trans['month_diff'] = ((datetime.datetime.today() - df_hist_trans['purchase_date']).dt.days)//30
df_hist_trans['month_diff'] += df_hist_trans['month_lag']

df_new_merchant_trans['purchase_date'] = pd.to_datetime(df_new_merchant_trans['purchase_date'])
df_new_merchant_trans['month_diff'] = ((datetime.datetime.today() - df_new_merchant_trans['purchase_date']).dt.days)//30
df_new_merchant_trans['month_diff'] += df_new_merchant_trans['month_lag']


# In[ ]:


df_hist_trans[:5]


# In[ ]:


df_hist_trans = pd.get_dummies(df_hist_trans, columns=['category_2', 'category_3'])
df_new_merchant_trans = pd.get_dummies(df_new_merchant_trans, columns=['category_2', 'category_3'])


# In[ ]:


df_hist_trans = reduce_mem_usage(df_hist_trans)
df_new_merchant_trans = reduce_mem_usage(df_new_merchant_trans)

agg_fun = {'authorized_flag': ['mean']}
auth_mean = df_hist_trans.groupby(['card_id']).agg(agg_fun)
auth_mean.columns = ['_'.join(col).strip() for col in auth_mean.columns.values]
auth_mean.reset_index(inplace=True)

authorized_transactions = df_hist_trans[df_hist_trans['authorized_flag'] == 1]
historical_transactions = df_hist_trans[df_hist_trans['authorized_flag'] == 0]


# In[ ]:


historical_transactions.head()


# In[ ]:


historical_transactions['purchase_month'] = historical_transactions['purchase_date'].dt.month
authorized_transactions['purchase_month'] = authorized_transactions['purchase_date'].dt.month
df_new_merchant_trans['purchase_month'] = df_new_merchant_trans['purchase_date'].dt.month


# In[ ]:


def aggregate_transactions(history):
    
    history.loc[:, 'purchase_date'] = pd.DatetimeIndex(history['purchase_date']).                                      astype(np.int64) * 1e-9
    
    agg_func = {
    'category_1': ['sum', 'mean'],
    'category_2_1.0': ['mean'],
    'category_2_2.0': ['mean'],
    'category_2_3.0': ['mean'],
    'category_2_4.0': ['mean'],
    'category_2_5.0': ['mean'],
    'category_3_A': ['mean'],
    'category_3_B': ['mean'],
    'category_3_C': ['mean'],
    'merchant_id': ['nunique'],
    'merchant_category_id': ['nunique'],
    'state_id': ['nunique'],
    'city_id': ['nunique'],
    'subsector_id': ['nunique'],
    'purchase_amount': ['sum', 'mean', 'max', 'min', 'std'],
    'installments': ['sum', 'mean', 'max', 'min', 'std'],
    'purchase_month': ['mean', 'max', 'min', 'std'],
    'purchase_date': [np.ptp, 'min', 'max'],
    'month_lag': ['mean', 'max', 'min', 'std'],
    'month_diff': ['mean']
    }
    
    agg_history = history.groupby(['card_id']).agg(agg_func)
    agg_history.columns = ['_'.join(col).strip() for col in agg_history.columns.values]
    agg_history.reset_index(inplace=True)
    
    df = (history.groupby('card_id')
          .size()
          .reset_index(name='transactions_count'))
    
    agg_history = pd.merge(df, agg_history, on='card_id', how='left')
    
    return agg_history


# In[ ]:


history = aggregate_transactions(historical_transactions)
history.columns = ['hist_' + c if c != 'card_id' else c for c in history.columns]
history[:5]


# In[ ]:


authorized = aggregate_transactions(authorized_transactions)
authorized.columns = ['auth_' + c if c != 'card_id' else c for c in authorized.columns]
authorized[:5]


# In[ ]:


new = aggregate_transactions(df_new_merchant_trans)
new.columns = ['new_' + c if c != 'card_id' else c for c in new.columns]
new[:5]


# In[ ]:


def aggregate_per_month(history):
    grouped = history.groupby(['card_id', 'month_lag'])

    agg_func = {
            'purchase_amount': ['count', 'sum', 'mean', 'min', 'max', 'std'],
            'installments': ['count', 'sum', 'mean', 'min', 'max', 'std'],
            }

    intermediate_group = grouped.agg(agg_func)
    intermediate_group.columns = ['_'.join(col).strip() for col in intermediate_group.columns.values]
    intermediate_group.reset_index(inplace=True)

    final_group = intermediate_group.groupby('card_id').agg(['mean', 'std'])
    final_group.columns = ['_'.join(col).strip() for col in final_group.columns.values]
    final_group.reset_index(inplace=True)
    
    return final_group
#___________________________________________________________
final_group =  aggregate_per_month(authorized_transactions) 
final_group[:10]


# In[ ]:


def successive_aggregates(df, field1, field2):
    t = df.groupby(['card_id', field1])[field2].mean()
    u = pd.DataFrame(t).reset_index().groupby('card_id')[field2].agg(['mean', 'min', 'max', 'std'])
    u.columns = [field1 + '_' + field2 + '_' + col for col in u.columns.values]
    u.reset_index(inplace=True)
    return u


# In[ ]:


additional_fields = successive_aggregates(df_new_merchant_trans, 'category_1', 'purchase_amount')
additional_fields = additional_fields.merge(successive_aggregates(df_new_merchant_trans, 'installments', 'purchase_amount'),
                                            on = 'card_id', how='left')
additional_fields = additional_fields.merge(successive_aggregates(df_new_merchant_trans, 'city_id', 'purchase_amount'),
                                            on = 'card_id', how='left')
additional_fields = additional_fields.merge(successive_aggregates(df_new_merchant_trans, 'category_1', 'installments'),
                                            on = 'card_id', how='left')


# In[ ]:


# We now train the model with the features we previously defined. A first step consists in merging all the dataframes:


# In[ ]:


df_train = pd.merge(df_train, history, on='card_id', how='left')
df_test = pd.merge(df_test, history, on='card_id', how='left')

df_train = pd.merge(df_train, authorized, on='card_id', how='left')
df_test = pd.merge(df_test, authorized, on='card_id', how='left')

df_train = pd.merge(df_train, new, on='card_id', how='left')
df_test = pd.merge(df_test, new, on='card_id', how='left')

df_train = pd.merge(df_train, final_group, on='card_id', how='left')
df_test = pd.merge(df_test, final_group, on='card_id', how='left')

df_train = pd.merge(df_train, auth_mean, on='card_id', how='left')
df_test = pd.merge(df_test, auth_mean, on='card_id', how='left')

df_train = pd.merge(df_train, additional_fields, on='card_id', how='left')
df_test = pd.merge(df_test, additional_fields, on='card_id', how='left')


# In[ ]:


target = df_train['target']
drops = ['card_id', 'first_active_month', 'target']
use_cols = [c for c in df_train.columns if c not in drops]
features = list(df_train[use_cols].columns)
df_train[features].head()


# In[ ]:


param = {'num_leaves': 50,
         'min_data_in_leaf': 30, 
         'objective':'regression',
         'max_depth': -1,
         'learning_rate': 0.005,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9,
         "bagging_seed": 11,
         "metric": 'rmse',
         "lambda_l1": 0.1,
         "verbosity": -1}

folds = KFold(n_splits=5, shuffle=True, random_state=15)
oof = np.zeros(len(df_train))
predictions = np.zeros(len(df_test))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_train.values, target.values)):
    print('-')
    print("Fold {}".format(fold_ + 1))
    trn_data = lgb.Dataset(df_train.iloc[trn_idx][features], label=target.iloc[trn_idx])
    val_data = lgb.Dataset(df_train.iloc[val_idx][features], label=target.iloc[val_idx])

    num_round = 10000
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=100, early_stopping_rounds=100)
    oof[val_idx] = clf.predict(df_train.iloc[val_idx][features], num_iteration=clf.best_iteration)
    predictions += clf.predict(df_test[features], num_iteration=clf.best_iteration) / folds.n_splits


# In[ ]:


np.save('oof', oof)
np.save('predictions', predictions)
print("CV score: {:<8.5f}".format(mean_squared_error(oof, target)**0.5))


# In[ ]:


from sklearn.metrics import mean_squared_error
val_score = np.sqrt(mean_squared_error(target, oof))
val_score


# In[ ]:


fig, ax = plt.subplots(figsize=(12,10))
lgb.plot_importance(clf, max_num_features=30, height=0.5, ax=ax, title='Feature importance', xlabel='Feature importance', ylabel='Features')
plt.show()


# In[ ]:


sorted(list(zip(clf.feature_importance(), features)), reverse=True)


# In[ ]:


df_sub = pd.DataFrame({"card_id":df_test["card_id"].values})
df_sub["target"] = predictions
df_sub.to_csv("sub_val_{}.csv".format(val_score), index=False)


# In[ ]:


df_sub.head()


# In[ ]:


df_sub.to_csv("submission.csv", index=False)


# In[ ]:





# In[ ]:




