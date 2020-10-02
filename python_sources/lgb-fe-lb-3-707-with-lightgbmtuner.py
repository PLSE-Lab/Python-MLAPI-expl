#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install optuna')


# In[ ]:


import optuna
optuna.__version__


# In[ ]:


# coding: utf-8
import gc
import os
import time
import datetime
import pandas as pd
import numpy as np
import lightgbm as lgb_origin
import optuna.integration.lightgbm as lgb
from pathlib import Path
from tqdm import tqdm_notebook as tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold


# In[ ]:


input_path = Path("../input")
train_path = input_path / "train.csv"
test_path = input_path / "test.csv"
historical_tr_path = input_path / "historical_transactions.csv"
new_merchant_tr_path = input_path / "new_merchant_transactions.csv"


# # 1. Loading the data

# In[ ]:


train = pd.read_csv(train_path, parse_dates=["first_active_month"])
test = pd.read_csv(test_path, parse_dates=["first_active_month"])
new_merchant = pd.read_csv(new_merchant_tr_path, parse_dates=['purchase_date'])
ht = pd.read_csv(historical_tr_path, parse_dates=['purchase_date'])


# In[ ]:


for df in [train,test]:
    df['start_year'] = df['first_active_month'].dt.year
    df['start_month'] = df['first_active_month'].dt.month
    df['elapsed_time'] = (datetime.date(2018, 2, 1) - df['first_active_month'].dt.date).dt.days

ytrain = train['target']
del train['target']


# # Historical data

# In[ ]:


# binarize the categorical variables where it makes sense
ht['authorized_flag'] = ht['authorized_flag'].map({'Y':1, 'N':0})
ht['category_1'] = ht['category_1'].map({'Y':1, 'N':0})


# In[ ]:


ht['category_2x1'] = (ht['category_2'] == 1) + 0
ht['category_2x2'] = (ht['category_2'] == 2) + 0
ht['category_2x3'] = (ht['category_2'] == 3) + 0
ht['category_2x4'] = (ht['category_2'] == 4) + 0
ht['category_2x5'] = (ht['category_2'] == 5) + 0


# In[ ]:


ht['category_3A'] = (ht['category_3'].astype(str) == 'A') + 0
ht['category_3B'] = (ht['category_3'].astype(str) == 'B') + 0
ht['category_3C'] = (ht['category_3'].astype(str) == 'C') + 0


# In[ ]:


def aggregate_historical_transactions(history):
    
    history.loc[:, 'purchase_date'] = pd.DatetimeIndex(history['purchase_date']).                                      astype(np.int64) * 1e-9
    
    agg_func = {
        'authorized_flag': ['sum', 'mean'],
        'category_1': ['sum', 'mean'],
        'category_2': ['nunique'],
        'category_3A': ['sum'],
        'category_3B': ['sum'],
        'category_3C': ['sum'],
        'category_2x1': ['sum','mean'],
        'category_2x2': ['sum','mean'],
        'category_2x3': ['sum','mean'],
        'category_2x4': ['sum','mean'],
        'category_2x5': ['sum','mean'],        
        'city_id': ['nunique'],
        'installments': ['sum', 'median', 'max', 'min', 'std'],
        'merchant_category_id': ['nunique'],
        'merchant_id': ['nunique'],
        'month_lag': ['min', 'max'],
        'purchase_amount': ['sum', 'median', 'max', 'min', 'std'],
        'purchase_date': [np.ptp, 'max', 'min'],
        'state_id': ['nunique'],
        'subsector_id': ['nunique'],

        }
    agg_history = history.groupby(['card_id']).agg(agg_func)
    agg_history.columns = ['hist_' + '_'.join(col).strip() 
                           for col in agg_history.columns.values]
    agg_history.reset_index(inplace=True)
    
    df = (history.groupby('card_id')
          .size()
          .reset_index(name='hist_transactions_count'))
    
    agg_history = pd.merge(df, agg_history, on='card_id', how='left')
    
    return agg_history

history = aggregate_historical_transactions(ht)

del ht
gc.collect()


# #New data

# In[ ]:


new_merchant['authorized_flag'] = new_merchant['authorized_flag'].map({'Y':1, 'N':0})
new_merchant['category_1'] = new_merchant['category_1'].map({'Y':1, 'N':0})
new_merchant['category_3A'] = (new_merchant['category_3'].astype(str) == 'A') + 0
new_merchant['category_3B'] = (new_merchant['category_3'].astype(str) == 'B') + 0
new_merchant['category_3C'] = (new_merchant['category_3'].astype(str) == 'C') + 0

new_merchant['category_2x1'] = (new_merchant['category_2'] == 1) + 0
new_merchant['category_2x2'] = (new_merchant['category_2'] == 2) + 0
new_merchant['category_2x3'] = (new_merchant['category_2'] == 3) + 0
new_merchant['category_2x4'] = (new_merchant['category_2'] == 4) + 0
new_merchant['category_2x5'] = (new_merchant['category_2'] == 5) + 0


# In[ ]:


new_merchant['purchase_date'] = pd.DatetimeIndex(new_merchant['purchase_date']).                                      astype(np.int64) * 1e-9


# In[ ]:


def aggregate_new_transactions(new_trans):    
    
    
    agg_func = {
        'authorized_flag': ['sum', 'mean'],
        'category_1': ['sum', 'mean'],
        'category_2': ['nunique'],
        'category_3A': ['sum'],
        'category_3B': ['sum'],
        'category_3C': ['sum'],     
        'category_2x1': ['sum','mean'],
        'category_2x2': ['sum','mean'],
        'category_2x3': ['sum','mean'],
        'category_2x4': ['sum','mean'],
        'category_2x5': ['sum','mean'],        

        'city_id': ['nunique'],
        'installments': ['sum', 'median', 'max', 'min', 'std'],
        'merchant_category_id': ['nunique'],
        'merchant_id': ['nunique'],
        'month_lag': ['min', 'max'],
        'purchase_amount': ['sum', 'median', 'max', 'min', 'std'],
        'purchase_date': [np.ptp, 'max', 'min'],
        'state_id': ['nunique'],
        'subsector_id': ['nunique']        
        }
    agg_new_trans = new_trans.groupby(['card_id']).agg(agg_func)
    agg_new_trans.columns = ['new_' + '_'.join(col).strip() 
                           for col in agg_new_trans.columns.values]
    agg_new_trans.reset_index(inplace=True)
    
    df = (new_trans.groupby('card_id')
          .size()
          .reset_index(name='new_transactions_count'))
    
    agg_new_trans = pd.merge(df, agg_new_trans, on='card_id', how='left')
    
    return agg_new_trans

new_trans = aggregate_new_transactions(new_merchant)

del new_merchant


# # Combine

# In[ ]:


print(train.shape)
print(test.shape)

xtrain = pd.merge(train, new_trans, on='card_id', how='left')
xtest = pd.merge(test, new_trans, on='card_id', how='left')

del new_trans

print(xtrain.shape)
print(xtest.shape)

xtrain = pd.merge(xtrain, history, on='card_id', how='left')
xtest = pd.merge(xtest, history, on='card_id', how='left')

del history

print(xtrain.shape)
print(xtest.shape)


# In[ ]:


xtrain.head(3)


# In[ ]:


xtrain.drop('first_active_month', axis = 1, inplace = True)
xtest.drop('first_active_month', axis = 1, inplace = True)


# In[ ]:


categorical_feats = ['feature_1', 'feature_2', 'feature_3']

for col in categorical_feats:
    print(col)
    lbl = LabelEncoder()
    lbl.fit(list(xtrain[col].values.astype('str')) + list(xtest[col].values.astype('str')))
    xtrain[col] = lbl.transform(list(xtrain[col].values.astype('str')))
    xtest[col] = lbl.transform(list(xtest[col].values.astype('str')))


# In[ ]:


df_all = pd.concat([xtrain, xtest])
df_all = pd.get_dummies(df_all, columns=categorical_feats)

len_train = xtrain.shape[0]

xtrain = df_all[:len_train]
xtest = df_all[len_train:]


# # Model

# In[ ]:


# prepare for modeling
id_train = xtrain['card_id'].copy(); xtrain.drop('card_id', axis = 1, inplace = True)
id_test = xtest['card_id'].copy(); xtest.drop('card_id', axis = 1, inplace = True)


# In[ ]:


n_splits = 5
folds = KFold(n_splits=n_splits, shuffle=True, random_state=15)
oof = np.zeros(len(xtrain))
predictions = np.zeros(len(xtest))


# In[ ]:


start = time.time()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(xtrain.values, ytrain.values)):
    print('Fold {}/{}'.format(fold_ + 1, folds.n_splits))
    
    x0,y0 = xtrain.iloc[trn_idx], ytrain[trn_idx]
    x1,y1 = xtrain.iloc[val_idx], ytrain[val_idx]
    
    trn_data = lgb.Dataset(x0, label= y0)
    val_data = lgb.Dataset(x1, label= y1)
    
     # LightGBMTuner
    # Reference: https://gist.github.com/smly/367c53e855cdaeea35736f32876b7416
    best_params = {}
    tuning_history = []

    params = {
                'objective': 'regression',
                'metric': 'rmse',
            }
    
    num_round = 10000
    lgb.train(
        params,
        trn_data,
        num_boost_round=num_round,
        valid_sets=[trn_data, val_data],
        early_stopping_rounds=100,
        verbose_eval=500,
        best_params=best_params,
        tuning_history=tuning_history)
    
    pd.DataFrame(tuning_history).to_csv('./tuning_history_{}.csv'.format(fold_ + 1))
    best_params['learning_rate'] = 0.05
    
    # origin LightGBM Model
    clf = lgb_origin.train(
                    best_params,
                    trn_data,
                    num_boost_round=num_round * 2,
                    valid_names=['train', 'valid'],
                    valid_sets=[trn_data, val_data],
                    verbose_eval=1000,
                    early_stopping_rounds=1000)
    
    oof[val_idx] = clf.predict(xtrain.iloc[val_idx], num_iteration=clf.best_iteration)
    
    predictions += clf.predict(xtest, num_iteration=clf.best_iteration) / folds.n_splits
    print("Fold {} CV score: {:<8.5f}".format(fold_ + 1, mean_squared_error(ytrain[val_idx],
                                                                            oof[val_idx])**0.5))
print("CV score: {:<8.5f}".format(mean_squared_error(oof, ytrain)**0.5))


# In[ ]:


xsub = pd.DataFrame()
xsub['card_id']  = id_test
xsub['target'] = predictions
xsub.to_csv('sub_lgb.csv', index = False)


# In[ ]:




