#!/usr/bin/env python
# coding: utf-8

# In[ ]:




# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold,KFold
from bayes_opt import BayesianOptimization
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, confusion_matrix, accuracy_score, roc_auc_score, f1_score, roc_curve, auc,precision_recall_curve
from sklearn import metrics
from sklearn import preprocessing
import itertools
from scipy import interp
# Lgbm
import lightgbm as lgb
import seaborn as sns


import matplotlib.pylab as plt


import os
import gc

import datetime

print(os.listdir("../input"))

import matplotlib.pyplot as plt
import os


import gc

# Any results you write to the current directory are saved as output.


# In[ ]:


def reduce_mem_usage_sd(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage(deep=True).sum() / 1024 ** 2 # just added 
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
    end_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    percent = 100 * (start_mem - end_mem) / start_mem
    print('Mem. usage decreased from {:5.2f} Mb to {:5.2f} Mb ({:.1f}% reduction)'.format(start_mem, end_mem, percent))
    return df


# In[ ]:



def values_normalization(dt_df, periods, columns):
    for period in periods:
        for col in columns:
            new_col = col +'_'+ period
            dt_df[col] = dt_df[col].astype(float)  

            temp_min = dt_df.groupby([period])[col].agg(['min']).reset_index()
            temp_min.index = temp_min[period].values
            temp_min = temp_min['min'].to_dict()

            temp_max = dt_df.groupby([period])[col].agg(['max']).reset_index()
            temp_max.index = temp_max[period].values
            temp_max = temp_max['max'].to_dict()

            temp_mean = dt_df.groupby([period])[col].agg(['mean']).reset_index()
            temp_mean.index = temp_mean[period].values
            temp_mean = temp_mean['mean'].to_dict()

            temp_std = dt_df.groupby([period])[col].agg(['std']).reset_index()
            temp_std.index = temp_std[period].values
            temp_std = temp_std['std'].to_dict()

            dt_df['temp_min'] = dt_df[period].map(temp_min)
            dt_df['temp_max'] = dt_df[period].map(temp_max)
            dt_df['temp_mean'] = dt_df[period].map(temp_mean)
            dt_df['temp_std'] = dt_df[period].map(temp_std)

            dt_df[new_col+'_min_max'] = (dt_df[col]-dt_df['temp_min'])/(dt_df['temp_max']-dt_df['temp_min'])
            dt_df[new_col+'_std_score'] = (dt_df[col]-dt_df['temp_mean'])/(dt_df['temp_std'])
            del dt_df['temp_min'],dt_df['temp_max'],dt_df['temp_mean'],dt_df['temp_std']
    return dt_df


# In[ ]:


train_transaction = reduce_mem_usage_sd(pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv', index_col='TransactionID'))
train_identity = reduce_mem_usage_sd(pd.read_csv('../input/ieee-fraud-detection/train_identity.csv', index_col='TransactionID'))


# In[ ]:


df_train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)


# In[ ]:


y = df_train.iloc[int(len(df_train)*0.6):]['isFraud'].values


# In[ ]:


a = len(df_train.iloc[int(len(df_train)*0.6):])


# In[ ]:


df_test = df_train.iloc[int(len(df_train)*0.6):]
df_train = df_train.iloc[:int(len(df_train)*0.6)]


# In[ ]:


del train_transaction, train_identity
gc.collect()


# In[ ]:


comb = pd.concat([df_train,df_test],axis=0,sort=True)
del df_train, df_test
gc.collect()


# In[ ]:


rm_cols = ['TransactionID', 'TransactionDT', 'isFraud']


# In[ ]:


features = []
features = [col for col in list(comb) if col not in rm_cols]


# In[ ]:


for f in features:
    if(str(comb[f].dtype)!="object" and str(comb[f].dtype) !="category") :
        comb[f] = comb[f].replace(np.nan,-999)


# In[ ]:


for f in features:
    if  (str(comb[f].dtype)=="object" or str(comb[f].dtype)=="category") :  
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(comb[f].values) )
        comb[f] = lbl.transform(list(comb[f].values))
comb = comb.reset_index()


# In[ ]:


START_DATE = datetime.datetime.strptime('2017-11-30', '%Y-%m-%d')


# In[ ]:


comb['DT'] = comb['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds = x)))
comb['DT_M'] = ((comb['DT'].dt.year-2017)*12 + comb['DT'].dt.month).astype(np.int8)
comb['DT_W'] = ((comb['DT'].dt.year-2017)*52 + comb['DT'].dt.weekofyear).astype(np.int8)
comb['DT_D'] = ((comb['DT'].dt.year-2017)*365 + comb['DT'].dt.dayofyear).astype(np.int16)


# In[ ]:


periods = ['DT_D','DT_W','DT_M']
i_cols = ['TransactionAmt']
comb = values_normalization(comb, periods, i_cols)


# In[ ]:


comb.drop(['DT','DT_D','DT_W','DT_M'], axis =1,inplace = True)


# In[ ]:


gc.collect()


# In[ ]:


target = 'isFraud'


# In[ ]:


lgb_params = {'num_leaves': 491,
          'min_child_weight': 0.03454472573214212,
          'feature_fraction': 0.3797454081646243,
          'bagging_fraction': 0.4181193142567742,
          'min_data_in_leaf': 106,
          'objective': 'binary',
          'max_depth': -1,
          'learning_rate': 0.006883242363721497,
          "boosting_type": "gbdt",
          "bagging_seed": 11,
          "metric": 'auc',
          "verbosity": -1,
          'reg_alpha': 0.3899927210061127,
          'reg_lambda': 0.6485237330340494,
          'random_state': 47,
          
    "seed": 2019
             }


# In[ ]:


rm_cols = ['TransactionID', 'TransactionDT', 'isFraud']
     


# In[ ]:


features = []
features = [col for col in list(comb) if col not in rm_cols]


# In[ ]:


plt.rcParams["axes.grid"] = True


#skf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=42)


gc.collect()

i = 0


trn_data = lgb.Dataset(comb.iloc[:int(len(comb)*0.6)][features].values,
                                   label=comb.iloc[:int(len(comb)*0.6)][target].values,feature_name=features 
                                   )
val_data = lgb.Dataset(comb.iloc[int(len(comb)*0.6):][features].values,
                                   label=y,feature_name=features 
                                   )   
    
clf = lgb.train(lgb_params, trn_data, num_boost_round = 1000, valid_sets = [trn_data, val_data], verbose_eval = 50)


# In[ ]:


# Features imp
fold_importance_df = pd.DataFrame()
fold_importance_df["Feature"] = features
fold_importance_df["importance"] = clf.feature_importance()
#fold_importance_df["fold"] = nfold + 1
feature_importance_df = pd.concat([fold_importance_df, fold_importance_df], axis=0)
    
gc.collect()


# In[ ]:


plt.style.use('dark_background')
cols = (feature_importance_df[["Feature", "importance"]]
    .groupby("Feature")
    .mean()
    .sort_values(by="importance", ascending=False)[:40].index)
best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]

plt.figure(figsize=(10,10))
sns.barplot(x="importance", y="Feature", data=best_features.sort_values(by="importance",ascending=False),
        edgecolor=('white'), linewidth=2, palette="rocket")
plt.title('LGB Features importance (averaged/folds)', fontsize=18)
plt.tight_layout()


# In[ ]:


del trn_data,val_data
gc.collect()


# # D Features Magic :

# In[ ]:


df_train = comb[:-a]
df_test = comb[-a:]
del comb
gc.collect()


# In[ ]:


for i in ['D3' , 'D4', 'D6', 'D7', 'D8','D10','D11','D12','D13','D14','D15'] :
    
    z = (df_train['TransactionDT'] - (df_train[i].astype('float64')*60*60)).min()
    df_train['TransactionDT_lag'] = (df_train['TransactionDT'] - z)/(60*60*24)
    df_test['TransactionDT_lag'] = (df_test['TransactionDT'] - z)/(60*60*24)
    df_train[i] = (df_train[i] - df_train['TransactionDT_lag']).astype('int64')
    df_test[i] = (df_test[i] - df_test['TransactionDT_lag']).astype('int64')


# In[ ]:


df_train.drop('TransactionDT_lag', axis = 1 , inplace = True)
df_test.drop('TransactionDT_lag', axis = 1 , inplace = True)
gc.collect()


# In[ ]:


comb = pd.concat([df_train,df_test],axis=0,sort=True)
del df_train, df_test
gc.collect()


# In[ ]:


plt.rcParams["axes.grid"] = True


#skf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=42)


gc.collect()

i = 0


trn_data = lgb.Dataset(comb.iloc[:int(len(comb)*0.6)][features].values,
                                   label=comb.iloc[:int(len(comb)*0.6)][target].values,feature_name=features 
                                   )
val_data = lgb.Dataset(comb.iloc[int(len(comb)*0.6):][features].values,
                                   label=y,feature_name=features 
                                   )   
    
clf = lgb.train(lgb_params, trn_data, num_boost_round = 1000, valid_sets = [trn_data, val_data], verbose_eval = 50)


# In[ ]:


fold_importance_df = pd.DataFrame()
fold_importance_df["Feature"] = features
fold_importance_df["importance"] = clf.feature_importance()
#fold_importance_df["fold"] = nfold + 1
feature_importance_df = pd.concat([fold_importance_df, fold_importance_df], axis=0)
    
gc.collect()


# In[ ]:


plt.style.use('dark_background')
cols = (feature_importance_df[["Feature", "importance"]]
    .groupby("Feature")
    .mean()
    .sort_values(by="importance", ascending=False)[:40].index)
best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]

plt.figure(figsize=(10,10))
sns.barplot(x="importance", y="Feature", data=best_features.sort_values(by="importance",ascending=False),
        edgecolor=('white'), linewidth=2, palette="rocket")
plt.title('LGB Features importance (averaged/folds)', fontsize=18)
plt.tight_layout()

