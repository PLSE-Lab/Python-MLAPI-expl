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

import os
print(os.listdir("../input"))

import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime as dt

import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().run_cell_magic('time', '', "df_train = pd.read_csv('../input/train.csv', low_memory=True, nrows=500000)")


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


df_train = reduce_mem_usage(df_train)


# ### I think the file can be read in more efficiently, but for now let's go with this and try to investigate some of the features. Our target variable is HasDetections. 

# In[ ]:


df_train.info()


# In[ ]:


df_train.isna().sum()


# ### A lot of NaNs which we shall have to tackle in due time

# ### From the Data description::
# > Each row in this dataset corresponds to a machine, uniquely identified by a MachineIdentifier. HasDetections is the ground truth and indicates that Malware was detected on the machine.

# ### Explore the distribution of the target variable HasDetections

# In[ ]:


plt.figure(figsize=(5,5))
ax = sns.countplot(df_train.HasDetections)
plt.xlabel('HasDetections');


# In[ ]:


df_train['HasDetections'].value_counts()


# ### That is an extremely balanced dataset. Doubt we'll need SMOTE for this one. 

# In[ ]:


df_train.shape # So. Many. Features.


# ### First lazy effort we will just retain the columns with numeric datatype and no nans

# In[ ]:


target = df_train['HasDetections']
df_train.drop(['HasDetections'],axis=1,inplace=True)


# In[ ]:


df_train = df_train.select_dtypes(['int8','int16','float16','float32'])
df_train.shape


# In[ ]:


df_train.isna().sum() # None of these have nans, let's go with these 15 features for now


# ### Let's explore these

# In[ ]:


# What % of these features are nans, only retain the ones with < 50% nans
cols_to_keep = []
for col in df_train.columns:
    if ((df_train[col].isna().sum()/len(df_train)))<0.5:
        print(col, (df_train[col].isna().sum()/len(df_train)))
        cols_to_keep.append(col)
        print('+++++++')


# In[ ]:


df_train = df_train[cols_to_keep]
df_train.shape


# ### So, while a number of these features have many unique values, from the names itself, we see that these are also all categorical variables. In keeping with the general lazy theme, I will only retain the variables with < 20 unique values

# In[ ]:


# Now just for a first effort
cols_to_keep_2 = []
for col in df_train.columns:
    if len(df_train[col].value_counts())<20:
        print(col, len(df_train[col].value_counts()))
        cols_to_keep_2.append(col)
        print()


# In[ ]:


df_train = df_train[cols_to_keep_2]
df_train.shape


# In[ ]:


cols_for_test = df_train.columns.tolist()


# ## We can work with this, let's do the whole get_dummies stuff and get a basic LightGBM going

# In[ ]:


df_train = pd.get_dummies(df_train,columns=df_train.columns.tolist())
df_train.head()


# In[ ]:


df_train.shape


# ### Let's follow the same steps for the test set. Just read in numeric columns.

# In[ ]:


get_ipython().run_cell_magic('time', '', "df_test = pd.read_csv('../input/test.csv',low_memory=True, usecols=cols_for_test)")


# In[ ]:


df_test.head()


# In[ ]:


df_test = reduce_mem_usage(df_test)


# In[ ]:


df_test = df_test.select_dtypes(['int8','int16','float16','float32'])
cols_to_keep = []
for col in df_test.columns:
    if (((df_test[col].isna().sum()/len(df_test)))<0.5) and (len(df_test[col].value_counts())<20):
        cols_to_keep.append(col)
        
df_test = df_test[cols_to_keep]


# In[ ]:


df_test = pd.get_dummies(df_test,columns=df_test.columns.tolist())
df_test.shape


# In[ ]:


# Let's drop the columns (other than the target that are not in common between the train and test sets)
set1 = set(df_train.columns.tolist())
set2 = set(df_test.columns.tolist())

set1 - set2


# In[ ]:


set2 - set1


# In[ ]:


cols_in_common = list(set1 & set2)
df_train = df_train[cols_in_common]
df_test = df_test[cols_in_common]

print(df_train.shape, df_test.shape)


# In[ ]:


df_train = reduce_mem_usage(df_train)
df_test = reduce_mem_usage(df_test)


# ### Basic LightGBM (no parameter tuning yet, I just used what was working well for me on the ELO competition)

# In[ ]:


features = list(df_train.columns)

lgb_params = {'num_leaves': 100,
         'min_data_in_leaf': 30, 
         'objective':'regression',
         'max_depth': -1,
         'learning_rate': 0.01,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9,
         "bagging_seed": 42,
         "metric": 'auc',
         "lambda_l1": 0.1,
         "verbosity": -1}

folds = KFold(n_splits=10, shuffle=True, random_state=42)
oof_lgb = np.zeros(len(df_train))
predictions_lgb = np.zeros(len(df_test))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_train.values, target.values)):
    print('-')
    print("Fold {}".format(fold_ + 1))
    trn_data = lgb.Dataset(df_train.iloc[trn_idx][features], label=target.iloc[trn_idx])
    val_data = lgb.Dataset(df_train.iloc[val_idx][features], label=target.iloc[val_idx])

    num_round = 10000
    clf = lgb.train(lgb_params, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=100, early_stopping_rounds=150)
    oof_lgb[val_idx] = clf.predict(df_train.iloc[val_idx][features], num_iteration=clf.best_iteration)
    predictions_lgb += clf.predict(df_test[features], num_iteration=clf.best_iteration) / folds.n_splits


# In[ ]:


print("CV score: {}".format(roc_auc_score(target, oof_lgb)))


# In[ ]:


df_sub = pd.read_csv('../input/sample_submission.csv',low_memory=True)
df_sub['HasDetections'] = predictions_lgb
df_sub.shape


# In[ ]:


filename = 'subm_{:.6f}_{}.csv'.format(roc_auc_score(target, oof_lgb), 
                     dt.now().strftime('%Y-%m-%d-%H-%M'))
print('save to {}'.format(filename))

df_sub.to_csv(filename, index=False)


# ### Obviously this is a very very cursory effort, but I just feel more comfortable when I have the baseline going. Here's to a fun competition! 

# In[ ]:




