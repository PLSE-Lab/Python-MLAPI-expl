#!/usr/bin/env python
# coding: utf-8

# 1. 1. ## Kaggle competition  Santander customer transaction using LGBM
# 
# It's a fairly simple neural network with enhancements targeted for later. 
# Steps:
# 1. Load data
# 2. Add features
# 3. Model training using 5-fold StratifiedKfol

# In[1]:


get_ipython().system('rm -r /opt/conda/lib/python3.6/site-packages/lightgbm')
get_ipython().system('git clone --recursive https://github.com/Microsoft/LightGBM')


# In[2]:


get_ipython().system('apt-get install -y -qq libboost-all-dev')


# In[3]:


get_ipython().run_cell_magic('bash', '', 'cd LightGBM\nrm -r build\nmkdir build\ncd build\ncmake -DUSE_GPU=1 -DOpenCL_LIBRARY=/usr/local/cuda/lib64/libOpenCL.so -DOpenCL_INCLUDE_DIR=/usr/local/cuda/include/ ..\nmake -j$(nproc)')


# In[4]:


get_ipython().system('cd LightGBM/python-package/;python3 setup.py install --precompile')


# In[5]:


get_ipython().system('mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd')
get_ipython().system('rm -r LightGBM')


# In[6]:


get_ipython().system('nvidia-smi')


# In[8]:


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix,f1_score, precision_recall_curve, average_precision_score, roc_auc_score, confusion_matrix

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import gc

pd.set_option('display.max_columns', 1000)
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns

plt.style.use('ggplot')
import os
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)


# In[9]:


get_ipython().run_cell_magic('time', '', "df_train_raw = pd.read_csv('../input/train.csv')\ndf_test_raw = pd.read_csv('../input/test.csv')")


# In[10]:


get_ipython().run_cell_magic('time', '', 'df_train = df_train_raw.copy()\ndf_test = df_test_raw.copy()')


# In[11]:


train_cols = [col for col in df_train.columns if col not in ['ID_code', 'target']]
y_train = df_train['target']


# In[12]:


df_train.shape


# In[13]:


interactions= {'var_81':['var_53','var_139','var_12','var_76'],
               'var_12':['var_139','var_26','var_22', 'var_53','var_110','var_13'],
               'var_139':['var_146','var_26','var_53', 'var_6', 'var_118'],
               'var_53':['var_110','var_6'],
              'var_26':['var_110','var_109','var_12'],
              'var_118':['var_156'],
              'var_9':['var_89'],
              'var_22':['var_28','var_99','var_26'],
              'var_166':['var_110'],
              'var_146':['var_40','var_0'],
              'var_80':['var_12']}


# In[14]:


get_ipython().run_cell_magic('time', '', "for col in train_cols:\n        df_train[col+'_2'] = df_train[col] * df_train[col]\n        df_train[col+'_3'] = df_train[col] * df_train[col]* df_train[col]\n#         df_train[col+'_4'] = df_train[col] * df_train[col]* df_train[col]* df_train[col]\n        df_test[col+'_2'] = df_test[col] * df_test[col]\n        df_test[col+'_3'] = df_test[col] * df_test[col]* df_test[col]")


# In[15]:


get_ipython().run_cell_magic('time', '', "for df in [df_train, df_test]:\n    df['sum'] = df[train_cols].sum(axis=1)  \n    df['min'] = df[train_cols].min(axis=1)\n    df['max'] = df[train_cols].max(axis=1)\n    df['mean'] = df[train_cols].mean(axis=1)\n    df['std'] = df[train_cols].std(axis=1)\n    df['skew'] = df[train_cols].skew(axis=1)\n    df['kurt'] = df[train_cols].kurtosis(axis=1)\n    df['med'] = df[train_cols].median(axis=1)")


# In[16]:


get_ipython().run_cell_magic('time', '', "for key in interactions:\n    for value in interactions[key]:\n        df_train[key+'_'+value+'_mul'] = df_train[key]*df_train[value]\n        df_train[key+'_'+value+'_div'] = df_train[key]/df_train[value]\n        df_train[key+'_'+value+'_sum'] = df_train[key] + df_train[value]\n        df_train[key+'_'+value+'_sub'] = df_train[key] - df_train[value]\n        \n        df_test[key+'_'+value+'_mul'] = df_test[key]*df_test[value]\n        df_test[key+'_'+value+'_div'] = df_test[key]/df_test[value]\n        df_test[key+'_'+value+'_sum'] = df_test[key] + df_test[value]\n        df_test[key+'_'+value+'_sub'] = df_test[key] - df_test[value]")


# In[17]:


df_train['num_zero_rows'] = (df_train_raw[train_cols] == 0).astype(int).sum(axis=1)
df_test['num_zero_rows'] = (df_test_raw[train_cols] == 0).astype(int).sum(axis=1)


# In[18]:


df_train.head()


# In[19]:


all_columns = [col for col in df_train.columns if col not in ['ID_code', 'target']]


# ### Start LGBM

# In[20]:


params = {
        'num_leaves': 13,
        'max_bin': 63,
        'min_data_in_leaf': 80,
        'learning_rate': 0.0081,
        'min_sum_hessian_in_leaf': 10.0,
        'bagging_fraction': 0.331, 
        'bagging_freq': 5, 
        'max_depth': -1,
        'save_binary': True,
        'feature_fraction': 0.041,
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'verbose': 1,
        'metric': 'auc',
#         'is_unbalance': True,
        'boost_from_average': False,
        'device': 'gpu',
        'gpu_platform_id':0,
        'gpu_device_id': 0,
        'seed':44000
    }

num_round = 20000


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nfolds = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 44000)\noof = np.zeros(len(df_train))\n\npredictions = np.zeros(len(df_test))\nfeature_import_df = pd.DataFrame()\n\nfor n_fold, (train_idx, val_idx) in enumerate(folds.split(df_train, y_train)):\n    print("fold number =", n_fold+1)\n    train_data = lgb.Dataset(df_train.iloc[train_idx][all_columns], label = y_train.iloc[train_idx])\n    val_y = y_train.iloc[val_idx]\n    val_data = lgb.Dataset(df_train.iloc[val_idx][all_columns], label = val_y)\n    \n    \n    watchlist = [train_data,val_data]\n    clf = lgb.train(params, train_data, num_boost_round = num_round,\n                   valid_sets = watchlist, verbose_eval = 4000,\n                   early_stopping_rounds=3000)\n    \n    oof[val_idx] = clf.predict(df_train.iloc[val_idx][all_columns], num_iteration=clf.best_iteration)\n    \n    fold_import_df = pd.DataFrame()\n    fold_import_df[\'Feature\'] = all_columns\n    fold_import_df["importance"] = clf.feature_importance()\n    fold_import_df[\'fold\'] = n_fold +1\n    feature_import_df = pd.concat([feature_import_df,fold_import_df], axis = 0)\n    \n    predictions += clf.predict(df_test[all_columns])/folds.n_splits\n    \n    print("\\tFold AUC Score: {}\\tf1_score: {}\\n".format(roc_auc_score(val_y,oof[val_idx]),\n                                                       f1_score(val_y,np.round(oof[val_idx]))))\n    gc.collect()\n          \nprint("\\n CV AUC Score and std", roc_auc_score(y_train, oof),np.std(oof))\nprint("CV F1 Score", f1_score(y_train, np.round(oof)))')


# ### Checking best iteration

# ### Preparing submission file

# In[ ]:


sub = pd.DataFrame({'ID_code': df_test.ID_code.values,
                   'target': predictions})
sub.to_csv('lgbm_0401_kernelgpu.csv', index = False)


# In[ ]:




