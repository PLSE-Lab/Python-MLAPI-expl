#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
import gc


# In[2]:


train_df = pd.read_csv('../input/train.csv') 
test_df = pd.read_csv('../input/test.csv') 
features = [c for c in train_df.columns if c not in ['ID_code', 'target']]
target = train_df.pop('target')
train_df.drop('ID_code',axis=1,inplace=True)
test_df.drop('ID_code',axis=1,inplace=True)


# In[3]:


param = {
    'bagging_freq': 5,
    'bagging_fraction': 0.335,
    'boost_from_average':'false',
    'boost': 'gbdt',
    'feature_fraction': 0.041,
    'learning_rate': 0.0083,
    'max_depth': -1,
    'metric':'auc',
    'min_data_in_leaf': 80,
    'min_sum_hessian_in_leaf': 10.0,
    'num_leaves': 13,
    'num_threads': 8,
    'tree_learner': 'serial',
    'objective': 'binary',
    'verbosity': 1}


# In[4]:


test_df['target'] = -1
train_df['i_am_train'] = 1
test_df['i_am_train'] = 0

full_df = pd.concat([train_df, test_df], axis=0)


# In[10]:


full_df['var_20_counts'] = full_df['var_20'].map(full_df['var_20'].value_counts().to_dict())
full_df['var_155_counts'] = full_df['var_155'].map(full_df['var_155'].value_counts().to_dict())

full_df['var_198_counts'] = full_df['var_198'].map(full_df['var_198'].value_counts().to_dict())
full_df['var_191_counts'] = full_df['var_191'].map(full_df['var_191'].value_counts().to_dict())

full_df['var_177_counts'] = full_df['var_177'].map(full_df['var_177'].value_counts().to_dict())
full_df['var_88_counts'] = full_df['var_88'].map(full_df['var_88'].value_counts().to_dict())

full_df['var_116_counts'] = full_df['var_116'].map(full_df['var_116'].value_counts().to_dict())
full_df['var_4_counts'] = full_df['var_4'].map(full_df['var_4'].value_counts().to_dict())


# In[11]:


full_df['var20_155_countdiff'] = full_df['var_20_counts'] - full_df['var_155_counts']
full_df['var198_191_countdiff'] = full_df['var_198_counts'] - full_df['var_191_counts']
full_df['var177_88_countdiff'] = full_df['var_177_counts'] - full_df['var_88_counts']
full_df['var116_4_countdiff'] = full_df['var_116_counts'] - full_df['var_4_counts']


# In[12]:


train_df = full_df.loc[full_df['i_am_train']==1]
test_df = full_df.loc[full_df['i_am_train']==0]

del train_df['i_am_train'], test_df['i_am_train'], test_df['target'], full_df


# In[13]:


# random_state= 44000
num_folds = 5
folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
oof = np.zeros(len(train_df))
predictions = np.zeros(len(test_df))


# In[14]:


train_df.head()


# In[15]:


features = [c for c in train_df.columns if c not in ['ID_code', 'target']]


# In[ ]:


print('Training the Model:')
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df.values, target.values)):
    print("Fold idx:{}".format(fold_ + 1))
    to_appends = []
    for i in range(9):
        np.random.seed(i)
        to_append_train = train_df.iloc[trn_idx].loc[target==1].copy().apply(np.random.permutation) # Shuffle each column
        to_appends.append(to_append_train)
    full_append = pd.concat(to_appends,axis=0)
    full_append['target'] = 1
    
    trn_data = lgb.Dataset(pd.concat([train_df.iloc[trn_idx][features],full_append[features]], axis=0), label=pd.concat([target.iloc[trn_idx],full_append['target']],axis=0))
    val_data = lgb.Dataset(train_df.iloc[val_idx][features], label=target.iloc[val_idx])
    clf = lgb.train(param, trn_data, 1000000, valid_sets = [trn_data, val_data], verbose_eval=100, early_stopping_rounds = 4000)
    oof[val_idx] = clf.predict(train_df.iloc[val_idx][features], num_iteration=clf.best_iteration)
    predictions += clf.predict(test_df[features], num_iteration=clf.best_iteration) / folds.n_splits
print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))


# In[ ]:


train_df = pd.read_csv('../input/train.csv', usecols=['ID_code','target']) 
test_df = pd.read_csv('../input/test.csv', usecols=['ID_code','var_0']) 


# In[ ]:


sub = pd.DataFrame({"ID_code": test_df.ID_code.values})
sub["target"] = predictions
sub.to_csv('submission_upsampled.csv', index=False)


# In[ ]:


oofs = pd.DataFrame({"ID_code": train_df.ID_code.values})
oofs["target"] = oof
oofs.to_csv('oof_upsampled.csv', index=False)


# In[ ]:





# In[ ]:




