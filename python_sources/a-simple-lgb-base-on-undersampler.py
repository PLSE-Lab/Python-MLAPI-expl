#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgb
import os
print(os.listdir("../input"))
import catboost as cb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve


# In[ ]:


test = pd.read_csv("../input/test.csv")
train = pd.read_csv("../input/train.csv")


# In[ ]:


ytrain = train['target']
xtrain = train.iloc[:,2:]
xtest = test.iloc[:,1:]


# In[ ]:


xtrain.shape


# In[ ]:


from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=1, replacement=True)
xtrain1,ytrain1  = rus.fit_sample(xtrain, ytrain )


# In[ ]:


xtrain1.shape


# In[ ]:


param = {
        'num_leaves': 2,
        'learning_rate': 0.1,
        'feature_fraction': 0.2,
        'max_depth': -1,
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'metric': 'auc',
    }

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=4590)
oof = np.zeros(len(xtrain))
ypred = np.zeros(len(xtest))
feature_importance_df = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(xtrain1,ytrain1)):
    print("fold {}".format(fold_))
    
    rus = RandomUnderSampler(random_state=fold_, replacement=True)
    xtrain1,ytrain1  = rus.fit_sample(xtrain, ytrain )
    
    trn_data = lgb.Dataset(xtrain1[trn_idx], label=ytrain1[trn_idx])#, categorical_feature=categorical_feats)
    val_data = lgb.Dataset(xtrain1[val_idx], label=ytrain1[val_idx])#, categorical_feature=categorical_feats)

    num_round = 30000
    model = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=100, early_stopping_rounds = 300)
    oof += model.predict(xtrain)/ 5
    ypred += model.predict(xtest) / 5

roc_auc_score(ytrain,oof)


# In[ ]:


df = pd.DataFrame({'ID_code':test['ID_code'],'target':ypred})
df.to_csv('undersamping.csv',index=None)


# In[ ]:


roc_auc_score(ytrain,oof)


# In[ ]:




