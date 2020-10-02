#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
import numpy as np
import random as rn
from sklearn.model_selection import StratifiedKFold,KFold
from math import sqrt
from catboost import Pool, CatBoostClassifier,CatBoostRegressor
from sklearn.metrics import roc_auc_score
import tqdm
from tqdm import tqdm_notebook
from sklearn.utils import shuffle
import warnings
warnings.filterwarnings("ignore")
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(369)
rn.seed(369)
path="../input/"
os.listdir(path)


# In[ ]:


def augment(x,y,t=2):
    xs,xn = [],[]
    for i in range(t):
        mask = y>0
        x1 = x[mask].copy()
        ids = np.arange(x1.shape[0])
        for c in range(200):#(x1.shape[1]):
            np.random.shuffle(ids)
            x1[:,c] = x1[ids][:,c]
            x1[:,c+200] = x1[ids][:,c+200]
            x1[:,c+400] = x1[ids][:,c+400]
        xs.append(x1)

    for i in range(t//2):
        mask = y==0
        x1 = x[mask].copy()
        ids = np.arange(x1.shape[0])
        for c in range(200):#(x1.shape[1]):
            np.random.shuffle(ids)
            x1[:,c] = x1[ids][:,c]
            x1[:,c+200] = x1[ids][:,c+200]
            x1[:,c+400] = x1[ids][:,c+400]
        xn.append(x1)

    xs = np.vstack(xs)
    xn = np.vstack(xn)
    ys = np.ones(xs.shape[0])
    yn = np.zeros(xn.shape[0])
    x = np.vstack([x,xs,xn])
    y = np.concatenate([y,ys,yn])
    return x,y


# In[ ]:


train_data=pd.read_csv(path+"train.csv")
test_data=pd.read_csv(path+"test.csv")


# In[ ]:


features = [x for x in train_data.columns if x not in ['ID_code', "target"]]
label = "target"


# In[ ]:


df_test = test_data.values

unique_samples = []
unique_count = np.zeros_like(df_test)
for feature in tqdm_notebook(range(df_test.shape[1])):
   if feature in [0]:
       print('ok')
       continue
   _, index_, count_ = np.unique(df_test[:, feature], return_counts=True, return_index=True)
   unique_count[index_[count_ == 1], feature] += 1

# Samples which have unique values are real the others are fake
real_samples_indexes = np.argwhere(np.sum(unique_count, axis=1) > 0)[:, 0]
synthetic_samples_indexes = np.argwhere(np.sum(unique_count, axis=1) == 0)[:, 0]


# In[ ]:


idx_ = list(real_samples_indexes)
data=pd.concat([train_data,test_data.loc[idx_]])


# In[ ]:


for col in tqdm_notebook(features):
   
    count=data[col].value_counts()
    rank=len(data[col].unique())
    train_data["rank_"+col]=train_data[col].map(count.rank()/rank)
    test_data["rank_"+col]=test_data[col].map(count.rank()/rank)


# In[ ]:


import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.distplot(train_data['rank_var_45'],color="red")
sns.distplot(test_data['rank_var_45'],color="blue")


# In[ ]:


for col in tqdm_notebook(features):
   
    #if train_data['rank_'+col].nunique()>5:
    unique_value = train_data['rank_'+col].unique().tolist()+test_data['rank_'+col].unique().tolist()
    uq_v = np.min(sorted(list(set(unique_value))))
    train_data['cut_'+col] = train_data[col]
    test_data['cut_'+col] = test_data[col]
    mean1=train_data['cut_'+col].mean()
    mean2=test_data['cut_'+col].mean()
#     vmax = train_data[col][train_data['rank_'+col]==uq_v].min()
#     vmin = test_data[col][test_data['rank_'+col]==uq_v].max()
#     split_position = (vmax+vmin)//2
#     train_data['cut_'+col][(train_data['rank_'+col]==uq_v)&\
#                                         (train_data[col]<=uq_v)]=0
#     train_data['cut_'+col][(train_data['rank_'+col]==uq_v)&\
#                                         (train_data[col]>uq_v)]=1
    train_data['cut_'+col][(train_data['rank_'+col]==uq_v)]=mean1
#     test_data['cut_'+col][(test_data['rank_'+col]==uq_v)&\
#                                         (test_data[col]<=uq_v)]=0
#     test_data['cut_'+col][(test_data['rank_'+col]==uq_v)&\
#                                         (test_data[col]>uq_v)]=1
    test_data['cut_'+col][(test_data['rank_'+col]==uq_v)]=mean2


# In[ ]:


train=train_data
test=test_data

import gc
del train_data,test_data,data
gc.collect()


# In[ ]:


features = [x for x in train.columns if x not in ['ID_code', "target"]]
label = "target"


# In[ ]:


"feature count:",len(features)


# In[ ]:


skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=99999)
oof = train[['ID_code']]
oof['target'] = 0
predictions = test[['ID_code']]
val_aucs = []


# In[ ]:


for fold, (trn_idx, val_idx) in enumerate(skf.split(train, train['target'])):
    X_train, y_train = train.iloc[trn_idx][features], train.iloc[trn_idx]['target']
    X_valid, y_valid = train.iloc[val_idx][features], train.iloc[val_idx]['target']
    
    N = 3
    p_valid,yp = 0,0
    for i in range(N):
        X_t, y_t = augment(X_train.values, y_train.values,2)
        X_t = pd.DataFrame(X_t)
        X_t.columns = features
    
        trn_data = Pool(X_t, label=y_t)
        val_data = Pool(X_valid, label=y_valid)
        evals_result = {}
        model = CatBoostClassifier(iterations=10000000, 
        learning_rate = 0.03,
        od_type='Iter',
         od_wait=3500,
        loss_function="Logloss",
        eval_metric='AUC',
#         depth=3,
#         bootstrap_type='Bernoulli',
        bagging_temperature=0.7,                   
        random_seed = 2019,
        task_type='GPU'
                          )
        model.fit(trn_data,eval_set=val_data,
            use_best_model=True,
            verbose=3000
                         )
        p_valid += model.predict_proba(X_valid)[:,1]
        yp += model.predict_proba(test[features])[:,1]
        gc.collect()
    gc.collect()
    oof['target'][val_idx] = p_valid/N
    val_score = roc_auc_score(y_valid, p_valid)
    print("fold {}|5".format(fold+1),"auc score : ",val_score)
    val_aucs.append(val_score)
    
    predictions['fold{}'.format(fold+1)] = yp/N


# In[ ]:


print("oof auc")
roc_auc_score(train['target'], oof['target'])


# In[ ]:


"auc mean:",np.mean(val_aucs)


# In[ ]:


predictions['target'] = np.mean(predictions[[col for col in predictions.columns if col not in ['ID_code', 'target']]].values, axis=1)
predictions.to_csv('lgb_all_predictions.csv', index=None)
sub_df = pd.DataFrame({"ID_code":test["ID_code"].values})
sub_df["target"] = predictions['target']
sub_df.to_csv("lgb_submission.csv", index=False)
oof_all=pd.concat([oof,sub_df])
oof_all.to_csv('augment_cat_oof.csv', index=False)

