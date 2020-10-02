#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import gc
import pandas as pd
import numpy as np
import random as rn

from sklearn.model_selection import StratifiedKFold,KFold
from math import sqrt
from catboost import Pool, CatBoostClassifier,CatBoostRegressor
from sklearn.metrics import roc_auc_score
import tqdm
from tqdm import tqdm_notebook,tqdm
from sklearn.utils import shuffle
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)
path="../input/"
os.listdir("../input/")


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


# In[ ]:


data=pd.concat([train_data,test_data.loc[idx_]])


# In[ ]:



for col in tqdm_notebook(features):
   
    count=data[col].value_counts()
    rank=len(data[col].unique())
    train_data["rank_"+col]=train_data[col].map(count)
    test_data["rank_"+col]=test_data[col].map(count)
     


# In[ ]:


import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.distplot(train_data['rank_var_188'],color="red")
sns.distplot(test_data['rank_var_188'],color="blue")


# In[ ]:


for col in tqdm_notebook(features):
   
    unique_value = train_data['rank_'+col].unique().tolist()+test_data['rank_'+col].unique().tolist()
    uq_v = np.min(sorted(list(set(unique_value))))
    train_data['cut_'+col] = train_data[col]
    test_data['cut_'+col] = test_data[col]
    median1=data[col].mean()
#     median2=data[col].mean()

    train_data['cut_'+col][(train_data['rank_'+col]==uq_v)]=median1
    test_data['cut_'+col][(test_data['rank_'+col]==uq_v)]=median1


# In[ ]:


train=train_data#.join(train_clu)
test=test_data#.join(test_clu)


# In[ ]:



del train_data,test_data
gc.collect()


# In[ ]:


features = [x for x in train.columns if x not in ['ID_code', "target"]+['o-c_var_'+str(i) for i in range(200)]]
label = "target"


# In[ ]:


gc.collect()


# In[ ]:


features = ["cut_var_"+str(i) for i in range(200)]+["rank_var_"+str(i) for i in range(200)]


# In[ ]:


"feature count:",len(features)


# In[ ]:


import lightgbm as lgb
def run_cv_model(train, test, target, model_fn, params={}, eval_fn=None, label='model'):
    folds=5
    
    kf = StratifiedKFold(n_splits=folds, random_state=99999, shuffle=True)
    fold_splits = kf.split(train, target)
    
    cv_scores = []
    pred_full_test = 0
    
    pred_train = np.zeros((train.shape[0], folds))
    feature_importance_df = pd.DataFrame()
    i = 1
    for dev_index, val_index in fold_splits:
        print( label + ' | FOLD ' + str(i) + '/'+str(folds))
        if isinstance(train, pd.DataFrame):
            dev_X, val_X = train.iloc[dev_index], train.iloc[val_index]
            dev_y, val_y = target[dev_index], target[val_index]
        else:
            dev_X, val_X = train[dev_index], train[val_index]
            dev_y, val_y = target[dev_index], target[val_index]
        params2 = params.copy()
        pred_val_y, pred_test_y, importances = model_fn(dev_X, dev_y, val_X, val_y, test, params2)
        gc.collect()
        pred_full_test = pred_full_test + pred_test_y
        pred_train[val_index] = pred_val_y
       
        if eval_fn is not None:
            cv_score = eval_fn(val_y, pred_val_y)
            cv_scores.append(cv_score)
          
           
            print(label + ' cv score {}: AUC {} '.format(i, cv_score))
            print("##"*40)
        fold_importance_df = pd.DataFrame()
        fold_importance_df['feature'] =train.columns.values
        fold_importance_df['importance'] =importances
        fold_importance_df['fold'] = i
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)        
        i += 1
#     print('{} cv RMSE scores : {}'.format(label, cv_scores))
    print('{} cv mean AUC score : {}'.format(label, np.mean(cv_scores)))
    print('{} cv std AUC score : {}'.format(label, np.std(cv_scores)))
   

    
    pred_full_test = pred_full_test / float(folds)
    results = {'label': label,
               'train': pred_train, 'test': pred_full_test,
                'cv': cv_scores, 
               'importance': feature_importance_df,
               }
    return results

params = {
        'bagging_freq': 5,
        'bagging_fraction': 0.4,
        'boost_from_average': 'false',
        'boost': 'gbdt',
        'feature_fraction': 0.04,
        'learning_rate': 0.0083,
#         'max_depth': 3,
        'metric': 'auc',
        'min_data_in_leaf': 80,
        'min_sum_hessian_in_leaf': 10.0,
        'num_leaves': 13,
        'num_threads': -1,
        'tree_learner': 'serial',
        'objective': 'binary',
        # 'device_type':'gpu',
        # 'is_unbalance':True,
        'verbosity': -1,
     'verbose_eval': 4000,
         
          'num_rounds': 1000000,
     'early_stop': 4000,
    }

def runLGB(train_X, train_y, test_X, test_y, test_X2, params):
#     print('Prep LGB')
    d_train = lgb.Dataset(train_X, label=train_y)
    d_valid = lgb.Dataset(test_X, label=test_y)
    watchlist = [d_train, d_valid]
#     print('Train LGB')
    num_rounds = params.pop('num_rounds')
    verbose_eval = params.pop('verbose_eval')
    early_stop = None
    if params.get('early_stop'):
        early_stop = params.pop('early_stop')
    model = lgb.train(params,
                      train_set=d_train,
                      num_boost_round=num_rounds,
                      valid_sets=watchlist,
                      verbose_eval=verbose_eval,
                      early_stopping_rounds=early_stop)
    print('Predict 1/2')
    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
   
    pred_test_y2 = model.predict(test_X2, num_iteration=model.best_iteration)
    gc.collect()
    return pred_test_y.reshape(-1, 1), pred_test_y2.reshape(-1, 1), model.feature_importance()
def runCAT(train_X, train_y, test_X, test_y, test_X2, params):
#     print('Prep LGB')
    d_train = Pool(train_X, label=train_y)
    d_valid = Pool(test_X, label=test_y)
    watchlist = (d_train, d_valid)
#     print('Train LGB')
    num_rounds = params.pop('num_rounds')
    verbose_eval = params.pop('verbose_eval')
    early_stop = None
    if params.get('early_stop'):
        early_stop = params.pop('early_stop')
    model = CatBoostClassifier(iterations=num_rounds, 
        learning_rate = 0.003,
        od_type='Iter',
         od_wait=early_stop,
        loss_function="Logloss",
        eval_metric='AUC',
#         depth=3,
        bagging_temperature=0.7,                   
        random_seed = 2019,
#         task_type='GPU'
                          )
    model.fit(d_train,eval_set=d_valid,
            use_best_model=True,
            verbose=verbose_eval
                         )
    
    print('Predict 1/2')
    pred_test_y = model.predict_proba(test_X)[:,1]
    
    pred_test_y2 = model.predict_proba(test_X2)[:,1]
    gc.collect()
    return pred_test_y.reshape(-1, 1), pred_test_y2.reshape(-1, 1), 0
 
results = run_cv_model(train[features], test[features], train[label], runLGB, params, roc_auc_score, 'LGB')


# In[ ]:


imports = results['importance'].groupby('feature')['feature', 'importance'].mean().reset_index()
imp=imports.sort_values('importance', ascending=False)
imp.index=range(len(imp))
imp.iloc[:610]


# In[ ]:


train_predictions = [r[0] for r in results['train']]
roc_auc_score(train[label],train_predictions)


# In[ ]:


train_predictions = [r[0] for r in results['train']]
test_predictions = [r[0] for r in results['test']]
sub=test[['ID_code']]
sub['target']=test_predictions
if not os.path.exists("submit"):
    os.mkdir("submit")
sub.to_csv("submit/lgb_submmision.csv",index=False)


# In[ ]:


train_oof=train[['ID_code']]
train_oof['target']=train_predictions 
oof=pd.concat([train_oof,sub])
if not os.path.exists("oof"):
    os.mkdir("oof")
oof.to_csv("./oof/lgb_oof.csv",index=False)


# ### augment

# In[ ]:


# https://stackoverflow.com/questions/50554272/randomly-shuffle-items-in-each-row-of-numpy-array
def disarrange(a, axis=-1):
    """
    Shuffle `a` in-place along the given axis.

    Apply numpy.random.shuffle to the given axis of `a`.
    Each one-dimensional slice is shuffled independently.
    """
    b = a.swapaxes(axis, -1)
    # Shuffle `b` in-place along the last axis.  `b` is a view of `a`,
    # so `a` is shuffled in place, too.
    shp = b.shape[:-1]
    for ndx in np.ndindex(shp):
        np.random.shuffle(b[ndx])
    return

def augment(x,y,t=2):
    xs,xn = [],[]
    for i in range(t):
        mask = y>0
        x1 = x[mask].copy()
        disarrange(x1,axis=0)
        xs.append(x1)

    for i in range(t//2):
        mask = y==0
        x1 = x[mask].copy()
        disarrange(x1,axis=0)
        xn.append(x1)

    xs = np.vstack(xs)
    xn = np.vstack(xn)
    ys = np.ones(xs.shape[0])
    yn = np.zeros(xn.shape[0])
    x = np.vstack([x,xs,xn])
    y = np.concatenate([y,ys,yn])
    return x,y


# In[ ]:


# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=99999)
# oof = train[['ID_code']]
# oof['target'] = 0
# predictions = test[['ID_code']]
# val_aucs = []
# feature_importance_df = pd.DataFrame()


# In[ ]:


# gc.collect()


# In[ ]:


# import lightgbm as lgb
# # lgb_params  = {
# #         'bagging_freq': 5,
# #         'bagging_fraction': 0.4,
# #         'boost_from_average': 'false',
# #         'boost': 'gbdt',
# #         'feature_fraction': 0.05,
# #         'learning_rate': 0.01,
# #         'max_depth': -1,
# #         'metric': 'auc',
# #         'min_data_in_leaf': 80,
# #         'min_sum_hessian_in_leaf': 10.0,
# #         'num_leaves': 13,
# #         'num_threads': -1,
# #         'tree_learner': 'serial',
# #         'objective': 'binary',
# #         # 'device_type':'gpu',
# #         # 'is_unbalance':True,
# #         'verbosity': -1,

# #           "random_state":1017,
# #     }
# lgb_params  = {
#         'bagging_freq': 5,
#         'bagging_fraction': 0.335,
#         'boost_from_average': 'false',
#         'boost': 'gbdt',
#         'feature_fraction': 0.043,
#         'learning_rate': 0.0083,
#         'max_depth': -1,
#         'metric': 'auc',
#         'min_data_in_leaf': 80,
#         'min_sum_hessian_in_leaf': 10.0,
#         'num_leaves': 13,
#         'num_threads': -1,
#         'tree_learner': 'serial',
#         'objective': 'binary',
#         # 'device_type':'gpu',
#         # 'is_unbalance':True,
#         'verbosity': -1,

        
#     }


# In[ ]:


# for fold, (trn_idx, val_idx) in enumerate(skf.split(train, train['target'])):
#     X_train, y_train = train.iloc[trn_idx][features], train.iloc[trn_idx]['target']
#     X_valid, y_valid = train.iloc[val_idx][features], train.iloc[val_idx]['target']
    
#     N = 5
#     p_valid,yp = 0,0
#     for i in range(N):
#         print("##"*40)
#         print("FOLD {} N {} ".format(fold+1,i+1))
#         X_t, y_t = augment(X_train.values, y_train.values)
#         X_t = pd.DataFrame(X_t)
#         X_t.columns = features
    
#         trn_data = lgb.Dataset(X_t, label=y_t)
#         val_data = lgb.Dataset(X_valid, label=y_valid)
#         evals_result = {}
#         lgb_clf = lgb.train(lgb_params,
#                         trn_data,
#                         10000000,
#                         valid_sets = [trn_data, val_data],
#                         early_stopping_rounds=4000,
#                         verbose_eval = 4000,
#                         evals_result=evals_result
#                        )
#         p_valid += lgb_clf.predict(X_valid,num_iteration=lgb_clf.best_iteration)
#         yp += lgb_clf.predict(test[features],num_iteration=lgb_clf.best_iteration)
#         gc.collect()
        
#     gc.collect()
#     oof['target'][val_idx] = p_valid/N
#     val_score = roc_auc_score(y_valid, p_valid)
#     print("fold {}|".format(fold+1),"auc score : ",val_score)
#     val_aucs.append(val_score)
    
#     predictions['fold{}'.format(fold+1)] = yp/N


# In[ ]:


# mean_auc = np.mean(val_aucs)
# std_auc = np.std(val_aucs)
# all_auc = roc_auc_score(train['target'], oof['target'])
# print("Mean auc: %.9f, std: %.9f. All auc: %.9f." % (mean_auc, std_auc, all_auc))


# In[ ]:


# if not os.path.exists("augment"):
#     os.mkdir("augment")
# predictions['target'] = np.mean(predictions[[col for col in predictions.columns if col not in ['ID_code', 'target']]].values, axis=1)
# predictions.to_csv('./augment/lgb_all_predictions.csv', index=None)
# sub_df = pd.DataFrame({"ID_code":test["ID_code"].values})
# sub_df["target"] = predictions['target']
# sub_df.to_csv("./augment/lgb_submission.csv", index=False)
# oof_all=pd.concat([oof,sub_df])
# oof_all.to_csv('./augment/augment_lgb_oof.csv', index=False)


# In[ ]:


# with open("log.txt","w")as f:
#     f.write("auc oof:{} mean:{}".format(roc_auc_score(train['target'], oof['target']),np.mean(val_aucs)))


# In[ ]:




