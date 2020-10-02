#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import warnings


# In[ ]:


from sklearn.metrics import roc_auc_score


# In[ ]:


from bayes_opt import BayesianOptimization


# In[ ]:


df_train = pd.read_csv('../input/hcdr-5-prediction-for-train-set/5_predictions.csv', index_col = 'SK_ID_CURR')
df_train.head()


# In[ ]:


for c in df_train.columns.drop('TARGET'):
    print(c, roc_auc_score(df_train['TARGET'], df_train[c]))


# In[ ]:


def ROC_evaluate(**params):
    warnings.simplefilter('ignore')
    
    s = sum(params.values())
    for p in params:
        params[p] = params[p] / s
    
    test_pred_proba = pd.Series(np.zeros(df_train.shape[0]), index = df_train.index)
    
    feats = [f for f in df_train.columns if f not in ['TARGET','SK_ID_CURR', 'index']]
    
    for f in feats:
        test_pred_proba += df_train[f] * params[f]
    
    return roc_auc_score(df_train['TARGET'], test_pred_proba)


# In[ ]:


params = {}
for c in df_train.columns.drop('TARGET'):
    params[c] = (0, 1)
    
bo = BayesianOptimization(ROC_evaluate, params)
bo.maximize(init_points = 50, n_iter = 10)


# In[ ]:


best_params = bo.res['max']['max_params']
print(bo.res['max']['max_val'])


# In[ ]:


best_params


# In[ ]:


best_normalized_params = {}

s = sum(best_params.values())
for p in best_params:
    best_normalized_params[p] = best_params[p] / s

best_normalized_params


# In[ ]:


prediction_train = pd.Series(np.zeros(df_train.shape[0]), index = df_train.index)
    
feats = [f for f in df_train.columns if f not in ['TARGET','SK_ID_CURR', 'index']]
    
for f in feats:
    prediction_train += df_train[f] * best_normalized_params[f]
    
roc_auc_score(df_train['TARGET'], prediction_train)

