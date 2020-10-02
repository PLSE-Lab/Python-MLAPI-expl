#!/usr/bin/env python
# coding: utf-8

# # Stacking code
# lgb + NN + knn  
# lgb: https://www.kaggle.com/masatomatsui/otto-simple-lgb  
# NN : https://www.kaggle.com/masatomatsui/otto-simple-nn/  
# knn: https://www.kaggle.com/masatomatsui/otto-knn  

# In[ ]:





# In[ ]:


import numpy as np 
import pandas as pd 
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import log_loss
import seaborn as sns
import sklearn.preprocessing as pp
from scipy import optimize
from sklearn.metrics import log_loss


# In[ ]:


train_base = pd.read_csv('../input/otto-group-product-classification-challenge/train.csv')
test_base = pd.read_csv('../input/otto-group-product-classification-challenge/test.csv')
sample_submit = pd.read_csv('../input/otto-group-product-classification-challenge/sampleSubmission.csv')


# In[ ]:


oof_lgb = pd.read_csv("../input/otto-simple-lgb/oof_lgb.csv")
oof_nn = pd.read_csv("../input/otto-simple-nn/oof_nn.csv")
oof_knn = pd.read_csv("../input/otto-knn/oof_knn.csv")

test_lgb = pd.read_csv("../input/otto-simple-lgb/submit_lgb.csv")
test_nn = pd.read_csv("../input/otto-simple-nn/submit_nn.csv")
test_knn = pd.read_csv("../input/otto-knn/submit_knn.csv")


# In[ ]:


train = pd.concat([oof_lgb, oof_nn, oof_knn, train_base], axis=1)
test = pd.concat([test_lgb, test_nn, test_knn, test_base], axis=1)


# In[ ]:


train['target'] = train['target'].str.replace('Class_', '')
train['target'] = train['target'].astype(int) - 1


# In[ ]:


NFOLDS = 5
RANDOM_STATE = 871975

excluded_column = ['target', 'id']
cols = [c for c in train.columns if c not in (excluded_column + [])]

folds = StratifiedKFold(n_splits=NFOLDS, shuffle=True, 
                        random_state=RANDOM_STATE)


# In[ ]:


params = {
    'bagging_freq': 5,          'bagging_fraction': 0.8,   'boost_from_average':'false',   'boost': 'gbdt',
    'feature_fraction': 0.3,   'learning_rate': 0.05,     'max_depth': -1,  'metrics':'multi_logloss',   
    'min_data_in_leaf': 10,     'min_sum_hessian_in_leaf': 8.0,'num_leaves': 8,           'num_threads': 8,
    'tree_learner': 'serial',   'objective': 'multiclass',   'num_class': 9,   'verbosity': 1
}


# In[ ]:


y_pred_lgb = np.zeros((len(test), 9))
oof = np.zeros((len(train), 9))
score = 0
feature_importance_df = pd.DataFrame()
valid_predict = []
for fold_n, (train_index, valid_index) in enumerate(folds.split(train, y = train['target'])):
    print('Fold', fold_n)
    X_train, X_valid = train.iloc[train_index], train.iloc[valid_index]
    y_train, y_valid = X_train['target'].astype(int), X_valid['target'].astype(int)
    
    train_data = lgb.Dataset(X_train[cols], label=y_train)
    valid_data = lgb.Dataset(X_valid[cols], label=y_valid)

    best_params = {}
    tuning_history = []

    lgb_model = lgb.train(params,train_data,num_boost_round=30000,
                    valid_sets = [train_data, valid_data],verbose_eval=300,early_stopping_rounds = 300)
    
    y_valid_predict = lgb_model.predict(X_valid[cols], num_iteration=lgb_model.best_iteration)
    score += log_loss(y_valid, y_valid_predict)
    oof[valid_index] = y_valid_predict
    print('Fold', fold_n, 'valid logloss', log_loss(y_valid, y_valid_predict))
    
    y_pred_lgb += lgb_model.predict(test[cols], num_iteration=lgb_model.best_iteration)/NFOLDS
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = cols
    fold_importance_df["importance"] = lgb_model.feature_importance(importance_type='gain')
    fold_importance_df["fold"] = fold_n + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
print('valid logloss average:', score/NFOLDS)


# In[ ]:


feature_importance_df[["feature", "importance"]].groupby("feature", as_index=False).mean().sort_values(by="importance", ascending=False).head(20)


# In[ ]:


sample_submit = pd.read_csv('../input/otto-group-product-classification-challenge/sampleSubmission.csv')
submit = pd.concat([sample_submit[['id']], pd.DataFrame(y_pred_lgb)], axis = 1)
submit.columns = sample_submit.columns
submit.to_csv('submit.csv', index=False)


# In[ ]:


column_name = ['stack_lgb_' + str(i) for i in range(9)]
pd.DataFrame(oof, columns = column_name).to_csv('oof_lgb.csv', index=False)
pd.DataFrame(y_pred_test, columns = column_name).to_csv('submit_lgb.csv', index=False)


# In[ ]:




