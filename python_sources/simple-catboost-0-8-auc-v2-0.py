#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier


# In[ ]:


PATH_TO_DATA = Path('../input/cat-in-the-dat/')


# In[ ]:


train_df = pd.read_csv(PATH_TO_DATA / 'train.csv')


# In[ ]:


train_df.head()


# In[ ]:


test_df = pd.read_csv(PATH_TO_DATA / 'test.csv')


# In[ ]:


test_df.head()


# In[ ]:


categ_feat_idx = np.where(train_df.drop('target', axis=1).dtypes == 'object')[0]
categ_feat_idx


# In[ ]:


X_train = train_df.drop('target', axis=1).values
y_train = train_df['target'].values
X_test = test_df.values


# In[ ]:


X_train_part, X_valid, y_train_part, y_valid = train_test_split(X_train, y_train, 
                                                                test_size=0.3, 
                                                                random_state=17)


# In[ ]:


SEED = 17
params = {'loss_function':'Logloss', # objective function
          'eval_metric':'AUC', # metric
          'verbose': 200, # output to stdout info about training process every 200 iterations
          'early_stopping_rounds': 200,
          'cat_features': categ_feat_idx,
          #'task_type': 'GPU',
          'random_seed': SEED
         }
ctb = CatBoostClassifier(**params)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'ctb.fit(X_train_part, y_train_part,\n        eval_set=(X_valid, y_valid),\n        use_best_model=True,\n        plot=True);')


# In[ ]:


ctb_valid_pred = ctb.predict_proba(X_valid)[:, 1]


# In[ ]:


roc_auc_score(y_valid, ctb_valid_pred)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'ctb.fit(X_train, y_train,\n        eval_set=(X_valid, y_valid),\n        use_best_model=True,\n        plot=True);')


# In[ ]:


ctb_test_pred = ctb.predict_proba(X_test)[:, 1]


# In[ ]:


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    
    sample_sub = pd.read_csv(PATH_TO_DATA / 'sample_submission.csv', 
                             index_col='id')
    sample_sub['target'] = ctb_test_pred
    sample_sub.to_csv('ctb_pred.csv')


# In[ ]:


get_ipython().system('head ctb_pred.csv')


# In[ ]:


ctb2.get_feature_importance(prettified=True)


# ## V.20 - Feature engineering & tuning
# 

# In[ ]:


train_df['nom_c1'] = train_df['nom_1'] + ' ' + train_df['nom_2']


# In[ ]:


train_df.head()


# In[ ]:


categ_feat_idx = np.where(train_df.drop('target', axis=1).dtypes == 'object')[0]
categ_feat_idx

X_train = train_df.drop('target', axis=1).values
y_train = train_df['target'].values
X_test = test_df.values


# In[ ]:


X_train_part, X_valid, y_train_part, y_valid = train_test_split(X_train, y_train, 
                                                                test_size=0.3, 
                                                                random_state=17)


# In[ ]:


SEED = 17
params = {'loss_function':'Logloss', # objective function
          'eval_metric':'AUC', # metric
          'verbose': 200, # output to stdout info about training process every 200 iterations
          'early_stopping_rounds': 200,
          'cat_features': categ_feat_idx,
          #'task_type': 'GPU',
          'random_seed': SEED
         }
ctb2 = CatBoostClassifier(**params)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'ctb2.fit(X_train_part, y_train_part,\n        eval_set=(X_valid, y_valid),\n        use_best_model=True,\n        plot=True);')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'ctb2.fit(X_train, y_train,\n        eval_set=(X_valid, y_valid),\n        use_best_model=True,\n        plot=True);')


# In[ ]:


ctb2_test_pred = ctb2.predict_proba(X_test)[:, 1]


# In[ ]:


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    
    sample_sub = pd.read_csv(PATH_TO_DATA / 'sample_submission.csv', 
                             index_col='id')
    sample_sub['target'] = ctb2_test_pred
    sample_sub.to_csv('ctb_pred2.csv')

