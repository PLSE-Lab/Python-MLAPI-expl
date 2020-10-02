#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import gc
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import xgboost as xgb
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from pprint import pprint
from tqdm import tqdm


# # Load and Preprocess

# In[ ]:


comp_path = Path('/kaggle/input/cat-in-the-dat/')

train = pd.read_csv(comp_path / 'train.csv', index_col='id')
test = pd.read_csv(comp_path / 'test.csv', index_col='id')
sample_submission = pd.read_csv(comp_path / 'sample_submission.csv', index_col='id')

y_train = train.pop('target')

# Simple label encoding
for c in tqdm(train.columns):
    le = LabelEncoder()
    # this is cheating in real life; you won't have the test data ahead of time ;-)
    le.fit(pd.concat([train[c], test[c]])) 
    train[c] = le.transform(train[c])
    test[c] = le.transform(test[c])

X_train, X_val, y_train, y_val = train_test_split(
    train, y_train, test_size=0.2, random_state=2019
)


# # Train

# In[ ]:


get_ipython().run_cell_magic('time', '', "clf = xgb.XGBClassifier(\n    learning_rate=0.05,\n    n_estimators=50000, # Very large number\n    seed=2019,\n    reg_alpha=5,\n    eval_metric='auc',\n    tree_method='gpu_hist'\n)\nclf.fit(\n    X_train, \n    y_train, \n    eval_set=[(X_train, y_train), (X_val, y_val)],\n    early_stopping_rounds=50,\n    verbose=50\n)")


# In[ ]:


results = clf.evals_result()
epochs = len(results['validation_0']['auc'])
x_axis = range(0, epochs)

# plot log loss
plt.figure(figsize=(15, 7))
plt.plot(x_axis, results['validation_0']['auc'], label='Train')
plt.plot(x_axis, results['validation_1']['auc'], label='Val')
plt.legend()
plt.ylabel('AUC')
plt.xlabel('# of iterations')
plt.title('XGBoost AUC')
plt.show()


# # Submit

# In[ ]:


sample_submission['target'] = clf.predict_proba(test, ntree_limit=clf.best_ntree_limit)[:, 1]
sample_submission.to_csv('xgb_submission.csv')

