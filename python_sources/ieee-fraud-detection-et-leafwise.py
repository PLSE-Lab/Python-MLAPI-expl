#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import random
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, PredefinedSplit


# In[ ]:


SEED = 31
N_ESTIMATORS = 4000
TARGET = 'isFraud'
VALIDATION_PERCENT = 0.01
SCORING = 'roc_auc'


# In[ ]:


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
seed_everything(SEED)


# In[ ]:


file_folder = '../input/ieee-fraud-detection-preprocess'
train = pd.read_csv(f'{file_folder}/train.csv')
test = pd.read_csv(f'{file_folder}/test.csv')
print(f'train={train.shape}, test={test.shape}')


# In[ ]:


excludes = {TARGET}
for i in range(1, 340):
    excludes.add(f'V{i}')


cols = set(train.columns.values) - excludes
PREDICTORS = list(cols)
print(f'{len(PREDICTORS)} predictors={PREDICTORS}')


# In[ ]:


val_size = int(VALIDATION_PERCENT * len(train))
train_size = len(train) - val_size
train_ind = [-1] * train_size
val_ind = [0] * val_size
ps = PredefinedSplit(test_fold=np.concatenate((train_ind, val_ind)))


# In[ ]:


get_ipython().run_cell_magic('time', '', "y_train = train[TARGET]\nx_train = train[PREDICTORS]\nmodel = ExtraTreesClassifier(n_estimators=N_ESTIMATORS, max_features='log2')\npipe = Pipeline([('model', model)])\nparam_grid = {\n    'model__max_leaf_nodes': [160],\n    'model__min_samples_leaf': [100]\n}\ncv = GridSearchCV(pipe, cv=ps, param_grid=param_grid, scoring=SCORING)\ncv.fit(x_train, y_train)\nprint('best_params_={}\\nbest_score_={}'.format(repr(cv.best_params_), repr(cv.best_score_)))")


# In[ ]:


x_test = test[PREDICTORS]
sub = pd.read_csv(f'../input/ieee-fraud-detection/sample_submission.csv')
sub[TARGET] = cv.predict_proba(x_test)[:,1]
sub.head()


# In[ ]:


sub.to_csv('submission.csv', index=False)
print(os.listdir("."))

