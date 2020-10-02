#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import gc
import pickle
from pathlib import Path
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#math
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import GridSearchCV, StratifiedKFold # Used to use Kfold to train our model


# In[ ]:


# config
MODEL_NAME = "santander_2"
RESCALE = False
MIN_MAX = (-1,1)


# In[ ]:


#file paths
input_path = Path("../input")
train_csv = str(input_path / "train.csv")
test_csv = str(input_path / "test.csv")


# In[ ]:


#load original features
train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)

train_df.head()


# In[ ]:


#rescale all feature cols to, -1 to 1
feature_columns = []
for i in range(0,200,1):
    key = 'var_' + str(i)
    feature_columns.append(key)
    if RESCALE:
        test_df[key] = minmax_scale(test_df[key].values.astype(np.float32), feature_range=MIN_MAX, axis=0)
        train_df[key] = minmax_scale(train_df[key].values.astype(np.float32), feature_range=MIN_MAX, axis=0)
    
if RESCALE:
    train_df.head()


# In[ ]:


from catboost import CatBoostClassifier, Pool

#make a model
def build_model():

    model = CatBoostClassifier(loss_function="CrossEntropy",
                               eval_metric="AUC",
                               learning_rate=0.01,
                               iterations=40000,
                               random_seed=42,
                               od_type="Iter",
                               depth=8,
                               border_count=32,
                               early_stopping_rounds=700,
                               task_type = "GPU",
                               logging_level='Verbose')
    return model


# In[ ]:


FOLDS = 5

# get the train data
X = train_df[feature_columns].values
y = np.array(train_df['target'].values)

#k folds, saving best validation
splits = list(StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42).split(X, y))
models = []
for idx, (train_idx, val_idx) in enumerate(splits):
    print("Beginning fold {}".format(idx+1))
    train_X, train_y, val_X, val_y = X[train_idx], y[train_idx], X[val_idx], y[val_idx]

    model = build_model()
    model.fit(train_X, train_y, eval_set=(val_X,val_y))
    models.append(model)


# In[ ]:


preds_test = []
X = test_df[feature_columns].values

for i in range(FOLDS):
    pred = models[i].predict_proba(X)
    preds_test.append(pred)


# In[ ]:


preds = np.mean(np.array(preds_test),axis=0)


# In[ ]:


#TODO, build this better target ends up being a list of one for each row
submission_df = pd.DataFrame(data={'ID_code': list(test_df['ID_code'].values), 'target': list(preds)})
submission_df['target'] = submission_df['target'].apply(lambda x: float(x[0])).astype('float')
submission_df[submission_df['target'] > 0.5].head(10)


# In[ ]:


submission_df.to_csv('submission.csv', index=False)

