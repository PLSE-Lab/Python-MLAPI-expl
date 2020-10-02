#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import os
import random
import sys
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


SEED = 31
TRIALS = 200
TARGET = 'scalar_coupling_constant'
PREDICTION = 'pred'


# In[ ]:


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
seed_everything(SEED)


# # Eval function

# In[ ]:


def group_mean_log_mae(y_true, y_pred, types, floor=1e-9):
    """
    Fast metric computation for this competition: https://www.kaggle.com/c/champs-scalar-coupling
    Code is from this kernel: https://www.kaggle.com/uberkinder/efficient-metric
    """
    maes = (y_true-y_pred).abs().groupby(types).mean()
    maes = np.log(maes.map(lambda x: max(x, floor)))
    return maes.mean()


# In[ ]:


lasso = pd.read_csv(f'../input/champs-scalar-coupling-lasso/submission.csv')
rf = pd.read_csv(f'../input/champs-scalar-coupling-rf/submission.csv')
xgb = pd.read_csv(f'../input/champs-scalar-coupling-xgb/submission.csv')
lgb = pd.read_csv(f'../input/champs-scalar-coupling-lgb/submission.csv')
keras = pd.read_csv(f'../input/champs-scalar-coupling-keras/submission.csv')
test_sets = [lgb, keras, xgb, rf, lasso]
print(f'lasso={lasso.shape}, rf={rf.shape}, xgb={xgb.shape}, lgb={lgb.shape}, keras={keras.shape}')


# In[ ]:


lasso_train = pd.read_csv(f'../input/champs-scalar-coupling-lasso/train.csv')
rf_train = pd.read_csv(f'../input/champs-scalar-coupling-rf/train.csv')
xgb_train = pd.read_csv(f'../input/champs-scalar-coupling-xgb/train.csv')
lgb_train = pd.read_csv(f'../input/champs-scalar-coupling-lgb/train.csv')
keras_train = pd.read_csv(f'../input/champs-scalar-coupling-keras/train.csv')
train_sets = [lgb_train, keras_train, xgb_train, rf_train, lasso_train]
print(f'Train sets\nlasso={lasso_train.shape}, rf={rf_train.shape}, xgb={xgb_train.shape}, lgb={lgb_train.shape}, keras={keras_train.shape}')


# # Trials

# In[ ]:


get_ipython().run_cell_magic('time', '', "\ndef weights(n, min_weight=0.01, max_allocation=0.5):\n    if n < 1:\n        raise ValueError('n must not be less than 1')\n    remainder = 1 - (n * min_weight)\n    if remainder <= 0:\n        raise ValueError('min weight exceeds budget of 1')\n    res = []\n    for _ in range(n - 1):\n        a = random.uniform(0.01, max_allocation) * remainder\n        res.append(a + min_weight)\n        remainder -= a\n    res.append(remainder + min_weight)\n    return res\n\n\ndef trial(train_sets, prediction_column, target_column):\n    ws = weights(len(train_sets), min_weight=0.05, max_allocation=0.9)\n    df = train_sets[0].copy()\n    df[prediction_column] = 0\n    for i, t in enumerate(train_sets):\n        df[prediction_column] += t[prediction_column] * ws[i]\n    score = group_mean_log_mae(df[target_column], df[prediction_column], df['type'])\n    return score, ws\n\n\nbest = sys.maxsize\nbest_weights = []\nfor _ in range(TRIALS):\n    score, ws = trial(train_sets=train_sets, prediction_column=PREDICTION, target_column=TARGET)\n    if score < best:\n        best = score\n        best_weights = ws\n        \nprint(f'best={best:.4f}')\nprint(f'''best weights (sum={sum(best_weights)})\n  lgb={best_weights[0]:.4f}\n  keras={best_weights[1]:.4f}\n  xgb={best_weights[2]:.4f}\n  rf={best_weights[3]:.4f}\n  lasso={best_weights[4]:.4f}\n''')")


# In[ ]:


submission = test_sets[0].copy()
submission[TARGET] = 0
for i, t in enumerate(test_sets):
    submission[TARGET] += t[TARGET] * best_weights[i]
submission.head()


# In[ ]:


submission.to_csv('submission.csv', index=False)
print(os.listdir("."))

