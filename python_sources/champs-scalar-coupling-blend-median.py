#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


files = [
    f'../input/champs-scalar-coupling-lasso/submission.csv',
    f'../input/champs-scalar-coupling-rf/submission.csv',
    f'../input/champs-scalar-coupling-xgb/submission.csv',
    f'../input/champs-scalar-coupling-lgb/submission.csv',
    f'../input/champs-scalar-coupling-keras/submission.csv'
]

print(f'len(files)={len(files)}')


# In[ ]:


def get_median(files):
    outs = [pd.read_csv(f, index_col=0) for f in files]
    concat_sub = pd.concat(outs, axis=1, sort=True)
    preds = concat_sub.median(axis=1).values
    return preds


preds = get_median(files)
print(f'len(preds)={len(preds)}')


# In[ ]:


#submission = pd.DataFrame({'id': test_sets[0].index, 'scalar_coupling_constant': preds})
submission = pd.read_csv(f'../input/champs-scalar-coupling/sample_submission.csv')
submission['scalar_coupling_constant'] = preds
submission.head()


# In[ ]:


submission.to_csv('submission.csv', index=False)
print(os.listdir("."))

