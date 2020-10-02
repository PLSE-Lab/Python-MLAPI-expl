#!/usr/bin/env python
# coding: utf-8

# @sggpls had mentioned he was able to get 1.5x CV using [LR traned row-wise](https://www.kaggle.com/c/santander-value-prediction-challenge/discussion/61189#357214). That seemed worth investigating, perhaps we can featurize it and throw it into our model. Currently, our models only use the train values and stats. Sergey's LR used the indices of the non-zero values. Let's give it a shot.

# In[ ]:


import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt 
from multiprocessing import Pool


# In[ ]:


def get_data():
    print('Reading data')
    data = pd.read_csv('../input/train.csv')
    test = pd.read_csv('../input/test.csv')
    
    print('Train shape ', data.shape, ' Test shape ', test.shape)
    return data, test


# In[ ]:


data, test = get_data()

del data['ID']

# Log space all, including target. LR won't work in the skewed raw space
data = np.log1p(data)
target = data.target.values
del data['target']

sub = test[['ID']].copy()
del test['ID']


# In[ ]:


# True. Target is nonzero values. Data is indices of nonzero features...
tv = data.values

def min_it(lindex, return_data=False):
    lr = LinearRegression(n_jobs=1)
    
    preds  = np.zeros(tv.shape[0])
    errors = np.ones(tv.shape[0])
    for i in range(tv.shape[0]):
        nzv = tv[i]
        indices = np.array(np.nonzero(nzv > 0) [0]).reshape(-1,1)
        nzv     = nzv[nzv>0]

        # predict next value in time series?
        # what should be index? len(nzv)? len(dset)? something else?
        # last_index = np.array([nzv.shape[0]]*2).reshape(-1,1)
        # last_index = np.array([data.shape[1]//2 + nzv.shape[0]*4 ]).reshape(-1,1)
        last_index = np.array([lindex]).reshape(-1,1)

        if nzv.shape[0] == 0:
            #contigency -- use mean or something else. just put 0
            preds[i] = np.mean(nzv)
            print('WTX')
            continue

        lr.fit(indices,nzv)
        pred      = lr.predict(last_index)
        preds[i]  = pred[0]
        errors[i] = mean_squared_error([target[i]], pred) ** .5

    score = mean_squared_error(target, preds) ** .5
    return score, preds, errors


# In[ ]:


# # If you want to run the full thing... don't try in kaggle kernel:
# pool    = Pool(processes=11) #12 core
# results = pool.map(min_it, range(1,data.shape[1]*2))


# In[ ]:


score, preds, errors = min_it(data.shape[1]//2, return_data=True)


# In[ ]:


'RMSLE', score


# In[ ]:


errors[:100]


# The min result after running the optimization was **1.6595739169729133**. That can't possible be the best we can perform. For sure, the number of non-zeros must play a role. I know because I scored 1.6586905090389363 CV by using the row dependent formula `data.shape[1]//2 + nzv.shape[0]*4`.
# 
# By changing the formula in the method to: `last_index = np.array([lindex + nzv.shape[0]*4 ]).reshape(-1,1)`
# 
# And re-running the optimization again for `results = pool.map(min_it, range(2000,3000))`,
# 
# The following chart is produced:
# 
# ![Opt](https://image.ibb.co/d9ojmy/download.png)
# 
# 514+2000 being passed in produces the minimal value of 1.658632 RMSLE.

# **TODO**
# 
# * Brute force the opimal avg index to use for prediction
# * Investigate train[:100] independently, and find the optimal index. Is there a way to predict the optimal index to use for target prediction using the non-zero indices, row stats, or non-zero values directly?
# * Investigate if dropping std()==0 columns improves scores, as this would affect the column indices.
# * Add predicted LR score, and LR-Confidence, trained using BLL or Regression, to current best scoring model (1.37) and evaluate
# * Explore introducing, for example, [:100] non-zero indices as inputs into GBDT models...

# In[ ]:




