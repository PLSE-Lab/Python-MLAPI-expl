#!/usr/bin/env python
# coding: utf-8

# # Overview
# 
# - Based on [Ensemble Starter](https://www.kaggle.com/khoongweihao/ensemble-starter)
# - Implements a stochastic ensemble
# - Starter kernel for ensembling submission files
# - As reinforced by @arashnic, ensembling should come in later into the competition, if you have several good models to test diversity on performance!
# - Data for ensemble can be found at: https://www.kaggle.com/khoongweihao/m5-forecasting-ensemble-data
# - The submissions used here for ensembling are from the following kernels:
#     - https://www.kaggle.com/ragnar123/very-fst-model
#     - https://www.kaggle.com/zmnako/lgbm-update-0-85632
#     - https://www.kaggle.com/siavrez/simple-eda-simpleexpsmoothing
#     - https://www.kaggle.com/tnmasui/m5-forecasting-lstm-w-custom-data-generator
#     
# Happy Kaggling! :)

# In[ ]:


import numpy as np 
import pandas as pd 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import random


# # Data Import

# In[ ]:


dfs = list()
for N in ['060288', '060389', '063384', '064127', '084855', '084901', '084918', '085051', '085405', '085632', '085758']:
    dfs.append(pd.read_csv(f"../input/m5-forecasting-ensemble-data/submission_{N}.csv"))


# In[ ]:


dfs[0].head()


# In[ ]:


# sort each submission by id
for i in range(len(dfs)):
    dfs[i].sort_values('id',inplace=True)
ids = dfs[0]['id']


# In[ ]:


# flatten the submissions for easier combination
for i in range(len(dfs)):
    dfs[i] = np.concatenate(dfs[i].drop('id',axis=1).values.reshape([30490*28*2,1]))


# # The Ensemble

# In[ ]:


# probabilities for the ensemble should be derived from the LB scores but we don't have these in this case
LBs = np.asarray([1 for i in range(len(dfs))])
probs = LBs / sum(LBs)
probs = np.cumsum(probs)


# In[ ]:


# select a submission at random for each prediction based on the probs
rs = [random.random() for i in range(30490*28*2)]
def f(rs): 
    return np.argmax(rs<probs)
rcols = np.vectorize(f)(rs)

y_pred = np.select([rcols==0, rcols==1, rcols==2, rcols==3, rcols==4, rcols==5, rcols==6, rcols==7, rcols==8, rcols==9, rcols==10], dfs)
y_pred[np.isnan(y_pred)] = 0


# # Make Submission

# In[ ]:


test = pd.DataFrame.from_dict({'id': np.repeat(ids,28), 
                               'date': np.tile(np.arange(np.datetime64("2016-04-25"),np.datetime64("2016-05-23")),30490*2),
                               'demand': y_pred})

predictions = test.pivot(index = 'id', columns = 'date', values = 'demand').reset_index()
predictions.columns = ['id'] + ['F' + str(i + 1) for i in range(28)]

predictions.to_csv('submission.csv', index = False)

