#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[2]:


dtypes = {
    'ip'            : 'uint32',
    'app'           : 'uint16',
    'device'        : 'uint16',
    'os'            : 'uint16',
    'channel'       : 'uint16',
    'is_attributed' : 'uint8',
    'click_id'      : 'uint32'
}

print('Loading the test data...')
test = pd.read_csv("../input/test.csv", dtype=dtypes)


# In[3]:


print("The percentage of not ordered index is {0}".format(np.mean(test.index.values != test.click_id.values)))
test[test.index.values != test.click_id.values]


# Simulating effect: Messing the order of random 0,1 with the same frequency as test

# In[4]:


N_SIM = 2500000
results = np.zeros((N_SIM)).astype(np.int8)
THRESH = 0.0025
np.random.seed(42)
results = (np.random.uniform(size=N_SIM) < THRESH).astype(int)


# Creating disorder same as in the test

# In[5]:


predictions = results[test["click_id"].values[:N_SIM]] 


# In[6]:


from sklearn.metrics import roc_auc_score
roc_auc_score(results, predictions)


# AUC goes down from 1.0 to 0.92665
# Watch your ids :)
