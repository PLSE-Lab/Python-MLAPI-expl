#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet


# In[ ]:


get_ipython().run_line_magic('pinfo2', 'ElasticNet')


# In[ ]:


PATH = Path('../input')


# In[ ]:


train = pd.read_csv(PATH/'train.csv')
test = pd.read_csv(PATH/'test.csv').drop(columns=['id'])


# In[ ]:


train_Y = train['target']
train_X = train.drop(columns=['target', 'id'])


# In[ ]:


best_parameters = {
    'alpha': 0.2,
    'l1_ratio': 0.31,
    'precompute': True,
    'selection': 'random',
    'tol': 0.001, 
    'random_state': 2
}


# In[ ]:


net = ElasticNet(**best_parameters)
net.fit(train_X, train_Y)


# In[ ]:


sub = pd.read_csv(PATH/'sample_submission.csv')
sub['target'] = net.predict(test)


# In[ ]:


sub.head()


# In[ ]:


sub.to_csv('submission.csv', index=False)


# In[ ]:


FileLink('submission.csv')


# In[ ]:




