#!/usr/bin/env python
# coding: utf-8

# Plain stock code don't give usefull information, we can get better feature and groups with exchange information, this RIC Stock code parser do this
# 
# If you liked, please up vote =)

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.metrics import accuracy_score
from kaggle.competitions import twosigmanews


# In[ ]:


env = twosigmanews.make_env()
(market_train, _) = env.get_training_data()


# In[ ]:


from IPython.display import display

market_train['assetCode_asset'] = market_train['assetCode']
market_train['assetCode_exchange'] = market_train['assetCode']
tmp_map_a, tmp_map_b = {}, {}
for i in market_train['assetCode'].unique():
    a,b = i.split('.')
    tmp_map_a[i] = a
    tmp_map_b[i] = b
market_train['assetCode_asset']=market_train['assetCode_asset'].map(tmp_map_a)
market_train['assetCode_exchange']=market_train['assetCode_exchange'].map(tmp_map_b)


# In[ ]:


display(market_train.head())


# In[ ]:


display(market_train['assetCode_exchange'].unique())


# Now, do some magic :) group by exchange as a new feature, create a exchange index to help models with "systematic risk information"
# 
# Good luck =)

# In[ ]:




