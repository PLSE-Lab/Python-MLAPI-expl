#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import lightgbm as lgb
import pandas as pd
from kaggle.competitions import twosigmanews


# In[ ]:


env = twosigmanews.make_env()

(market_train_df, news_train_df) = env.get_training_data()
market_train, news_train = market_train_df.copy(), news_train_df.copy()


# When i call `news_train.assetName.value_counts()`
# The result show many assetName with value **0**
# Anyone could explain for me? 

# In[ ]:


news_train.assetName.value_counts()


# In[ ]:




