#!/usr/bin/env python
# coding: utf-8

# The purpose of this notebook is to find correlations between data (open, returns) in the market dataset.

# In[ ]:


import numpy as np
import pandas as pd

from kaggle.competitions import twosigmanews

# Read data but just once
if 'env' not in globals():
    env = twosigmanews.make_env()
    (market_train_df, news_train_df) = env.get_training_data()


# In[ ]:


market_train_df.head().T


# Preparing dataframe copy (snce we may read data onlyone and I don't want to waste time to re-run data reading)

# In[ ]:


df = market_train_df[[
    'assetCode',
    'open', 
    'returnsOpenPrevRaw1', 
    'returnsOpenPrevMktres1', 
    'returnsOpenPrevRaw10', 
    'returnsOpenPrevMktres10', 
    'returnsOpenNextMktres10']].copy()


# # 1 Day analysis

# In[ ]:


df['prevopen'] = df.groupby(['assetCode'])['open'].shift(1)
df['returnsMyOpenPrevRaw1_prev'] = (df.open - df.prevopen) / df.prevopen
df['returnsMyOpenPrevRaw1_current'] = (df.open - df.prevopen) / df.open


# In[ ]:


df[['returnsOpenPrevRaw1', 'returnsMyOpenPrevRaw1_prev', 'returnsMyOpenPrevRaw1_current']].corr()


# 1. So, the right calculation for **returnsOpenPrevRaw1**:
# > **(open - prevopen) / prevopen**[](http://)
# > 
# 
# And values are mostly the same

# In[ ]:


df[['returnsOpenPrevRaw1', 'returnsMyOpenPrevRaw1_prev']].sample(10).T


# Let us check correlation between Raw and Mktres

# In[ ]:


df[['returnsOpenPrevRaw1', 'returnsOpenPrevMktres1']].corr()


# # 10 Days

# In[ ]:


df['prevopen10'] = df.groupby(['assetCode'])['open'].shift(10)
df['returnsMyOpenPrevRaw10'] = (df.open - df.prevopen10) / df.prevopen10
df[['returnsOpenPrevRaw10', 'returnsMyOpenPrevRaw10']].corr()


# In[ ]:


df[['returnsOpenPrevRaw10', 'returnsMyOpenPrevRaw10']].sample(10).T


# In[ ]:


df[['returnsOpenPrevRaw10', 'returnsOpenPrevMktres10']].corr()


# # Mktres Prev/Next Correlation

# In[ ]:


df['nextmktres10_10'] = df.groupby(['assetCode'])['returnsOpenNextMktres10'].shift(10)
df['nextmktres10_11'] = df.groupby(['assetCode'])['returnsOpenNextMktres10'].shift(11)
df['nextmktres10_12'] = df.groupby(['assetCode'])['returnsOpenNextMktres10'].shift(12)


# In[ ]:


df[['returnsOpenPrevMktres10', 'nextmktres10_10', 'nextmktres10_11', 'nextmktres10_12']].corr()


# In[ ]:


df[['returnsOpenPrevMktres10', 'nextmktres10_11']].sample(10).T


# So, **returnsOpenPrevMktres10** and **returnsOpenNextMktres10** are the same but with shift in 11 days

# In[ ]:




