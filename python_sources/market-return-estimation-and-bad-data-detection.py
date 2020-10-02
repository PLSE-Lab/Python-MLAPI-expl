#!/usr/bin/env python
# coding: utf-8

# Estimation of an average market return used for calculation of returnsOpenPrevMktres1

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from kaggle.competitions import twosigmanews

# Read data but just once
if 'env' not in globals():
    env = twosigmanews.make_env()
    (market_train_df, news_train_df) = env.get_training_data()


# # Data Preparation

# In[ ]:


df = market_train_df[[
    'time',
    'assetCode',
    'open',
    'returnsOpenPrevRaw1', 
    'returnsOpenPrevMktres1',
    'returnsOpenNextMktres10',
]].copy()
df = df[~df.returnsOpenPrevMktres1.isnull() & ~df.returnsOpenPrevRaw1.isnull()]


# Finding the closes value for **returnsOpenPrevMktres1** - then we may use **returnsOpenPrevRaw1** as an estimation

# In[ ]:


df['returnsOpenPrevMktres1_abs'] = df['returnsOpenPrevMktres1'].abs()
df['returnsOpenPrevMktres1_abs_min'] = df.groupby('time').returnsOpenPrevMktres1_abs.transform('min')


# Filtering

# In[ ]:


df_min = df[df.returnsOpenPrevMktres1_abs == df.returnsOpenPrevMktres1_abs_min]


# In[ ]:


df_min.head()


# Sanity check if we have only one row per day

# In[ ]:


(df_min.groupby('time').size()==1).all()


# # Graphs

# How it looks like

# In[ ]:


df_min.groupby('time').returnsOpenPrevRaw1.mean().plot(figsize=(15,6))


# Graphs by year

# In[ ]:


for y in df_min.time.dt.year.unique():
    df_min[df_min.time.dt.year==y].groupby('time').returnsOpenPrevRaw1.mean().plot(title=str(y), figsize=(15,6))
    plt.show()


# # Zero plateau analysis

# Do we have data in July 2016? yes

# In[ ]:


df_min[(df_min.time>='2016-07-01') & (df_min.time<'2016-08-01')]


# It looks like EBRYY.OB is an index, lets have a list assets** with 0 returnsOpenPrevRaw1

# In[ ]:


df_min[df_min.returnsOpenPrevRaw1==0].groupby('assetCode').size().sort_values(ascending=False).head(10)


# Total amount of records of 'PGN.N', 'EBRYY.OB'

# In[ ]:


market_train_df[market_train_df.assetCode.isin(['PGN.N', 'EBRYY.OB'])].groupby('assetCode').size()


# Amount of unique open values

# In[ ]:


market_train_df[market_train_df.assetCode.isin(['PGN.N', 'EBRYY.OB'])].groupby('assetCode').open.nunique()


# In[ ]:


market_train_df[market_train_df.assetCode == 'PGN.N']


# In[ ]:


market_train_df[market_train_df.assetCode == 'EBRYY.OB']


# In[ ]:


market_train_df[market_train_df.assetName=='Unknown'].assetCode.unique()


# # Filtering strange data

# Let's filter 'Unknown' and PGN.N and plot new data

# In[ ]:


df_filtered = df[~df.assetCode.isin(market_train_df[market_train_df.assetName=='Unknown'].assetCode.unique())]
df_filtered = df_filtered[df_filtered.assetCode != 'PGN.N']


# In[ ]:


df_filtered['returnsOpenPrevMktres1_abs'] = df_filtered['returnsOpenPrevMktres1'].abs()
df_filtered['returnsOpenPrevMktres1_abs_min'] = df_filtered.groupby('time').returnsOpenPrevMktres1_abs.transform('min')


# In[ ]:


df_min_filtered = df_filtered[df_filtered.returnsOpenPrevMktres1_abs == df_filtered.returnsOpenPrevMktres1_abs_min]


# New Graphs

# In[ ]:


df_min_filtered.groupby('time').returnsOpenPrevRaw1.mean().plot(figsize=(15,6))


# In[ ]:


for y in df_min_filtered.time.dt.year.unique():
    df_min_filtered[df_min_filtered.time.dt.year==y].groupby('time').returnsOpenPrevRaw1.mean().plot(title=str(y), figsize=(15,6))
    plt.show()


# In[ ]:




