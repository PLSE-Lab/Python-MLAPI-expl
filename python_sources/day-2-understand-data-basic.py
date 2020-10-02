#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from kaggle.competitions import twosigmanews
# You can only call make_env() once, so don't lose it!
env = twosigmanews.make_env()
print('Done!')


# In[ ]:


(market_train_df, news_train_df) = env.get_training_data()
market_train_df = market_train_df.tail(100_000)
news_train_df = news_train_df.tail(300_000)
market_train_df.shape, news_train_df.shape


# In[ ]:


import lightgbm as lgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import chain

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


news_cols_agg = {
    'urgency': ['min', 'count'],
    'takeSequence': ['max'],
    'bodySize': ['min', 'max', 'mean', 'std'],
    'wordCount': ['min', 'max', 'mean', 'std'],
    'sentenceCount': ['min', 'max', 'mean', 'std'],
    'companyCount': ['min', 'max', 'mean', 'std'],
    'marketCommentary': ['min', 'max', 'mean', 'std'],
    'relevance': ['min', 'max', 'mean', 'std'],
    'sentimentNegative': ['min', 'max', 'mean', 'std'],
    'sentimentNeutral': ['min', 'max', 'mean', 'std'],
    'sentimentPositive': ['min', 'max', 'mean', 'std'],
    'sentimentWordCount': ['min', 'max', 'mean', 'std'],
    'noveltyCount12H': ['min', 'max', 'mean', 'std'],
    'noveltyCount24H': ['min', 'max', 'mean', 'std'],
    'noveltyCount3D': ['min', 'max', 'mean', 'std'],
    'noveltyCount5D': ['min', 'max', 'mean', 'std'],
    'noveltyCount7D': ['min', 'max', 'mean', 'std'],
    'volumeCounts12H': ['min', 'max', 'mean', 'std'],
    'volumeCounts24H': ['min', 'max', 'mean', 'std'],
    'volumeCounts3D': ['min', 'max', 'mean', 'std'],
    'volumeCounts5D': ['min', 'max', 'mean', 'std'],
    'volumeCounts7D': ['min', 'max', 'mean', 'std']
}


# In[ ]:


news_train_df['time'] = (news_train_df['time'] - np.timedelta64(22,'h')).dt.ceil('1D')
market_train_df['time'] = market_train_df['time'].dt.floor('1D')


# In[ ]:


news_train_df['assetCodes'].head()


# In[ ]:


news_train_df['assetCodes'] = news_train_df['assetCodes'].str.findall(f"'([\w\./]+)'") 


# In[ ]:


news_train_df['assetCodes'].head()


# In[ ]:


assetCodes_expanded = list(chain(*news_train_df['assetCodes']))
assetCodes_index = news_train_df.index.repeat( news_train_df['assetCodes'].apply(len) )
assert len(assetCodes_index) == len(assetCodes_expanded)


# In[ ]:


df_assetCodes = pd.DataFrame({'level_0': assetCodes_index, 'assetCode': assetCodes_expanded})


# In[ ]:


news_cols = ['time', 'assetCodes'] + sorted(news_cols_agg.keys())


# In[ ]:


news_train_df_expanded = pd.merge(df_assetCodes, news_train_df[news_cols], left_on='level_0', right_index=True, suffixes=(['','_old']))
news_train_df_expanded.head()


# In[ ]:


del news_train_df, df_assetCodes
news_train_df_aggregated = news_train_df_expanded.groupby(['time', 'assetCode']).agg(news_cols_agg)
del news_train_df_expanded


# In[ ]:


news_train_df_aggregated = news_train_df_aggregated.apply(np.float32)
news_train_df_aggregated.columns = ['_'.join(col).strip() for col in news_train_df_aggregated.columns.values]
news_train_df_aggregated.columns


# In[ ]:


market_train_df = market_train_df.join(news_train_df_aggregated, on=['time', 'assetCode'])
del news_train_df_aggregated
x=market_train_df


# In[ ]:




