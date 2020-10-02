#!/usr/bin/env python
# coding: utf-8

# Was my LB Probe kernel correct?  
# [LB probe -> weights, N of positives, scoring](https://www.kaggle.com/kambarakun/lb-probe-weights-n-of-positives-scoring/output)

# In[ ]:


import pandas as pd


# In[ ]:


df_1 = pd.read_csv('../input/rsna2019-csv/stage_1_sample_submission.csv')
df_2 = pd.read_csv('../input/rsna2019-csv/stage_2_train.csv')


# In[ ]:


set_df_1_id = set(df_1['ID'])
idx         = [True if id_tmp in set_df_1_id else False for id_tmp in df_2['ID']]


# In[ ]:


df_2[idx]['Label'].values.reshape(-1, 6).sum(axis=0)


# My LB Probe of stage-1 was correct!!
# 
# |N of epidural|N of intraparenchymal|N of intraventricular|N of subarachnoid|N of subdural|N of any|
# |-------------|---------------------|---------------------|-----------------|-------------|--------|
# |          384|                 3554|                 2439|             3553|         4670|   10830|
