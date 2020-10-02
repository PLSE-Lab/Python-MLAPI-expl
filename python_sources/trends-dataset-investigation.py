#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


# # fnc.csv

# In[ ]:


directory = '/kaggle/input/trends-assessment-prediction/'
df_icn = pd.read_csv(directory + 'ICN_numbers.csv')
print(df_icn.shape)
print(df_icn['ICN_number'].unique().shape[0] == df_icn.shape[0])
print('The minimun ICN number:',  min(df_icn['ICN_number']))
print('The maximum ICN number:',  max(df_icn['ICN_number']))
df_icn.head(10)
# There are 53 unique ICN numbers. But they are not from 0 to 53, they are sparse


# # loading.csv

# In[ ]:


loading = pd.read_csv(directory + 'loading.csv')
print(loading.shape)
loading.columns
# it has 11,754 instances (rows) and 27 columns
loading.head(10)


# # reveal_id_site2

# In[ ]:


reveal = pd.read_csv(directory + 'reveal_ID_site2.csv')
print(len(reveal['Id'].unique()))
reveal.head(10)
# This dataset has 510 unique id's


# # train_scores
# 

# In[ ]:


train = pd.read_csv(directory + 'train_scores.csv')
train.head(10)
# The training has the variables (5) for 5,877 unique id's


# # Sample_submission

# In[ ]:


submission = pd.read_csv(directory + 'sample_submission.csv')
print('Number of unique ids in the test set:', len(submission)/5)
submission.head(10)


# # Conclussion:
# * There are 5877 ids for the training set, and another 5877 for the test set
# * Loading: Contains the IC information of all the ids (training and test set)
# * reveal_ID_site2 dataset: contains 510 ids, which have been processed in another facilities. All of this ids are in the test set

# In[ ]:




