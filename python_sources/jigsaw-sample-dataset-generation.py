#!/usr/bin/env python
# coding: utf-8

# Objective of this kernel:
# 
# * Extract a 10% sample of the training set to be used for experimenting with different models
# * Extract a different 10% sample of the training set to be used as a test set for evaluating above models
# * Create a dataset with the new smaller train & test files.

# In[ ]:


get_ipython().system('ls ../input')


# In[ ]:


import pandas as pd


# In[ ]:


train_df = pd.read_csv('../input/train.csv')


# In[ ]:


train_df


# In[ ]:


size = len(train_df)
sample_size = size // 10
size, sample_size


# In[ ]:


train_test_sample_df = train_df.sample(2*sample_size, random_state=999)


# In[ ]:


train_sample_df = train_test_sample_df[:sample_size]
test_sample_df = train_test_sample_df[sample_size:]


# In[ ]:


train_sample_df


# In[ ]:


test_sample_df


# In[ ]:


train_sample_df.to_csv('train_sample.csv', index=False)


# In[ ]:


get_ipython().system('head train_sample.csv')


# In[ ]:


test_sample_df.to_csv('test_sample.csv', index=False)


# In[ ]:


get_ipython().system('head test_sample.csv')


# In[ ]:


get_ipython().system('cp ../input/test.csv .')
get_ipython().system('cp ../input/sample_submission.csv .')
get_ipython().system('ls ')


# In[ ]:





# In[ ]:




