#!/usr/bin/env python
# coding: utf-8

# # Benchmark
# This is a very simple benchmark to find out how many `new_whales` are in the test set.

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


train_df = pd.read_csv('../input/train.csv')


# In[ ]:


n_all = train_df.shape[0]
n_new_whale = train_df[train_df['Id'] == 'new_whale'].shape[0]
print("We have {}/{} ({:.2f}%) `new_whale` in the training set.".format(n_new_whale, n_all, n_new_whale/n_all*100))


# Let's see if we get similar distribution if we use a random fraction of the data.

# In[ ]:


res = []
for i in range(10):
    tmp_df = train_df.sample(frac=.2)
    n_all = tmp_df.shape[0]
    n_new_whale = tmp_df[tmp_df['Id'] == 'new_whale'].shape[0]
    res.append(n_new_whale/n_all)
    print("We have {}/{} ({:.2f}%) `new_whale` in the random subset #{}.".format(n_new_whale, n_all, n_new_whale/n_all*100, i))
    
print("Average: {:.4f}, std: {:.4f}".format(np.mean(res), np.std(res)))


# *I assume a similar 'new_whale' distribution between the public (20%) and the private (80%) set.*

# In[ ]:


test_df = pd.read_csv('../input/sample_submission.csv')
test_df.head()


# In[ ]:


test_df['Id'] = 'new_whale'
test_df.head()


# I got `0.276` LB score when I submitted the `new_whale` predictions. For every image I either got `P(1) = 0` or `1` score, see [my MAP@5 explanation kernel](https://www.kaggle.com/pestipeti/explanation-of-map5-scoring-metric) for more details.

# In[ ]:


n_all = test_df.shape[0]
# After submission (public LB score: 0.276)
n_new_whale = 2197
print("We have {}/{} ({:.2f}%) `new_whale` in the test set.".format(n_new_whale, n_all, n_new_whale/n_all*100))


# In[ ]:


test_df.to_csv('new_whale_benchmark.csv', index=False)


# **Thanks for reading.**
