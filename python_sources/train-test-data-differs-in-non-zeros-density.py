#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt 


# In[ ]:


train = pd.read_csv('../input/train.csv'); train = train[train.columns[2:]]; train_row_nzr = train.astype(bool).sum(axis=1) / train.shape[1]; train_col_nzr = train.astype(bool).sum(axis=0) / train.shape[0]
test = pd.read_csv('../input/test.csv'); test = test[test.columns[1:]]; test_row_nzr = test.astype(bool).sum(axis=1) / test.shape[1]; test_col_nzr = test.astype(bool).sum(axis=0) / test.shape[0];

train.to_sparse(0.).density, test.to_sparse(0.).density


# In[ ]:


fig, axarr = plt.subplots(1,2, figsize=(15, 6))
fig.suptitle('Ratio of non-zeros per row')
train_row_nzr.plot.hist(bins=100, ax=axarr[0], title='train.csv')
test_row_nzr.plot.hist(bins=100, ax=axarr[1], title='test.csv')

train_row_nzr.max(), test_row_nzr.max()


# In[ ]:


fig, axarr = plt.subplots(1,2, figsize=(15, 6))
fig.suptitle('Ratio of non-zeros per column')
train_col_nzr.plot.hist(bins=100, ax=axarr[0], title='train.csv')
test_col_nzr.plot.hist(bins=100, ax=axarr[1], title='test.csv')

train_col_nzr.max(), test_col_nzr.max()


# Note that we have quite a number of training samples with over a ratio of 0.2 non-zero features in the training set, while **all** of the testing samples has ratio less than 0.2. Similarly column-wise.
# 
# Furthermore, the density of sparse matrix for train and test features is 0.031 and 0.014 respectively.
# 
# Suspect that there is a dropout of 0.5 in test data features, any thought?

# In[ ]:




