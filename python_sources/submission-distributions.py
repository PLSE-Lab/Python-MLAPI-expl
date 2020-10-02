#!/usr/bin/env python
# coding: utf-8

# # Submission distributions
# 
# One of the interesting things I've noticed about this competition is how different my prediction of f1 score is from the LB score, in this kernel I wanted to examine this a bit further
# 
# **TL; DR:** the training set and test set have slightly different distrubutions of target values. You should take this into account when you train your model

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


from sklearn.metrics import f1_score


# In[ ]:


df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')


# In[ ]:


df_train.target.value_counts(normalize=True)


# In[ ]:


len(df_train)


# First, lets do the dumbest thing and label everything as positive in our training set, then see what f1 score we get

# In[ ]:


f1_score(df_train.target.values, np.ones(len(df_train)))


# Let's now do the same for the test set. I submitted this in version 1 of this kernel

# In[ ]:


df_test['prediction'] = 1
df_test = df_test[['qid', 'prediction']]

df_test.to_csv('submission.csv', index=False)


# This gets a score of `0.113` - which obviously means the distribution of target values is different between the test and training sets, but how different?
# 
# well since the f1 score is defined as
# 
# $$f1 = {{2 * precision * recall}\over{precision + recall}}$$
# 
# which is the same as
# 
# $$ f1 = {{2 TP}\over{2TP + FP + FN}}$$
# 
# because we labelled everything as positive, we know $FP = \sum - TP$ (where $\sum$ is the total number of results) and $FN = 0$, this becomes
# 
# $$ f1 = {{2 TP}\over{TP + \sum}}$$
# 
# after some rearraging we get
# 
# $${TP\over{\sum}} = {{f1}\over{2 - f1}}$$
# 
# ie we can relate the ratio of positive labells to the f1 score
# 
# Doing this we get

# In[ ]:


positive_ratio = 0.113 / (2 - 0.113)

print(positive_ratio)


# the training set positive ratio is `0.06187` as shown above
# 
# This isn't a huge difference, but if there's a similar difference between the test set and the private test set it could really mix up the leaderboard!
# 
# 
# ## Appendix: sanity check :)

# In[ ]:


0.11653 / (2 - 0.11653)


# phew :)

# In[ ]:




