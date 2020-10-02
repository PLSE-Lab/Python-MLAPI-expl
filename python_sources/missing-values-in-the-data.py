#!/usr/bin/env python
# coding: utf-8

# # A quick intro into missing values in the data
# 
# The purpose of the kernel is to have a quick guide through missing values

# In[ ]:


import numpy as np # linear algebra
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
#plt.xkcd()


# In[ ]:


import os
print(os.listdir("../input"))


# In[ ]:


train = pd.read_csv('../input/train.csv')
test  = pd.read_csv('../input/test.csv')


# Quick look at the data 

# In[ ]:


train.info()


# # Missing values

# In[ ]:


train_null = train.isnull().sum()
train_null_non_zero = train_null[train_null>0] / train.shape[0]


# In[ ]:


train_null_non_zero


# In[ ]:


sns.barplot(x=train_null_non_zero, y=train_null_non_zero.index)
_ = plt.title('Fraction of NaN values, %')


# > `v2a1`, Monthly rent payment
# 
# This one is unclear, as it already contains 0, which is suspicious on itself
# 
# > `v18q1`, number of tablets household owns
# 
# This one is correlated with *v18q, owns a tablet* and corresponds to no tablet. Safe to fill 0.
# 
# > `rez_esc`, Years behind in school
# 
# This one is unclear, as `rez_esc` contains 0. The safe option can be to fill in -1 or the mean
# 
# > `meaneduc`,average years of education for adults (18+)
# 
# > `SQBmeaned`, meaned squared
# 
# These two are correlated and the fraction of `NaN` is really small, so we can simply fill in 0 or -1

# ### Quick look at missing values in the test data

# In[ ]:


test_null = test.isnull().sum()
test_null_non_zero = test_null[test_null>0] / test.shape[0]


# In[ ]:


sns.barplot(x=test_null_non_zero, y=test_null_non_zero.index)
_ = plt.title('Fraction of NaN values in TEST data, %')


# **Conclusion**: the distribution of missing data is the same in train and test data, which means where will be less trouble

# In[ ]:




