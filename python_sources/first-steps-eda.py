#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import pyarrow.parquet as pq
import os
print(os.listdir("../input"))

import matplotlib.pyplot as plt
import seaborn as sns

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().run_cell_magic('time', '', "# Read in 5 signals\n# Each column contains one signal\nsubset_train = pq.read_pandas('../input/train.parquet', columns=[str(i) for i in range(5)]).to_pandas()")


# In[ ]:


subset_train.info()


# In[ ]:


subset_train.head()


# In[ ]:


subset_train.tail()


# In[ ]:


get_ipython().run_cell_magic('time', '', "meta_train = pd.read_csv('../input/metadata_train.csv')\nmeta_test = pd.read_csv('../input/metadata_test.csv')")


# In[ ]:


meta_train.head()


# In[ ]:


meta_test.head()


# In[ ]:


meta_train.isna().sum()


# In[ ]:


meta_test.isna().sum()


# In[ ]:


meta_train['signal_id']


# In[ ]:


meta_test['signal_id']


# In[ ]:


meta_train['id_measurement'] # ID code for a trio of signals


# In[ ]:


meta_test['id_measurement']


# ### Train and test signal_ids and id_measurements are sequential

# In[ ]:


sns.countplot(meta_train['target']); # Quite unbalanced, 1 => fault in the power line


# In[ ]:


meta_train['phase'].value_counts() # 3 phase values


# In[ ]:


meta_test['phase'].value_counts() # 3 phase values


# In[ ]:


sns.countplot(meta_train['phase']);


# In[ ]:


sns.countplot(meta_test['phase']);


# In[ ]:


subset_train.head()


# In[ ]:


get_ipython().run_cell_magic('time', '', "subset_test = pq.read_pandas('../input/test.parquet', columns=[str(i) for i in range(8712,8717)]).to_pandas()")


# In[ ]:


subset_test.shape


# In[ ]:


subset_test.head()


# In[ ]:


subset_test.tail()


# In[ ]:


meta_train.head()


# In[ ]:


# Let's look at some of the signals

fig, ((ax1, ax2), (ax3,ax4)) = plt.subplots(2,2, figsize=(12,12))
ax1.hist(subset_train['0'], bins=100);
ax2.hist(subset_train['1'], bins=100)
ax3.hist(subset_train['2'], bins=100)
ax4.hist(subset_train['3'], bins=100);


# In[ ]:


# How well does just the phase reflect the target?
meta_train.corr()['target']
# Not very well it seems


# .. to be continued
