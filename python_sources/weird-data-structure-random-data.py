#!/usr/bin/env python
# coding: utf-8

# Fork from https://www.kaggle.com/brandenkmurray/weird-data-structure-random-data
# 
# This is same distribution version.

# In[ ]:


import pandas as pd
import numpy as np
from pathlib import Path

import seaborn as sns
from matplotlib import pyplot as plt


# In[ ]:


ls ../input


# In[ ]:


tr = pd.read_csv('../input/train.csv').iloc[:, 2:]
te = pd.read_csv('../input/test.csv').iloc[:, 1:]


# In[ ]:


trte = pd.concat([tr, te])


# In[ ]:


trte_mean = trte.mean()
trte_std = trte.std()


# In[ ]:


np.round(np.random.normal(trte_mean[0], trte_std[0], 200000), 4)


# In[ ]:


train = pd.concat([pd.Series(np.round(np.random.normal(trte_mean[i], trte_std[i], 200000), 4)) for i in range(200)], axis=1)
train.columns = ["var_" + str(i) for i in range(200)]

test = pd.concat([pd.Series(np.round(np.random.normal(trte_mean[i], trte_std[i], 200000), 4)) for i in range(200)], axis=1)
test.columns = ["var_" + str(i) for i in range(200)]


# # Check distribution

# ## var_0

# In[ ]:


trte.var_0.hist(bins=100); plt.title('original')


# In[ ]:


train.var_0.hist(bins=100); plt.title('simulated train var_0')


# In[ ]:


test.var_0.hist(bins=100); plt.title('simulated test var_0')


# ## var_1

# In[ ]:


trte.var_1.hist(bins=100); plt.title('original')


# In[ ]:


train.var_1.hist(bins=100); plt.title('simulated train var_1')


# In[ ]:


test.var_1.hist(bins=100); plt.title('simulated test var_1')


# I can't say it's perfect, but lgtm

# In[ ]:


x = train['var_0'].value_counts()
x = x[x==1].reset_index(drop=False)
x.head()


# In[ ]:


candidates = []
for c in train.columns[1:-1]:
    if(train[train[c] == x['index'][0]].shape[0]==1):
        candidates.append(c)
indexes = []
for c in candidates:
    indexes.append(train[train[c] == x['index'][0]].index.values[0])
y = train.iloc[indexes][candidates]
y.head(y.shape[0])


# In[ ]:


candidates = []
for c in test.columns[1:-1]:
    if(test[test[c] == x['index'][0]].shape[0]==1):
        candidates.append(c)
indexes = []
for c in candidates:
    indexes.append(test[test[c] == x['index'][0]].index.values[0])
y = test.iloc[indexes][candidates]
y.head(y.shape[0])


# In[ ]:


candidates = []
for c in train.columns[1:-1]:
    if(train[train[c] == x['index'][1]].shape[0]==1):
        candidates.append(c)
indexes = []
for c in candidates:
    indexes.append(train[train[c] == x['index'][1]].index.values[0])
y = train.iloc[indexes][candidates]
y.head(y.shape[0])


# In[ ]:


candidates = []
for c in test.columns[1:-1]:
    if(test[test[c] == x['index'][1]].shape[0]==1):
        candidates.append(c)
indexes = []
for c in candidates:
    indexes.append(test[test[c] == x['index'][1]].index.values[0])
y = test.iloc[indexes][candidates]
y.head(y.shape[0])


# In[ ]:





# In[ ]:




