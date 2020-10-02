#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from itertools import permutations
import time
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.simplefilter('ignore')
sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


get_ipython().run_cell_magic('time', '', "train = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv', index_col='TransactionID')\ntest = pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv', index_col='TransactionID')")


# In[ ]:


#train
tr1 = train[['D{}'.format(i) for i in range(1,16) if i not in [8,9]]].notnull().sum().to_frame()/train.shape[0]
tr1.plot(kind='barh')


# In[ ]:


#test
te1 = test[['D{}'.format(i) for i in range(1,16) if i not in [8,9]]].notnull().sum().to_frame()/test.shape[0]
te1.plot(kind='barh')


# ### Group by value_counts 4+1 columns at once: C1,C2,C13, C14 + D1 (for example), then look at the popular five values data in D2 - if for the five values on the right (in column D2) there is only one value, for example 0 (+ NaNs), then fill NaN with zeros. You can change the depth and number, for example 1 instead of 0, but you need to look at the data and "patterns", if other columns. Iterate (permutations) all possible pairs of columns from cols. You can do several iterations for the same columns.

# In[ ]:


get_ipython().run_cell_magic('time', '', "cols = ['D{}'.format(j) for j in range(1,16) if j not in [8,9]]\n\nr = list(range(0,len(cols)))\nl = list(permutations(r,2))\ndata = pd.concat([train[['C1','C2','C13','C14'] + cols],test[['C1','C2','C13','C14'] + cols]] )\n\nprint(data[data==0].count().sum())\n\nfor q in l:\n\n    data['count'] = 0\n    x = data.groupby(['C1','C2','C13','C14',cols[q[0]]])['count'].count().reset_index()\n    x.sort_values(['count','C1','C2','C13','C14',cols[q[0]]], ascending=False,inplace=True)\n\n    for i in range(0,500):\n        \n           if x.iloc[i,4] == 0:\n\n                if data.loc[(data['C1']==x.iloc[i,0]) & (data['C2']==x.iloc[i,1]) & (data['C13']==x.iloc[i,2]) & (data['C14']==x.iloc[i,3]) & (data[cols[q[0]]]==x.iloc[i,4]),cols[q[1]]].value_counts(dropna=False).shape[0] == 2:\n\n                    if data.loc[(data['C1']==x.iloc[i,0]) & (data['C2']==x.iloc[i,1]) & (data['C13']==x.iloc[i,2]) & (data['C14']==x.iloc[i,3]) & (data[cols[q[0]]]==x.iloc[i,4]),cols[q[1]]].value_counts().shape[0] == 1:\n                        \n                        val = data.loc[(data['C1']==x.iloc[i,0]) & (data['C2']==x.iloc[i,1]) & (data['C13']==x.iloc[i,2]) & (data['C14']==x.iloc[i,3]) & (data[cols[q[0]]]==x.iloc[i,4]),cols[q[1]]].max()\n                   \n                        data.loc[(data['C1']==x.iloc[i,0]) & (data['C2']==x.iloc[i,1]) & (data['C13']==x.iloc[i,2]) & (data['C14']==x.iloc[i,3]) & (data[cols[q[0]]]==x.iloc[i,4]),cols[q[1]]] = val\n                     \n                        if x.iloc[i,5]<100: \n                            \n                              break\n\nprint(data[data==0].count().sum())\n\ntrain = data[:590540]\ntest = data[590540:]\ndel data")


# In[ ]:


#train
tr2 = train[['D{}'.format(i) for i in range(1,16) if i not in [8,9]]].notnull().sum().to_frame()/train.shape[0]
tr1['after filling NaN'] = tr2
tr1.plot(kind='barh')


# In[ ]:


#test
te2 = test[['D{}'.format(i) for i in range(1,16) if i not in [8,9]]].notnull().sum().to_frame()/test.shape[0]
te1['after filling NaN'] = te2
te1.plot(kind='barh')


# #### You can try it yet! =) Just do it!
