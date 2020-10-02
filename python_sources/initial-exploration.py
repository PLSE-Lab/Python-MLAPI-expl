#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


train_raw = pd.read_csv('../input/train.csv', header=0, index_col='ID')


# In[ ]:


train_raw.info()


# In[ ]:


train_raw['target'].value_counts()


# In[ ]:


string_cols = []

for col in train_raw.columns:
    if train_raw[col].dtype == 'object':
        string_cols.append(col)


# In[ ]:


string_cols


# In[ ]:


for sc in string_cols:
    print('{:4s} {:05d}'.format(sc, train_raw[sc].value_counts().values.size))


# With the exception of `v22`, the text columns might be suitable for dummy indexes

# In[ ]:


sorted(train_raw['v22'].value_counts().index.tolist())


# In[ ]:




