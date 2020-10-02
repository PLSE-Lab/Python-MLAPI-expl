#!/usr/bin/env python
# coding: utf-8

# # Credit to Konstantin Yakovlev for the last code bite.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


data1 = pd.read_csv('../input/m5-more-data-table-and-xgb/submission_lgbm.csv')
data2 = pd.read_csv('../input/m5-dark-magic/submission.csv')


# In[ ]:


for i in range(1,29):
    data1['F'+str(i)] *= 1.04


# In[ ]:


categories = []
for i in range(1, 29):
    categories.append(f'F{i}')


# In[ ]:


sub_col = data1['id']


# In[ ]:


all_cols = pd.DataFrame({})
all_cols['id'] = sub_col
all_cols[categories] = 0.60*data1[categories] + 0.40*data2[categories]


# In[ ]:


all_cols.to_csv('sub.csv', index=False)

