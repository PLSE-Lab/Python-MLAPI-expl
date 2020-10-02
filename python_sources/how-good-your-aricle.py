#!/usr/bin/env python
# coding: utf-8

# In[ ]:




import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
df1 = pd.read_csv('../input/blendpow/ridgeCV_200k_stemming_lemmatizing.csv')
df2 = pd.read_csv('../input/blendpow/submission (1).csv')
df1['log_recommends'] = 0.3*df1['log_recommends'] + 0.5*df2['log_recommends']
df1.head()


# In[ ]:


df1.to_csv('sub.csv',index=False)

