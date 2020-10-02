#!/usr/bin/env python
# coding: utf-8

# In[ ]:




import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
df1 = pd.read_csv('../input/blendscore/submission2.csv')
df2 = pd.read_csv('../input/blendscore/submission_0.9952185750007629.csv')
df1['label'] = .8*df1['label'] + .2*df2['label']
df1.head()


# In[ ]:


df1.to_csv('sub.csv',index=False)

