#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('ls ../input/combining-your-model-with-a-model-without-o-0d991a/')


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


df1 = pd.read_csv('../input/combining-your-model-with-a-model-without-o-0d991a/combining_submission.csv')
df2 = pd.read_csv('../input/simple-lightgbm-without-blending/submission.csv')


# In[ ]:


df2.head()


# In[ ]:


df1['target'] = .5*df1['target'] + .5*df2['target']


# In[ ]:


df1.head()


# In[ ]:


df1.to_csv('ens.csv',index=False)


# In[ ]:




