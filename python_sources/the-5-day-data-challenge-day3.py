#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from scipy.stats import ttest_ind
import os
# print(os.listdir("../input"))


# In[2]:


df = pd.read_csv('../input/cereal.csv')


# In[3]:


df.head()


# In[4]:


hot = df['sugars'][df['type']== 'H']
cold = df['sugars'][df['type']== 'C']
ttest_ind(hot, cold, equal_var=False)


# In[12]:


plt.hist(hot, label='Hot')
plt.hist(cold, label='Cold', alpha=0.5)
plt.title('Sugar content of hot and cold cereals')


# In[ ]:




