#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[2]:


import pandas as pd


# In[6]:


train = pd.read_csv('../input/application_train.csv')
test = pd.read_csv('../input/application_test.csv')


# Seems that there is some interesting sampling method apply on train/test set, notice the difference of x-axis, but the pattern is similar for both dataset

# In[17]:


import matplotlib.pyplot as plt
plt.plot(train.SK_ID_CURR.diff(),'.')
plt.plot(test.SK_ID_CURR.diff(),'.')


# In[13]:


train.SK_ID_CURR.head()


# In[14]:


train.SK_ID_CURR.head().diff()


# In[16]:


plt.plot(train.SK_ID_CURR.diff(),'.')


# In[15]:


plt.plot(test.SK_ID_CURR.diff(),'.')


# In[ ]:




