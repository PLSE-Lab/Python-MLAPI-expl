#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import ks_2samp

import gc


# In[14]:


tr = pd.read_csv('../input/train.csv')
te = pd.read_csv('../input/test.csv')


# In[15]:


te['target'] = -1
tr['i_am_train'] = 1
te['i_am_train'] = 0

tr_te = pd.concat([tr,te],axis=0, sort=True)


# # var_20 and var_155:

# In[16]:


print('Length of unique var_20 :',len(tr_te['var_20'].unique()))
print('Length of unique var_155:',len(tr_te['var_155'].unique()))


# In[19]:


col = 'var_20'

statistic, pvalue = ks_2samp(tr.loc[tr['target']==0, col], tr.loc[tr['target']==1, col])
fig, ax = plt.subplots(1, 1, figsize=(10, 4))
sns.kdeplot(tr.loc[tr['target']==0, col], ax=ax, label='Target == 0')
sns.kdeplot(tr.loc[tr['target']==1, col], ax=ax, label='Target == 1')

ax.set_title('name: {}, statistics: {:.5f}, pvalue: {:5f}'.format(col, statistic, pvalue))
plt.show()


# In[20]:


col = 'var_155'

statistic, pvalue = ks_2samp(tr.loc[tr['target']==0, col], tr.loc[tr['target']==1, col])
fig, ax = plt.subplots(1, 1, figsize=(10, 4))
sns.kdeplot(tr.loc[tr['target']==0, col], ax=ax, label='Target == 0')
sns.kdeplot(tr.loc[tr['target']==1, col], ax=ax, label='Target == 1')

ax.set_title('name: {}, statistics: {:.5f}, pvalue: {:5f}'.format(col, statistic, pvalue))
plt.show()


# # var_198 and var_191:

# In[21]:


print('Length of unique var_198:',len(tr_te['var_198'].unique()))
print('Length of unique var_191:',len(tr_te['var_191'].unique()))


# In[22]:


col = 'var_198'

statistic, pvalue = ks_2samp(tr.loc[tr['target']==0, col], tr.loc[tr['target']==1, col])
fig, ax = plt.subplots(1, 1, figsize=(10, 4))
sns.kdeplot(tr.loc[tr['target']==0, col], ax=ax, label='Target == 0')
sns.kdeplot(tr.loc[tr['target']==1, col], ax=ax, label='Target == 1')

ax.set_title('name: {}, statistics: {:.5f}, pvalue: {:5f}'.format(col, statistic, pvalue))
plt.show()


# In[23]:


col = 'var_191'

statistic, pvalue = ks_2samp(tr.loc[tr['target']==0, col], tr.loc[tr['target']==1, col])
fig, ax = plt.subplots(1, 1, figsize=(10, 4))
sns.kdeplot(tr.loc[tr['target']==0, col], ax=ax, label='Target == 0')
sns.kdeplot(tr.loc[tr['target']==1, col], ax=ax, label='Target == 1')

ax.set_title('name: {}, statistics: {:.5f}, pvalue: {:5f}'.format(col, statistic, pvalue))
plt.show()


# # var_177 and var_88:

# In[24]:


print('Length of unique var_177:',len(tr_te['var_177'].unique()))
print('Length of unique var_88 :',len(tr_te['var_88'].unique()))


# In[25]:


col = 'var_177'

statistic, pvalue = ks_2samp(tr.loc[tr['target']==0, col], tr.loc[tr['target']==1, col])
fig, ax = plt.subplots(1, 1, figsize=(10, 4))
sns.kdeplot(tr.loc[tr['target']==0, col], ax=ax, label='Target == 0')
sns.kdeplot(tr.loc[tr['target']==1, col], ax=ax, label='Target == 1')

ax.set_title('name: {}, statistics: {:.5f}, pvalue: {:5f}'.format(col, statistic, pvalue))
plt.show()


# In[26]:


col = 'var_88'

statistic, pvalue = ks_2samp(tr.loc[tr['target']==0, col], tr.loc[tr['target']==1, col])
fig, ax = plt.subplots(1, 1, figsize=(10, 4))
sns.kdeplot(tr.loc[tr['target']==0, col], ax=ax, label='Target == 0')
sns.kdeplot(tr.loc[tr['target']==1, col], ax=ax, label='Target == 1')

ax.set_title('name: {}, statistics: {:.5f}, pvalue: {:5f}'.format(col, statistic, pvalue))
plt.show()


# # var_116 and var_4:

# In[27]:


print('Length of unique var_116:',len(tr_te['var_116'].unique()))
print('Length of unique var_4  :',len(tr_te['var_4'].unique()))


# In[28]:


col = 'var_116'

statistic, pvalue = ks_2samp(tr.loc[tr['target']==0, col], tr.loc[tr['target']==1, col])
fig, ax = plt.subplots(1, 1, figsize=(10, 4))
sns.kdeplot(tr.loc[tr['target']==0, col], ax=ax, label='Target == 0')
sns.kdeplot(tr.loc[tr['target']==1, col], ax=ax, label='Target == 1')

ax.set_title('name: {}, statistics: {:.5f}, pvalue: {:5f}'.format(col, statistic, pvalue))
plt.show()


# In[29]:


col = 'var_4'

statistic, pvalue = ks_2samp(tr.loc[tr['target']==0, col], tr.loc[tr['target']==1, col])
fig, ax = plt.subplots(1, 1, figsize=(10, 4))
sns.kdeplot(tr.loc[tr['target']==0, col], ax=ax, label='Target == 0')
sns.kdeplot(tr.loc[tr['target']==1, col], ax=ax, label='Target == 1')

ax.set_title('name: {}, statistics: {:.5f}, pvalue: {:5f}'.format(col, statistic, pvalue))
plt.show()


# # Another oddity: var_16, var_41, and var_60:

# In[30]:


print('Length of unique var_16:',len(tr_te['var_16'].unique()))
print('Length of unique var_41:',len(tr_te['var_41'].unique()))
print('Length of unique var_60:',len(tr_te['var_60'].unique()))


# In[31]:


col = 'var_16'

statistic, pvalue = ks_2samp(tr.loc[tr['target']==0, col], tr.loc[tr['target']==1, col])
fig, ax = plt.subplots(1, 1, figsize=(10, 4))
sns.kdeplot(tr.loc[tr['target']==0, col], ax=ax, label='Target == 0')
sns.kdeplot(tr.loc[tr['target']==1, col], ax=ax, label='Target == 1')

ax.set_title('name: {}, statistics: {:.5f}, pvalue: {:5f}'.format(col, statistic, pvalue))
plt.show()


# In[32]:


col = 'var_41'

statistic, pvalue = ks_2samp(tr.loc[tr['target']==0, col], tr.loc[tr['target']==1, col])
fig, ax = plt.subplots(1, 1, figsize=(10, 4))
sns.kdeplot(tr.loc[tr['target']==0, col], ax=ax, label='Target == 0')
sns.kdeplot(tr.loc[tr['target']==1, col], ax=ax, label='Target == 1')

ax.set_title('name: {}, statistics: {:.5f}, pvalue: {:5f}'.format(col, statistic, pvalue))
plt.show()


# In[33]:


col = 'var_60'

statistic, pvalue = ks_2samp(tr.loc[tr['target']==0, col], tr.loc[tr['target']==1, col])
fig, ax = plt.subplots(1, 1, figsize=(10, 4))
sns.kdeplot(tr.loc[tr['target']==0, col], ax=ax, label='Target == 0')
sns.kdeplot(tr.loc[tr['target']==1, col], ax=ax, label='Target == 1')

ax.set_title('name: {}, statistics: {:.5f}, pvalue: {:5f}'.format(col, statistic, pvalue))
plt.show()


# # You can see these three variables all have this small little blue bump in the middle.

# In[ ]:




