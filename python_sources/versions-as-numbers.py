#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# I am thinking of using versions as numbers - Ias such I looked at the two version columns that Olivier found in his adversarial approach

# In[ ]:


train = pd.read_csv('../input/train.csv',usecols=['MachineIdentifier','EngineVersion','AvSigVersion','HasDetections'])


# In[ ]:


train.head()


# In[ ]:


def MungeVersionInfo(data, c):
    data[c] = data[c].str.replace(r'[^\.|0-9]','')#There is a weird value in AvSigVersion
    mysplit = data[c].str.split('.',expand=True).astype(int)
    mymax = np.ceil(np.log10(np.maximum(10,mysplit.max())))
    mysum = mymax.sum()
    pseudonumber = np.zeros(data.shape[0])
    placeholder = 0
    for i in range(mysplit.shape[1]):
        myfactor = (10**(mysum - mymax[i]-placeholder)).astype(int)
        pseudonumber += myfactor*mysplit[mysplit.columns[i]].values
        placeholder += mymax[i]
    return pseudonumber.astype(int)


# In[ ]:


for c in ['EngineVersion','AvSigVersion']:
    train[c] = MungeVersionInfo(train, c)


# In[ ]:


train.head()


# In[ ]:


a = train.groupby('EngineVersion')['HasDetections'].mean().reset_index(drop=False)
plt.figure(figsize=(15,15))
plt.plot(a.EngineVersion,a.HasDetections)


# In[ ]:


a = train.groupby('AvSigVersion')['HasDetections'].mean().reset_index(drop=False)
plt.figure(figsize=(15,15))
plt.plot(a[(a.AvSigVersion>0)&(a.AvSigVersion<1.05e9)].AvSigVersion,a[(a.AvSigVersion>0)&(a.AvSigVersion<1.05e9)].HasDetections)


# In[ ]:


print(train['EngineVersion'].min())
print(train['EngineVersion'].mean())
print(train['EngineVersion'].max())


# In[ ]:


print(train['AvSigVersion'].min())
print(train['AvSigVersion'].mean())
print(train['AvSigVersion'].max())


# In[ ]:


del train
gc.collect()


# In[ ]:


test = pd.read_csv('../input/test.csv',usecols=['MachineIdentifier','EngineVersion','AvSigVersion'])


# In[ ]:


for c in ['EngineVersion','AvSigVersion']:
    test[c] = MungeVersionInfo(test, c)


# In[ ]:


print(test['EngineVersion'].min())
print(test['EngineVersion'].mean())
print(test['EngineVersion'].max())


# In[ ]:


print(test['AvSigVersion'].min())
print(test['AvSigVersion'].mean())
print(test['AvSigVersion'].max())


# In[ ]:




