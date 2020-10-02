#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
import os 
import pandas as pd


# In[ ]:


test = pd.read_csv('../input/test.csv')
train = pd.read_csv('../input/train.csv')


# In[ ]:


test.head()


# In[ ]:


test.describe()


# In[ ]:


plt.matshow(test.corr())
plt.colorbar()
plt.show()


# In[ ]:


train.head()


# In[ ]:


train.describe()


# In[ ]:


plt.matshow(train.corr())
plt.colorbar()
plt.show()


# In[ ]:


sns.lineplot(x='atom_index_0', y='atom_index_1', data=train)


# In[ ]:


sns.lineplot(x='atom_index_1', y='scalar_coupling_constant', data=train)


# In[ ]:


p = train.hist(figsize = (20,20))


# In[ ]:


plt.figure()
sns.distplot(train['scalar_coupling_constant'])
plt.show()
plt.close()

