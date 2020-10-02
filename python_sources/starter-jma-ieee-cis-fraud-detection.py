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

test = pd.read_csv('../input/test_identity.csv')
train = pd.read_csv('../input/train_identity.csv')


# In[ ]:


train.head()


# In[ ]:


train.describe()


# In[ ]:


plt.matshow(train.corr())
plt.colorbar()
plt.show()


# In[ ]:


p = train.hist(figsize = (20,20))

