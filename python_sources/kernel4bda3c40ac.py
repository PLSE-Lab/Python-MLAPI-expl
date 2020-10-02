#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
from scipy.stats import norm #Analysis 
import pandas as pd 
import os
from random import sample 
from scipy import stats #Analysis
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import matplotlib

print(os.listdir("../input"))


# In[2]:


train_csv = pd.read_csv('../input/train.csv')
test_csv = pd.read_csv('../input/test.csv')
print (train_csv.shape)
print (train_csv.shape)


# In[3]:


train_csv.head()


# In[4]:


test_csv.head()


# In[ ]:





# In[6]:


print("skewness: %f" % train_csv['SalePrice'].skew())
print("kurtosis: %f" % train_csv['SalePrice'].kurt())


# In[7]:


fig = plt.figure(figsize = (15,10))

fig.add_subplot(1,2,1)
res = stats.probplot(train_csv['SalePrice'], plot=plt)

fig.add_subplot(1,2,2)
res = stats.probplot(np.log1p(train_csv['SalePrice']), plot=plt)

