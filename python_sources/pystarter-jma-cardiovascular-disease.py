#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math as mt
import scipy

from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")


# In[ ]:


url = '../input/cardiovascular-disease/cardiovascular.txt'
data = pd.read_csv(url,sep=';',decimal=',')

# let's separate index from other columns
data.index = data.iloc[:,0]
df = data.iloc[:,1:]

df = df.drop(['chd','famhist'],axis=1)


# In[ ]:


data.head()


# In[ ]:


data.shape


# In[ ]:


df.head()


# In[ ]:


df.dtypes


# In[ ]:


df = df.astype('float')


# In[ ]:


df.dtypes


# In[ ]:


df.describe()


# In[ ]:


sns.pairplot(df)


# In[ ]:


ax = sns.scatterplot(x="sbp", y="obesity",  data=df)

