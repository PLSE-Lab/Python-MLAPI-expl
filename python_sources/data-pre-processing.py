#!/usr/bin/env python
# coding: utf-8

# In[151]:


import pandas as pd
import numpy as np


# In[152]:


data = pd.read_csv("../input/nflplaybyplay2009to2016/NFL Play by Play 2009-2017 (v4).csv")

data.head(10)


# In[153]:


data.describe()


# # Handling Missing Values

# In[154]:


data.isnull().sum()


# In[155]:


total_cells = np.product(data.shape)
total_missing = data.isnull().sum().sum()

(total_missing/total_cells)*100


# In[156]:


data = data.fillna(method='bfill').fillna(0,axis=0)


# In[157]:


data.isnull().sum()


# **Using Imputer**

# In[158]:


data = pd.read_csv("../input/nflplaybyplay2009to2016/NFL Play by Play 2009-2017 (v4).csv")

data.head(5)


# In[159]:


data['airEPA'].isnull().sum()
data['EPA'].isnull().sum()


# In[160]:


from sklearn.impute import SimpleImputer

imputer = SimpleImputer()
df = imputer.fit_transform(data.loc[:,['airEPA','EPA']])


# In[161]:


data['airEPA'] = df[:,0]
data['EPA'] = df[:,1]


# In[162]:


data.head(3)


# In[163]:


data['EPA'].isnull().sum()
data['airEPA'].isnull().sum()


# In[164]:


import seaborn as sns
import matplotlib.pyplot as plt
data = np.random.exponential(size = 1000)

fig,ax = plt.subplots()
ax=sns.distplot(data)
plt.show()


# # Scaling - Convert in 0-1 range

# In[165]:


from sklearn.preprocessing import minmax_scale

scaled_data = minmax_scale(data)

scaled_data

fig,ax = plt.subplots()
ax = sns.distplot(scaled_data)
plt.show()


# In[166]:


print(data[0:10])
print(scaled_data[0:10])


# # Normalization - Convert in Bell Curve (Gaussian Distribution)

# In[175]:


from sklearn.preprocessing import normalize
from scipy import stats






data = np.random.exponential(size = 1000)



normalize_data = stats.boxcox(data)

normalize_data = normalize_data[0]



fig,ax = plt.subplots()
ax = sns.distplot(normalize_data)
plt.show()

type(normalize_data)




