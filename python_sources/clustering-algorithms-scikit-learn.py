#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv("../input/Wholesale customers data.csv")
df.head()


# In[ ]:


plt.scatter(x = df['Milk'], y = df['Grocery'])


# In[ ]:


df.describe()


# In[ ]:


temp = df[['Milk', 'Grocery']]
temp = temp.as_matrix().astype('float32', copy = False)
std_scaler = StandardScaler().fit(temp)
temp = std_scaler.transform(temp)
dbscan = DBSCAN(eps = 0.5, min_samples = 15).fit(temp)
core_samples = np.zeros_like(dbscan.labels_, dtype = bool)
core_samples[dbscan.core_sample_indices_] = True


# In[ ]:


x = pd.DataFrame(core_samples, columns=['cluster']) 
ind = x.index[x['cluster'] == True].tolist()
p1 = df.iloc[ind, :]
p2 = pd.concat([df,p1]).drop_duplicates(keep=False)
plt.scatter(x = p1['Milk'], y = p1['Grocery'])
plt.scatter(x = p2['Milk'], y = p2['Grocery'])

