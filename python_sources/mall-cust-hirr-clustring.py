#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


data=pd.read_csv("../input/mall-customer-dataset/Mall_Customers.csv")
data.head()


# In[ ]:


data.shape


# In[ ]:


data.info()


# In[ ]:


data.describe()


# In[ ]:


data.isna().sum()


# In[ ]:


import seaborn as sb


# In[ ]:


sb.countplot(x="Genre",data=data)


# In[ ]:


data["Annual Income (k$)"].plot(kind="hist")


# In[ ]:


data["Spending Score (1-100)"].plot(kind="hist")


# In[ ]:


X=data.iloc[:,[3,4]].values


# In[ ]:


y=data.iloc[:,[3]].values


# In[ ]:


from sklearn.preprocessing import StandardScaler
scale=StandardScaler()


# In[ ]:


X=scale.fit_transform(X)
y=scale.fit_transform(y)


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


from scipy.cluster.hierarchy import dendrogram , linkage
linked=linkage(X,"ward")
plt.figure(figsize=(10,7))
plt.xlabel("Customers")
dendrogram(linked,
           orientation='top',
            distance_sort='descending',
            show_leaf_counts=True)
plt.show()


# In[ ]:




