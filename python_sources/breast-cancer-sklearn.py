#!/usr/bin/env python
# coding: utf-8

# https://github.com/satyamuralidhar/breast_cancer  

# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


import seaborn as sns


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from sklearn.datasets import load_breast_cancer


# In[ ]:


cancer = load_breast_cancer()


# In[ ]:


cancer.keys()


# In[ ]:


print(cancer['feature_names'])


# In[ ]:


print(cancer['DESCR'])


# In[ ]:


df = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])


# In[ ]:


df.head()


# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


scaler = StandardScaler()


# In[ ]:


scaler_data = scaler.fit_transform(df)


# In[ ]:


scaler_data


# In[ ]:


from sklearn.decomposition import PCA


# In[ ]:


pca = PCA(n_components=2)


# In[ ]:


pca


# In[ ]:


x_pca = pca.fit_transform(scaler_data)


# In[ ]:


x_pca


# In[ ]:


scaler_data.shape


# In[ ]:


x_pca.shape


# In[ ]:


plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1],c=cancer['target'],cmap='plasma')
plt.xlabel('first principle component')
plt.ylabel('second principle component')
plt.show()


# In[ ]:




