#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from sklearn.datasets import load_breast_cancer


# In[ ]:


cancer=load_breast_cancer()


# In[ ]:


type(cancer)


# In[ ]:


cancer.keys()


# In[ ]:


#cancer.values()


# In[ ]:


print(cancer['DESCR'])


# In[ ]:


df=pd.DataFrame(cancer['data'],columns=cancer['feature_names'])


# In[ ]:


df.head(5)


# In[ ]:


cancer['target']


# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


scale=StandardScaler()


# In[ ]:


scale.fit(df)


# In[ ]:


for i in cancer['feature_names']:
    print(i);


# In[ ]:


np.std(df['worst area'])


# In[ ]:


df['worst area'].plot(kind='kde',figsize=(12,6))


# In[ ]:


scaled_data=scale.transform(df)


# In[ ]:


scaled_data


# In[ ]:


#PCA
from sklearn.decomposition import PCA


# In[ ]:


pca=PCA(n_components=2)


# In[ ]:


pca.fit(scaled_data)


# In[ ]:


x_pca=pca.transform(scaled_data)


# In[ ]:


scaled_data.shape


# In[ ]:


x_pca.shape
#dimension_reductionality from 30 to 2


# In[ ]:


plt.figure(figsize=(10,6))
plt.scatter(x_pca[:,0],x_pca[:,1],c=cancer['target'])
plt.xlabel('first principal component')
plt.ylabel('second principal component')
plt.legend()


# In[ ]:


print(pca.components_)


# In[ ]:


df_comp=pd.DataFrame(data=pca.components_ , columns=cancer['feature_names'])


# In[ ]:


df_comp


# In[ ]:


plt.tight_layout()
df_comp.plot(figsize=(12,6))


# In[ ]:


plt.figure(figsize=(12,7))
sns.heatmap(df_comp)


# **The End**
