#!/usr/bin/env python
# coding: utf-8

# IMPORT LIB

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


d0=pd.read_csv("/kaggle/input/digit-recognizer/train.csv")


# In[ ]:


d0.head()


# In[ ]:


l=d0['label']
l


# In[ ]:


d=d0.drop("label",axis=1)


# In[ ]:


d.head()


# THE DATA (X)

# In[ ]:


d.shape


# THE DEPENDENT VARIABLE (Y)

# In[ ]:


l.shape


# PRINTING AN IMAGE

# In[ ]:


plt.figure(figsize=(7,7))
idx=5
grid_data=d.iloc[idx].to_numpy().reshape(28,28)
plt.imshow(grid_data,interpolation="none",cmap="gray")
plt.show()
print (l[idx])


# In[ ]:


print (d.shape)
print(l.shape)


# STANDARDISING DATA

# In[ ]:


from sklearn.preprocessing import StandardScaler
standardized_data=StandardScaler().fit_transform(d)
print (standardized_data.shape)


# In[ ]:


standardized_data


# NEXT WE USE COVARIANCE MATRIX, EIGEN VALUES FOR PCA
# HERE WE ARE RUNNING PCA ALGORITHM STEP BY STEP AND NOT USING THE INBUILT

# In[ ]:


sample_data=standardized_data
covar_matrix=np.matmul(sample_data.T,sample_data)/len(l)
covar_matrix.shape


# In[ ]:


from scipy.linalg import eigh
values,vectors=eigh(covar_matrix,eigvals=(782,783))


# In[ ]:


vectors.shape


# In[ ]:


vectors=vectors.T


# In[ ]:


vectors.shape


# In[ ]:


new_coordinates=np.matmul(vectors,sample_data.T)
new_coordinates.shape


# ADDING LABEL TO THE NEW COORDINATES

# In[ ]:


new_coordinates = np.vstack((new_coordinates, l)).T
dataframe = pd.DataFrame(data=new_coordinates, columns=("1st_principal", "2nd_principal", "label"))
print(dataframe.head())


# VISUALISING USING SEABORN

# In[ ]:


import seaborn as sn
sn.FacetGrid(dataframe, hue="label", size=6).map(plt.scatter, '1st_principal', '2nd_principal').add_legend()


# DIRECT PCA USING SCIKIT

# In[ ]:


from sklearn import decomposition
pca=decomposition.PCA()
pca.n_components=2
pca_data=pca.fit_transform(sample_data)


# In[ ]:


pca_data.shape


# In[ ]:


pca_data=np.vstack((pca_data.T,l)).T


# In[ ]:


pca_df=pd.DataFrame(data=pca_data, columns=("1st_principal", "2nd_principal", "label"))


# In[ ]:


sn.FacetGrid(pca_df, hue="label", size=6).map(plt.scatter, '1st_principal', '2nd_principal').add_legend()


# PCA for Dimensionality Reduction

# IN THE GRAPH WE CAN SEE THAT USING AROUND 450 COMPONENTS i.e. 500 FEATURES, WE CAN EXPLAIN 99% VARIANCE
# SO WE CAN REDUCE DIMENSIONALITY FROM 784 TO 450

# In[ ]:


pca.n_components=784
pca_data=pca.fit_transform(sample_data)
percentage_var_explained=pca.explained_variance_/np.sum(pca.explained_variance_)
cum_var_explained=np.cumsum(percentage_var_explained)

plt.figure(1,figsize=(6,4))
plt.clf()
plt.plot(cum_var_explained,linewidth=2)
plt.axis('tight')
plt.grid()
plt.xlabel("n_components")
plt.ylabel("Cum_var_Exp")
plt.show()


# TSNE

# USING TSNE
# AS IT PRESERVES LOCAL STRUCTURE

# In[ ]:


from sklearn.manifold import TSNE
data=standardized_data
labels=l


# In[ ]:


model=TSNE(n_components=2,random_state=0)
tsne_data=model.fit_transform(data)


# In[ ]:


tsne_data=np.vstack((tsne_data.T,labels)).T
tsne_df=pd.DataFrame(data=tsne_data,columns=("D1","D2","label"))


# In[ ]:


sn.FacetGrid(tsne_df, hue="label", size=6).map(plt.scatter, 'D1', 'D2').add_legend()


# In[ ]:




