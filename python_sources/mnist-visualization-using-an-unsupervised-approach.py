#!/usr/bin/env python
# coding: utf-8

# ### Importing necessary libraries

# In[ ]:


import pandas as pd

#Handle the numerical arrays
import numpy as np 

#Plotting the data
import seaborn as sns
import matplotlib.pyplot as plt

#Download MNIST dataset
from sklearn import datasets
from sklearn import manifold

get_ipython().run_line_magic('matplotlib', 'inline')


# ### Fetching the data

# In[ ]:


#Feth data from SKlearn
data = datasets.fetch_openml(
    'mnist_784',
    version=1,
    return_X_y=True
)
features, target = data 


# In[ ]:


features


# In[ ]:


# size of each image is 28X28
features.shape


# In[ ]:


target


# In[ ]:


target = target.astype(int)
target


# In[ ]:


target.shape


# In[ ]:


features[1,:].shape


# In[ ]:


# Selecting first row and rehape into 28X28
img = features[0,:].reshape(28,28)


# In[ ]:


plt.imshow(img, cmap = 'gray')


# In[ ]:


# tsne used for dimensionality reduction and used the technique of similarity score
tsne = manifold.TSNE(
    n_components=2,
    random_state=42
)


# In[ ]:


tsne_data = tsne.fit_transform(features[:2000,])


# In[ ]:


tsne_data.shape


# In[ ]:


df = pd.DataFrame(np.column_stack((tsne_data, target[:2000])), columns = ['X','Y','output'])


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


df.dtypes


# In[ ]:


grid = sns.FacetGrid(df, hue = 'output', size= 8)
grid.map(plt.scatter, 'X', 'Y').add_legend()


# In[ ]:




