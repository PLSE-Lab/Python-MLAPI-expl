#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.linalg import eigh
import random
from sklearn.manifold import TSNE


# ## 1. Loading Data 

# In[ ]:


df = pd.read_csv('../input/train.csv')
df.info()


# In[ ]:


print(df.head())
df = df[:1000]


# We have 28x28 px images flattened into 784 px per datapoint. hence for each datapoint we have 784 columns. 
# label field is the class attribute for which numeric digit this image belongs to. 

# In[ ]:


"Seperating labels and feature vectors"
label = df.label
data = df.drop('label', axis = 1)


# ## 2. Visualizing Data 
# Making this function to show random datapoint and it corresponding value. 

# In[ ]:


def print_index(id):
    idx = id
    grid_data = data.iloc[idx].values.reshape(28,28)
    plt.imshow(grid_data, interpolation = None, cmap = 'gray')
    plt.title(label[idx])


# In[ ]:


print_index(random.randint(0, 1000)) 


# ## 3. Column Standardization : 
# Data preprocessing such that the mean of each feature vector lies at the origin and standard deviation of each feature vector is 1. 

# In[ ]:


standardized_data = StandardScaler().fit_transform(data)
print(standardized_data.shape)


# ## 4. PCA Mathematics : 
# 1. Finding Co variance matrix for the fiven feature matrix as X<sup>T</sup>X.
# 2. Since we are going from 784 dimensions to 2 dimensions, we only need highest 2 eigen values and vectors(V). 
# 3. Data matrix in 2 dimensions will be equal to X<sup>'</sup><sub>i</sub> = X<sup>T</sup><sub>i</sub>V.

# In[ ]:


cov_matrix = np.dot(standardized_data.T, standardized_data)
values, vectors = eigh(cov_matrix, eigvals = (782, 783))
new_data = np.dot(vectors.T, standardized_data.T)
print(new_data.shape)
xdash = np.vstack((new_data, label)).T
print(xdash.shape)


# ## 4. Visualizing in 2D space. 

# In[ ]:


df = pd.DataFrame(data = xdash, columns = ('1st Principal', '2nd Principal', 'Labels'))
sns.FacetGrid(df, hue = 'Labels', height = 7).map(plt.scatter, '1st Principal', '2nd Principal').add_legend()
plt.title('PCA on MNIST')
plt.show()


# ## 6. t-SNE :
# t-SNE, stands for t distribution Stochastic Neighborhood Embedding, is a state of the art technique used tor dimensionality reduction and visualizing higher dimension data easily. 

# In[ ]:


model = TSNE(n_components = 2, random_state = 0)
tsne_model = model.fit_transform(standardized_data)


# In[ ]:


tsne_data = np.vstack((tsne_model.T, label)).T
# tsne_data.shape
# labels.shape
tse_df = pd.DataFrame(data= tsne_data, columns=('Dim_1', 'Dim_2', 'Labels'))


# In[ ]:


sns.FacetGrid(tse_df, hue = 'Labels', height = 6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
plt.show()


# It is visible that t-SNE performs much better dimensionality reduction as the clusters formed in PCA are very sparse but in t-SNE they are much closer.

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




