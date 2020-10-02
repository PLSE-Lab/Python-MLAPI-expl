#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.linalg import eigh as sc_eigh
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


# In[3]:


# loading dataset
train = pd.read_csv('../input/train.csv')
print(train.head())

#features
train_x = train.iloc[:,1:]
print('Training features:\n', train_x.head())

#label
train_y = train['label']
print('Labels:\n', train_y.head())


# In[4]:


# shape of our data:
print('Feature\'s shape:', train_x.shape)
print('Label\'s shape:', train_y.shape)


# In[5]:


# plotting some number images:
print('Some images are : ')
fig, axes = plt.subplots(nrows = 2, ncols = 2)

num_img = train_x.iloc[7].values.reshape(28, 28)
axes[0,0].imshow(num_img, interpolation = 'none')

num_img2 = train_x.iloc[43].values.reshape(28,28)
axes[0,1].imshow(num_img2, interpolation = 'none')

num_img3 = train_x.iloc[34].values.reshape(28,28)
axes[1,0].imshow(num_img3, interpolation = 'none')

num_img4 = train_x.iloc[718].values.reshape(28,28)
axes[1,1].imshow(num_img4, interpolation = 'none')

plt.show()


# In[6]:


# STEP 1: Data Standardization: mean = 0 , standard deviation = 1
sc = StandardScaler()
train_x_scaled = sc.fit_transform(train_x)
print('Shape of scaled data: ',train_x_scaled.shape)


# In[7]:


# STEP 2: Compute Covariance:
covar_train_x = np.cov(train_x_scaled.T) # S = X.T*X
print('Shape of our covariance matrix:' ,covar_train_x.shape) # square symmetric matrix


# In[8]:


# calculating eigen values and corresponding eigrn vectors using scipy eigh:
eig_val, eig_vec = sc_eigh(covar_train_x, eigvals = (782,783)) # Only getting top 2 eigen values and corresponding eigen vectors.
print('eigen values:\n',eig_val)
print('eigen vectors:\n',eig_vec.T)
print('eigen vectors shape:\n',eig_vec.T.shape)
print('train_x_scaled transposed shape:\n', train_x_scaled.T.shape)


# In[9]:


# Getting new coordinates:
new_values = np.matmul(eig_vec.T, train_x_scaled.T) # matrix multiplication
print('New values:\n', new_values)
print('New values shape:\n', new_values.shape)


# In[10]:


# Adding labels so we can perfrom visualization based on labels:
new_values = np.vstack((new_values, train_y)).T
print(new_values)


# In[11]:


# Converting to dataframe:
pca_df = pd.DataFrame(new_values, columns = ['1st principal component', '2nd principal component', 'label'])
print(pca_df.head())


# In[12]:


# visualization after dimensionality reduction (PCA):
sns.FacetGrid(pca_df, hue = 'label', height = 6).map(plt.scatter, '2nd principal component', '1st principal component').add_legend()
plt.show()


# In[13]:


#PCA USING SCIKIT LEARN


# In[14]:


pca = PCA()
pca.n_components = 2 # top 2
sci_pca = pca.fit_transform(train_x_scaled)
print('Shape after reducing dimensionality:', sci_pca.shape)
print('pca data:\n', sci_pca.T)


# In[15]:


sci_new_values = np.vstack((sci_pca.T, train_y)).T
print(sci_new_values)


# In[16]:


# creating the dataframe of our new coordinates:
sci_pca_df = pd.DataFrame(sci_new_values, columns = ['1st principal component', '2nd principal component', 'label'])
print(sci_pca_df.head()) # first 5 rows


# In[17]:


# Visualization in 2D:
sns.FacetGrid(sci_pca_df, hue = 'label', height = 6).map(plt.scatter, '1st principal component', '2nd principal component').add_legend()
plt.show()


# In[18]:


#VARIANCE EXPLANATION WITH NO. OF FEATURES


# In[19]:


pca.n_components = 784 # considering all the features
pca_dat = pca.fit_transform(train_x_scaled)

# pca.explained_variance_ -  The amount of variance explained by each of the selected components
var_explained = pca.explained_variance_ / np.sum(pca.explained_variance_) 
var_explained_cummulative = np.cumsum(var_explained)

plt.plot(var_explained_cummulative, linewidth = 3)
plt.grid()
plt.ylabel('Cummulative Variance')
plt.xlabel('number of components')
plt.show()


# In[ ]:




