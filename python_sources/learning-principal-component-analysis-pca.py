#!/usr/bin/env python
# coding: utf-8

# ## IMPORTING REQUIRED LIBRARIES

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.linalg import eigh as sc_eigh
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


# ## IMPORTING DATASET

# In[ ]:


# loading dataset
train = pd.read_csv('../input/train.csv')
print(train.head())

#features
train_x = train.iloc[:,1:]
print('Training features:\n', train_x.head())

#label
train_y = train['label']
print('Labels:\n', train_y.head())


# In[ ]:


# shape of our data:
print('Feature\'s shape:', train_x.shape)
print('Label\'s shape:', train_y.shape)


# In[ ]:


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


# ## APPLYING PCA STEP BY STEP FOR VISUALIZATION

# In[ ]:


# STEP 1: Data Standardization: mean = 0 , standard deviation = 1
sc = StandardScaler()
train_x_scaled = sc.fit_transform(train_x)
print('Shape of scaled data: ',train_x_scaled.shape)


# In[ ]:


# STEP 2: Compute Covariance:
covar_train_x = np.cov(train_x_scaled.T) # S = X.T*X
print('Shape of our covariance matrix:' ,covar_train_x.shape) # square symmetric matrix


# In[ ]:


# calculating eigen values and corresponding eigrn vectors using scipy eigh:
eig_val, eig_vec = sc_eigh(covar_train_x, eigvals = (782,783)) # Only getting top 2 eigen values and corresponding eigen vectors.
print('eigen values:\n',eig_val)
print('eigen vectors:\n',eig_vec.T)
print('eigen vectors shape:\n',eig_vec.T.shape)
print('train_x_scaled transposed shape:\n', train_x_scaled.T.shape)


# In[ ]:


# Getting new coordinates:
new_values = np.matmul(eig_vec.T, train_x_scaled.T) # matrix multiplication
print('New values:\n', new_values)
print('New values shape:\n', new_values.shape)


# In[ ]:


# Adding labels so we can perfrom visualization based on labels:
new_values = np.vstack((new_values, train_y)).T
print(new_values)


# In[ ]:


# Converting to dataframe:
pca_df = pd.DataFrame(new_values, columns = ['1st principal component', '2nd principal component', 'label'])
print(pca_df.head())


# In[ ]:


# visualization after dimensionality reduction (PCA):
sns.FacetGrid(pca_df, hue = 'label', height = 6).map(plt.scatter, '2nd principal component', '1st principal component').add_legend()
plt.show()


# ## PCA using scikit learn

# In[ ]:


pca = PCA()
pca.n_components = 2 # top 2
sci_pca = pca.fit_transform(train_x_scaled)
print('Shape after reducing dimensionality:', sci_pca.shape)
print('pca data:\n', sci_pca.T)


# In[ ]:


sci_new_values = np.vstack((sci_pca.T, train_y)).T
print(sci_new_values)


# In[ ]:


# creating the dataframe of our new coordinates:
sci_pca_df = pd.DataFrame(sci_new_values, columns = ['1st principal component', '2nd principal component', 'label'])
print(sci_pca_df.head()) # first 5 rows


# In[ ]:


# Visualization in 2D:
sns.FacetGrid(sci_pca_df, hue = 'label', height = 6).map(plt.scatter, '1st principal component', '2nd principal component').add_legend()
plt.show()


# ## VARIANCE EXPLAINATION WITH NO. OF FEATURES

# In[ ]:


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


# * * In above plot, we can see that more than 90% variance is explained by 350 components which is less than half of the total number of components.

# ## REFERENCES:
# 
# 1. Applied AI Course : [www.appliedaicourse.com]
# 2. Dataset : [https://www.kaggle.com/c/digit-recognizer]
# 
