#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # Load The Data 

# In[ ]:


data = pd.read_csv("../input/train.csv")

data.head(5)


# In[ ]:


data.shape


# ## Seperating labels from original data

# In[ ]:


label = data['label']
final_data= data.drop('label', axis=1)


# # Visualising data samples

# In[ ]:


fig = plt.figure(figsize=(20,15))

for i in range(1,10,1):
    
    plt.subplot(3,3,i)
    image = final_data.iloc[i].as_matrix().reshape(28,28)
    plt.imshow(image, cmap='gray')  
    plt.xlabel(label[i])
plt.show()


# # 2D visualisation of Data using PCA
# 
# *  First we will standardise the data.
# *  Then compute the Co-Variance Matrix of the data
# *  Then Compute Eigen values and Eigen Vectors
# *  projecting the original data sample on the plane formed by two principal eigen vectors by vector-vector multiplication.

# In[ ]:


# STEP-1
std_data = StandardScaler().fit_transform(final_data)
print(std_data.shape)


# In[ ]:


# STEP-2
CoVMat = np.matmul(std_data.T , std_data)
print(CoVMat.shape)


# In[ ]:


# STEP-3
from scipy.linalg import eigh

# the parameter 'eigvals' is defined (low value to heigh value) 
# eigh function will return the eigen values in asending order
# this code generates only the top 2 (782 and 783) eigenvalues.
values, vectors = eigh(CoVMat , eigvals=(782,783)) 

print("The shape of Eigen Vectors", vectors.shape)


# In[ ]:


#STEP-4
new_data = np.matmul(vectors.T , std_data.T)
print("The shape of new data", new_data.shape)


# ## Creating new dataframe from new coordinates 

# In[ ]:


stacking = np.vstack((new_data , label)).T
dataframe = pd.DataFrame(data=stacking , columns=("1st component", "2nd Component" , "Labels"))

dataframe.head(5)


# ## Plotting the data

# In[ ]:


import seaborn as sns
sns.FacetGrid(dataframe , hue='Labels', height=8).map(plt.scatter, '1st component' , '2nd Component').add_legend()
plt.show()


# # PCA using Scikit Learn

# In[ ]:


pca = PCA(n_components=2).fit_transform(std_data)
print("The shape of PCA reduced " ,pca.shape)


# In[ ]:


stacking = np.vstack((pca.T , label)).T
dataframe = pd.DataFrame(data=stacking , columns=("1st component", "2nd Component" , "Labels"))
dataframe.head(5)


# In[ ]:


sns.FacetGrid(dataframe , hue='Labels', height=8).map(plt.scatter, '1st component' , '2nd Component').add_legend()
plt.show()


# # PCA for dimensionality reduction (Not Visualisation)

# In[ ]:


pca = PCA(n_components=784).fit(std_data)
#print("The shape of PCA reduce", pca.shape)


# In[ ]:


pca_variance = pca.explained_variance_ratio_
cumsum_var = np.cumsum(pca_variance)

plt.figure(figsize=(8,6))
plt.clf()
plt.plot(cumsum_var, linewidth=2)
plt.xlabel('n_components')
plt.ylabel('Cumulative_explained_pca')
plt.grid()
plt.axis('tight')

