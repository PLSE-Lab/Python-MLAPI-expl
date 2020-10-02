#!/usr/bin/env python
# coding: utf-8

# # Principal Component Analysis (PCA) is Easy

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


# # Don't know what is PCA? Check out Siraj's video on PCA [here](https://youtu.be/jPmV3j1dAv4)

# ## Loading libraries and input data

# In[2]:


import seaborn as sns
import matplotlib.pyplot as plt
# reading dataset
df=pd.read_csv('../input/Admission_Predict_Ver1.1.csv')


# **Checking the data:**

# In[3]:


display(df.head())
df.info()
display(df.describe())


# ## No Missing data, thats great!!

# In[4]:


#dividing into features and labels
features=df.iloc[:,1:-1]
labels=df.iloc[:,-1]
display(features[:5])
display(labels[:5])


# In[5]:


new_labels=pd.cut(np.array(labels),3, labels=["bad", "medium", "good"])
print(new_labels.shape)
new_labels[:5]


# ## Standardization/Normalization of features

# In[6]:


from sklearn.preprocessing import StandardScaler


# In[7]:


# Normalizing the data
standardized_data=StandardScaler().fit_transform(features)
standardized_data[:5]


# # PCA using only Numpy

# In[10]:


print('NumPy covariance matrix: \n%s' %np.cov(standardized_data.T))
cov_mat = np.cov(standardized_data.T)


# In[11]:


#Calculating the eigen values and vectors
eig_vals, eig_vecs = np.linalg.eig(cov_mat)

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)


# In[12]:


# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])


# In[14]:


matrix_w = np.hstack((eig_pairs[0][1].reshape(7,1), 
                      eig_pairs[1][1].reshape(7,1)))

print('Matrix W:\n', matrix_w)


# In[16]:


Y = standardized_data.dot(matrix_w)


# In[19]:


Y.shape


# In[23]:


pca_data=np.vstack((Y.T,new_labels)).T
pca_df=pd.DataFrame(data=pca_data,columns=("1st Component","2nd Component","Chances of getting in?"))
pca_df.head()


# In[24]:


fig, ax = plt.subplots()
# the size of A4 paper
fig.set_size_inches(12, 8)
sns.scatterplot(x="1st Component", y="2nd Component", hue="Chances of getting in?", data=pca_df);


# ## Using the sklearn to perform PCA. sklearn is amazing, it has it all!

# In[25]:


from sklearn import decomposition
pca=decomposition.PCA()


# In[26]:


pca.n_components=2
pca_data=pca.fit_transform(standardized_data)
print("The reduced shape is", pca_data.shape)


# In[27]:


pca_data[:5]


# In[28]:


pca_data=np.vstack((pca_data.T,new_labels)).T
pca_df=pd.DataFrame(data=pca_data,columns=("1st Component","2nd Component","Chances of getting in?"))
pca_df.head()


# In[29]:


fig, ax = plt.subplots()
# the size of A4 paper
fig.set_size_inches(12, 8)
sns.scatterplot(x="1st Component", y="2nd Component", hue="Chances of getting in?", data=pca_df);


# # You can see the grouping already! Yayy!

# In[ ]:




