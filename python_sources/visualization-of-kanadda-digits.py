#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("/kaggle/input/Kannada-MNIST/train.csv")
df


# In[ ]:


# Save the label column in variable l
l = df["label"]
l


# In[ ]:


d = df.drop("label",axis=1)
d


# In[ ]:


print(d.shape)
print(l.shape)


# In[ ]:


# display or plot a number 
plt.figure(figsize=(10,10))
idx=3502

grid_data = d.iloc[idx].as_matrix().reshape(28,28)
plt.imshow(grid_data,interpolation=None,cmap="gray")

plt.show()

print(l[idx])


# In[ ]:


d = df.drop("label",axis=1)
d


# In[ ]:


l = df["label"]
l


# In[ ]:


# As first our data should be standardized 
# 

from sklearn.preprocessing import StandardScaler
standardized_data = StandardScaler().fit_transform(d)
standardized_data


# In[ ]:


# Building a covariance matrix
sample_data = standardized_data
import numpy as np
covar_matrix = np.matmul(sample_data.T,sample_data)
covar_matrix
print("Covariance matrix shape: ",covar_matrix.shape)


# In[ ]:


from scipy.linalg import eigh
values,vectors = eigh(covar_matrix,eigvals=(782,783))
# print(values)
# print(vectors)
print(vectors.shape)
print(values.shape)


# In[ ]:


# Storing transpose of vectors variable 

vectors = vectors.T
#The shape of Transposed vector is 
print("New shape is : ",vectors.shape)


# In[ ]:


# sample_data.shape

# creating new coordinates
new_cordinates = np.matmul(vectors,sample_data.T)
new_cordinates


# In[ ]:


new_cordinates.shape


# In[ ]:


import pandas as pd
new_cordinates = np.vstack((new_cordinates,l)).T
new_cordinates.shape


# In[ ]:


# Creating a Dataframe
dataframe = pd.DataFrame(new_cordinates,columns=("ist principal","2nd principal","labels"))
dataframe


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
# Visualize the data using seaborn
sns.FacetGrid(dataframe,hue="labels",size=10).map(plt.scatter,"ist principal","2nd principal").add_legend()


# 

# In[ ]:


# ## Steps to be carried out are:
# Step 1: split dataset into two parts one consisting of label column and second consisting of rest of the data
# Step 2: As we know for pca first our data should be standardized therefore we will convert our data to column standardized
# Step 3: After that we create a covariance matrix of our standardized_data i.e standardized_data^T*standardized_data
# Step 4: Then we find eigen values and eigen vectors (As our covariance matrix is 784*784 dimensions therefore in total we will get 784 eigen values and 784 eigen vectors).The scipy eigh funcion provides this very easily and gives eigen values and vectors in ascending order as we are only interested in last two(highest) eigen values and eigen vectors therefore we pass eigvals as 782,783
# Step 5: Taking traspose of this eigen vector so that dimention becomes (2,784)
# Step 6: taking eigen_vector.standardized_data^T
# step 7: creating a vstack with eigen_vector.standardized_data^T and labels
# step 8: creating a dataframe of vstack with corresponding column names and visualizng data using seaborn
# Step 9: checking the graph obtained


# In[ ]:


## To directly do PCA by
d = df.drop("label",axis=1)
d


# In[ ]:


l = df["label"]
l


# In[ ]:


## Standardizing the data
from sklearn.preprocessing import StandardScaler
standardized_data = StandardScaler().fit_transform(d)
standardized_data.shape


# In[ ]:


from sklearn.decomposition import PCA
pca = PCA()
pca.n_components = 2
pca_fitdata = pca.fit_transform(standardized_data)
pca_fitdata


# In[ ]:


pca_fitdata.shape


# In[ ]:


l.shape


# In[ ]:


# data = np.vstack((pca_fitdata,l))
# data


# In[ ]:


a = pd.DataFrame(pca_fitdata,columns=("1st principal","2nd principal"))
a


# In[ ]:


b = l.to_frame(name="label")
b


# In[ ]:


df1 = pd.concat([a,b],axis=1)
df1


# In[ ]:


# Visualizing the dataframe made using seaborn 
sns.FacetGrid(df1,hue="label",size=10).map(plt.scatter,"1st principal","2nd principal").add_legend();


# In[ ]:


# Standardized 
from sklearn.preprocessing import StandardScaler
standardized_data = StandardScaler().fit_transform(d)
sample_data = standardized_data
# PCA for dimentionality reduction
from sklearn.decomposition import PCA
pca = PCA()
pca.n_component = 784
pca_data = pca.fit_transform(sample_data)

percentage_var_explained = pca.explained_variance_/np.sum(pca.explained_variance_)

cumsum_percentage_var = np.cumsum(percentage_var_explained)

# Plotting 

plt.figure(1,figsize=(10,10))
plt.clf()
plt.plot(cumsum_percentage_var,linewidth=2)
plt.axis("tight")
plt.grid()
plt.xlabel("n_component")
plt.ylabel("cumsum")
plt.show()


# In[ ]:


# print(pca.explained_variance_)
# print("???????????????????????????????????????????????")
# print(pca.explained_variance_ratio_)
# print("/////////////////////////////////////////////////////////////")
# print(pca.explained_variance_ratio_.cumsum())


# # t-SNE(t-disb Stochastic Neighborhood Embedding):-  

# In[ ]:


df


# In[ ]:


data = df.drop("label",axis=1)
data


# In[ ]:


label = df["label"]
label


# In[ ]:


### As in case of PCA can only be applied to standardized data similar to that t-SNE 
from sklearn.preprocessing import StandardScaler
standardized_data = StandardScaler().fit_transform(data)
standardized_data


# In[ ]:


get_ipython().system('pip install sklearn.manifold')


# In[ ]:


from sklearn.manifold import TSNE

model = TSNE(random_state=0)
model.n_component=2
### Configuring the parameters
# the number of components = 2
# the default perplexity= 30
# default learning rate = 200
# default maximum number of iterations for optimization = 1000
tsne_data = model.fit_transform(standardized_data)


# In[ ]:


tsne_data


# In[ ]:


### Converting label to dataframe
a = pd.DataFrame(tsne_data,columns=("1st dim","2nd dim"))
a


# In[ ]:


b = label.to_frame(name="label")
b


# In[ ]:


df1 = pd.concat([a,b],axis=1)
df1


# In[ ]:


# Visualizing the dataframe made using seaborn 
sns.FacetGrid(df1,hue="label",size=10).map(plt.scatter,"1st dim","2nd dim").add_legend();


# In[ ]:


### Tweaking the parameter of t-SNE
from sklearn.manifold import TSNE

model = TSNE(random_state=0,perplexity=50,n_iter=5000)
model.n_component=2 #n_component basically means no of column we want in our t-SNE data to have
### Configuring the parameters
# the number of components = 2
# the default perplexity= 30
# default learning rate = 200
# default maximum number of iterations for optimization = 1000
tsne_data = model.fit_transform(standardized_data)


# In[ ]:


tsne_data


# In[ ]:


### Converting label to dataframe
a = pd.DataFrame(tsne_data,columns=("1st dim","2nd dim"))
a


# In[ ]:


b = label.to_frame(name="label")
b


# In[ ]:


df1 = pd.concat([a,b],axis=1)
df1


# In[ ]:


# Visualizing the dataframe made using seaborn 
sns.FacetGrid(df1,hue="label",size=10).map(plt.scatter,"1st dim","2nd dim").add_legend();

