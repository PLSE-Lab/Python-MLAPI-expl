#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.linalg import eigh
import seaborn as sns
from sklearn import decomposition


# In[ ]:


df=pd.read_csv("/kaggle/input/digit-recognizer/train.csv")


# In[ ]:


df.head(4)


# In[ ]:


df.shape


# In[ ]:


labeldf=df['label']
labeldf


# In[ ]:


labeldf.shape


# In[ ]:


#Drop label column
dataset=df.drop("label",axis=1)
dataset


# In[ ]:


dataset.shape


# In[ ]:


#dataset.iloc[50].as_matrix()
#type(dataset)
#Sanity checks
griddata=dataset.iloc[3].values.reshape(28,28)
plt.imshow(griddata,interpolation="none",cmap="gray")


# In[ ]:


labeldf.iloc[50]


# In[ ]:


#standardizing the columns in the data
scaler = StandardScaler()
dataset_standardized=scaler.fit_transform(dataset)



# In[ ]:


#finding the transpose
dataset_standardized_t=dataset_standardized.transpose()
dataset_standardized_t.shape


# In[ ]:


#covariance
cov=np.matmul(dataset_standardized_t,dataset_standardized)
cov.shape


# In[ ]:


#Finding the eigen values and eigen vector
#top two variance features i.e 783,783 because eigh funtion returns eigh values in asc order
eigen_values, eigen_vector = eigh(cov,eigvals=(782,783))
eigen_vector.shape 


# In[ ]:


eigen_vector_transpose=eigen_vector.transpose()
eigen_vector_transpose.shape


# In[ ]:


newdata=np.matmul(eigen_vector_transpose,dataset_standardized_t)
print(eigen_vector_transpose.shape)
print(dataset_standardized_t.shape)
print(newdata.shape)



# In[ ]:


mnist_newdata=newdata.transpose()
mnist_newdata.shape


# In[ ]:


type(mnist_newdata)


# In[ ]:


dataFrame=pd.DataFrame(data=mnist_newdata,columns=("feature1","feature2"))
dataFrame.shape


# In[ ]:


dataFrame.insert(value=labeldf,column="label",loc=2)
dataFrame.head(5)


# In[ ]:


sns.FacetGrid(dataFrame,hue="label",size=7).map(plt.scatter,"feature1","feature2").add_legend()


# In[ ]:


#alternate using pca function in sklearn
pca=decomposition.PCA()
pca.n_components=784
data_pca=pca.fit_transform(dataset_standardized)
data_pca.shape


# In[ ]:


data_pca[1:4]


# In[ ]:


pca_dataframe=pd.DataFrame(data=data_pca,columns=("f1","f2","f3","f4"))


# In[ ]:


pca_dataframe.insert(value=labeldf,column="label",loc=0)


# In[ ]:


pca_dataframe.head(5)


# In[ ]:


eig_val,eig_vect = eigh(cov)
eig_val.shape
eigen_values[0]+eigen_values[1]/eig_val.sum()


# In[ ]:



var=pca.explained_variance_/np.sum(pca.explained_variance_)
varr=np.cumsum(var)
varr


# In[ ]:


plt.plot(varr,linewidth=2)
plt.xlabel('n_components')
plt.ylabel('cum var')


# In[ ]:




