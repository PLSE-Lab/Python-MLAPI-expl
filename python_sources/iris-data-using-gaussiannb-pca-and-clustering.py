#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


iris  = sns.load_dataset("iris")
iris.head()


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

sns.pairplot(iris,hue = 'species',height = 2)


# In[ ]:


X = iris.drop('species',axis = 1)
y = iris['species']


# In[ ]:


X.shape


# In[ ]:


y.shape


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 1)


# # GaussianNB

# In[ ]:


from sklearn.naive_bayes import GaussianNB


# In[ ]:


model  = GaussianNB()
model.fit(X_train,y_train)


# In[ ]:


y_model = model.predict(X_test)


# In[ ]:


model.score(X_test,y_test)


# In[ ]:


#alternate scoring
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_model)


# # Dimensionality reduction

# In[ ]:


from sklearn.decomposition import PCA
model =  PCA(n_components = 2)
model.fit(X)
X_2d =model.transform(X)


# In[ ]:


iris['PCA1'] =X_2d[:,0]  #take all the data of the first dimension in the two-dimensional array
iris['PCA2'] =X_2d[:,1] #take all the data of the second dimension in the two-dimensional arra


# In[ ]:


sns.lmplot("PCA1","PCA2",hue ='species',data=iris,fit_reg = False)


# # Clustering

# In[ ]:


#using Gaussian mixture model. It models the data as a collection of Gaussian blobs

from sklearn.mixture import GaussianMixture
model = GaussianMixture(n_components = 3)
model.fit(X)
y_gmm = model.predict(X)


# In[ ]:


iris['cluster']=y_gmm
sns.lmplot("PCA1","PCA2",data=iris,hue = 'species',col ='cluster',fit_reg = False)


# In[ ]:




