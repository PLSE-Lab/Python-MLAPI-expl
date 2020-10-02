#!/usr/bin/env python
# coding: utf-8

# In[1]:


from numpy import *
from pandas import *
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

data = read_csv("../input/iris.data.csv", header = None)

data[4] = data[4].astype("category")
data[4] = data[4].cat.codes

Y = array(data.pop(4))
X = array(data)


# In[2]:


from sklearn.mixture import GaussianMixture as mix
model = mix(n_components = len(np.unique(Y)))


# In[3]:


model.fit(X).predict(X)


# In[10]:


from sklearn import datasets as data
mnist = data.load_digits()

plt.gray()
plt.matshow(mnist.images[100])
plt.show()

Y = mnist.target
X = mnist.images

X = X.reshape(len(X),-1)


# In[160]:


from sklearn.mixture import GaussianMixture as mix
model = mix(n_components = 10, init_params='kmeans',
           n_init = 5, max_iter = 5000, covariance_type = 'diag')
model.fit(X)

preds = model.predict(X)
labels = {}
seen = []

for dist in range(10):
    part = Y[where(preds==dist)]
    print(part)


# In[149]:





# In[133]:





# In[130]:


most


# In[67]:


preds = np.array([labels[x] for x in preds])


# In[70]:


sum(preds==Y)/len(Y)

