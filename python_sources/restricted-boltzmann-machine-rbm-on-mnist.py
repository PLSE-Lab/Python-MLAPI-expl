#!/usr/bin/env python
# coding: utf-8

# This is just a basic demonstration of scikit's Restricted Boltzmann Machine (RBM) applied to MNIST, and then used to generate some new images with Gibbs sampling. It takes a while to run, but produces some nice results.

# In[94]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['image.cmap'] = 'gray'

import pandas as pd


# In[95]:


def gen_mnist_image(X):
    return np.rollaxis(np.rollaxis(X[0:200].reshape(20, -1, 28, 28), 0, 2), 1, 3).reshape(-1, 20 * 28)


# In[96]:


X_train = pd.read_csv('../input/train.csv').values[:,1:]
X_train = (X_train - np.min(X_train, 0)) / (np.max(X_train, 0) + 0.0001)  # 0-1 scaling


# In[97]:


plt.figure(figsize=(10,20))
plt.imshow(gen_mnist_image(X_train));


# In[98]:


from sklearn.neural_network import BernoulliRBM
rbm = BernoulliRBM(n_components=100, learning_rate=0.01, random_state=0, verbose=True)
rbm.fit(X_train)


# In[99]:


xx = X_train[:40].copy()
for ii in range(1000):
    for n in range(40):
        xx[n] = rbm.gibbs(xx[n])


# In[100]:


plt.figure(figsize=(10,20))
plt.imshow(gen_mnist_image(xx))


# In[101]:


xx = X_train[:40].copy()
for ii in range(10000):
    for n in range(40):
        xx[n] = rbm.gibbs(xx[n])


# In[102]:


plt.figure(figsize=(10,20))
plt.imshow(gen_mnist_image(xx))


# In[119]:


plt.figure(figsize=(20, 20))
for i, comp in enumerate(rbm.components_):
    plt.subplot(10, 10, i + 1)
    plt.imshow(comp.reshape((28, 28)), cmap=plt.cm.RdBu,
               interpolation='nearest', vmin=-2.5, vmax=2.5)
    plt.axis('off')
plt.suptitle('100 components extracted by RBM', fontsize=16);

