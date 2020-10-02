#!/usr/bin/env python
# coding: utf-8

# # Self Organizing Map 
# ## SOM is an unsupervised learning method. One of its uses is for clustering the data

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs


# # Creating data with 4 clusters

# In[ ]:



data, _ = make_blobs(n_samples=1000, centers=[(-10,10), (10,-10), (10,10), (-10,-10)], n_features=2,
                  random_state=0)
print('data.shape :', data.shape)
plt.plot(data[:,0], data[:,1], '.')


# ## **Let's define the initial weight vector for SOM**

# In[ ]:


W = np.array([[20,20,-20,-20], [20,-20,20,-20]], dtype = 'float')
# W.shape
W = W.T
W.shape
plt.scatter(W[:,0], W[:,1])


# # Applying Kohonen's weight update rule

# In[ ]:


lr = 0.001
for iter in range(10):
    for i in range(data.shape[0]):
        edist = np.linalg.norm(W - data[i,:], axis=1)
        ind = np.argmin(edist)
        W[ind,:] = (1 - lr)*W[ind,:] + lr*data[i,:]
    


# # Updated Weights

# In[ ]:


plt.scatter(W[:,0], W[:,1])


# In[ ]:


print("updated weights :\n", W)


# # SOM was able to identify the optimum locations for all the nodes/Ws 

# In[ ]:





# ## WIP, will update the notebook soon

# 

# In[ ]:





# In[ ]:




