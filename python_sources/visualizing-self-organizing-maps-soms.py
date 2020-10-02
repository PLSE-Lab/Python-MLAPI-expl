#!/usr/bin/env python
# coding: utf-8

# **Introduction to cluster visualization using SOM**
# 
# This notebook will describe various cluster visualization techniques for SOMs. Then implement couple of methods to understand the concept better.
# 
# Following 3 types of visualization are more common in the literature.    
# 
# 1. U-matrix
# 2. Vector fields
# 3. Component planes

# **Existing implementations of SOMs as python libraries**
# 
# You can use following libraries to train and visualize self oranizing maps which is used for cluster analysis jobs of low/high dimensional data.
# It is not that hard to write the SOM algorithm on yourown.
# 
# **1. somclu**
# 
# you can use the implementation of this at following links
# 
# > Documentation :- https://peterwittek.github.io/somoclu/   
# > Github        :- https://somoclu.readthedocs.io/en/stable/    
# > Example       :- https://github.com/abhinavralhan/kohonen-maps/blob/master/somoclu-iris.ipynb
# 
# **2. minisom**
# 
# **3. GSom**
# 
# **4. SimpSPM**
# 
# > GitHub - https://github.com/fcomitani/SimpSOM    
# > Sample - https://www.kaggle.com/asparago/unsupervised-learning-with-som

# **Loading the dataset - Iris**

# In[ ]:


import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patches as patches
get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

Iris = pd.read_csv("../input/iris/Iris.csv")
Iris = Iris.drop(Iris.columns[-1],axis=1)
train = StandardScaler().fit_transform(Iris.values)
pd.DataFrame(train).head()


# **Use SimpSOM to visualize U-matrix**
# 
# Following documents and codes are used to demostrate this.    
# 
# > GitHub - https://github.com/fcomitani/SimpSOM    
# > Example - https://www.kaggle.com/asparago/unsupervised-learning-with-som

# In[ ]:


pip install SimpSOM


# In[ ]:


#Import the library
import SimpSOM as sps

#Build a network 20x20 with a weights format taken from the raw_data and activate Periodic Boundary Conditions. 
net = sps.somNet(20, 20, train, PBC=True)

#Train the network for 10000 epochs and with initial learning rate of 0.01. 
net.train(0.01, 10000)

#Save the weights to file
net.save('filename_weights')

#Print a map of the network nodes and colour them according to the first feature (column number 0) of the dataset
#and then according to the distance between each node and its neighbours.
net.nodes_graph(colnum=0)
net.diff_graph()


# **Implement an own SOM and visualize**
# 
# Following implementation from github is taken as a reference for the implementation   
# > https://github.com/abhinavralhan/kohonen-maps/blob/master/som-random.ipynb

# In[ ]:


raw_data = train.transpose()  ## already normalized by feature-wise

network_dimensions = np.array([10, 10])
n_iterations = 10000
init_learning_rate = 0.01

m = raw_data.shape[0]
n = raw_data.shape[1]

# initial neighbourhood radius
init_radius = max(network_dimensions[0], network_dimensions[1]) / 2
# radius decay parameter
time_constant = n_iterations / np.log(init_radius)

data = raw_data


# In[ ]:


net = np.random.random((network_dimensions[0], network_dimensions[1], m))


# In[ ]:


def find_bmu(t, net, m):
    """
        Find the best matching unit for a given vector, t
        Returns: bmu and bmu_idx is the index of this vector in the SOM
    """
    bmu_idx = np.array([0, 0])
    min_dist = np.iinfo(np.int).max
    
    # calculate the distance between each neuron and the input
    for x in range(net.shape[0]):
        for y in range(net.shape[1]):
            w = net[x, y, :].reshape(m, 1)
            sq_dist = np.sum((w - t) ** 2)
            sq_dist = np.sqrt(sq_dist)
            if sq_dist < min_dist:
                min_dist = sq_dist # dist
                bmu_idx = np.array([x, y]) # id
    
    bmu = net[bmu_idx[0], bmu_idx[1], :].reshape(m, 1)
    return (bmu, bmu_idx)


# In[ ]:


def decay_radius(initial_radius, i, time_constant):
    return initial_radius * np.exp(-i / time_constant)

def decay_learning_rate(initial_learning_rate, i, n_iterations):
    return initial_learning_rate * np.exp(-i / n_iterations)

def calculate_influence(distance, radius):
    return np.exp(-distance / (2* (radius**2)))


# In[ ]:


for i in range(n_iterations):
    # select a training example at random
    t = data[:, np.random.randint(0, n)].reshape(np.array([m, 1]))
    
    # find its Best Matching Unit
    bmu, bmu_idx = find_bmu(t, net, m)
    
    # decay the SOM parameters
    r = decay_radius(init_radius, i, time_constant)
    l = decay_learning_rate(init_learning_rate, i, n_iterations)
    
    # update weight vector to move closer to input
    # and move its neighbours in 2-D vector space closer
    
    for x in range(net.shape[0]):
        for y in range(net.shape[1]):
            w = net[x, y, :].reshape(m, 1)
            w_dist = np.sum((np.array([x, y]) - bmu_idx) ** 2)
            w_dist = np.sqrt(w_dist)
            
            if w_dist <= r:
                # calculate the degree of influence (based on the 2-D distance)
                influence = calculate_influence(w_dist, r)
                
                # new w = old w + (learning rate * influence * delta)
                # where delta = input vector (t) - old w
                new_w = w + (l * influence * (t - w))
                net[x, y, :] = new_w.reshape(1, 5)


# In[ ]:


net.shape


# In[ ]:


fig = plt.figure()

ax = fig.add_subplot(111, aspect='equal')
ax.set_xlim((0, net.shape[0]+1))
ax.set_ylim((0, net.shape[1]+1))
ax.set_title('Self-Organising Map after %d iterations' % n_iterations)

# plot
for x in range(1, net.shape[0] + 1):
    for y in range(1, net.shape[1] + 1):
        ax.add_patch(patches.Rectangle((x-0.5, y-0.5), 1, 1,
                     facecolor=net[x-1,y-1,:],
                     edgecolor='none'))
plt.show()

