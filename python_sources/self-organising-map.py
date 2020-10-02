#!/usr/bin/env python
# coding: utf-8

# What are Self-organising Maps (SOMS)?  That's a question I always stuggle to answer.  Are they a culstering algorithm, are they meant for dimensionality reduction, are they meant just for visualization? They are used for all of the above and what's wild is that they are an artificial neural network unlike any you have seen before. I have often been asked to describe SOMS to people before and have struggled, but a year ago discovered a great analogy for how SOMs work.  
#   
# A common assumption in the Data Sciences is the Manifold Assumption. This assumption proposes that that data which exists in high-dimensions actually exists on a low-dimensional subspace. For humans, we live on a low dimensional subspace. The earth is a big 3d ball but we only live on the surface and seldom travel straight through the centre of earths melton core. Using SOMs we are looking to model the data not in the original space but in this subspace.  In order to do this we initialize a lattice and slowly, but surely, move and spread this net over our data points. This process of spreading relies on a two methods of competitive and cooperative learning, and is can be thought of as a mechanism to wrap our subspace using our lattice- like wrapping a ball with a piece of paper. 
# 
# Using our competitive and cooperative learning steps, we slowly iterate through our dataset, find the node on our lattice closest to it and move both it and its neighbours along the lattice closer to that datapoints. This learning rate, the mechanism for deciding which neighbours get moved and by how much and the shape of our lattice are the critical hyperparameters which are critical to the success and insights of SOM.  

# In[ ]:


from typing import Optional, Tuple

import pandas as pd
import numpy as np
import networkx as nx

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics import pairwise_distances
from sklearn.datasets import make_blobs, make_regression
import holoviews as hv

hv.extension('bokeh')


# In[ ]:


class SOM(BaseEstimator, ClusterMixin):
    def __init__(self, shape: Tuple[int] = (3, 3), 
                 lattice: str = 'hexagonal_lattice_graph', 
                 n_iter: int = 100, 
                 learning_rate: float = 0.25, 
                 cooperative_learning_rate: float = 0.125, 
                 n_jobs: Optional[int] = None):
        
        self.lattice = lattice
        self.shape = shape
        self.learning_rate = learning_rate
        self.cooperative_learning_rate = cooperative_learning_rate
        self.n_jobs = n_jobs
        self.n_iter = n_iter
        
    def fit(self, X: np.ndarray):
        # initialize graph
        self.graph_ = getattr(nx.generators.lattice, self.lattice)(*self.shape)
        
        # initialize weights
        mean, var = X.mean(0), X.var(0)
        for key in self.graph_.nodes:
            self.graph_.nodes[key]['weight'] = np.random.normal(mean, var)
            
        for schedule in range(1, self.n_iter):
            for x in X:
                weights_dict = nx.get_node_attributes(self.graph_, 'weight')
                self.weights_ = np.vstack(list(weights_dict.values()))

                # competition
                argmin = int(pairwise_distances(X = x.reshape(1, -1), Y = self.weights_,
                                                n_jobs=self.n_jobs)
                             .argmin())
                min_node_key = list(weights_dict.keys())[argmin]

                self.graph_.nodes[key]['weight'] -= self.learning_rate * (self.graph_.nodes[key]['weight'] - x) / schedule

                # cooperation
                for key in self.graph_.neighbors(min_node_key):
                    self.graph_.nodes[key]['weight'] -= self.cooperative_learning_rate * (self.graph_.nodes[key]['weight'] - x) / schedule
                    
    def predict(self, X):
        return (pairwise_distances(X = X,
                                   Y = self.weights_,
                                   n_jobs=self.n_jobs)
                .argmin(1))
        
            
            


# In order to better understand this idea of wrapping a manifold with a lattice, we are going to have to generate data. Here we are going to take our classical S-curve manifold and try to generate Gaussian Blobs along the surface of the this manifold. 

# In[ ]:


def get_blobs_on_scurve(n_samples = 2500,
                        noise = 0.01,
                        centers=2):
    x_gaussian_blobs, y_gaussian_blobs = make_blobs(n_samples,  n_features=1, centers=centers)
    x_gaussian_blobs = x_gaussian_blobs.flatten()
    
    clipped_ = (x_gaussian_blobs - x_gaussian_blobs.min())/x_gaussian_blobs.max()

    t = 3 * np.pi * ( clipped_ - clipped_.mean() )
    x = np.sin(t)
    y = 2.0 * (np.random.rand(1, n_samples) - 0.5)
    z = np.sign(t) * (np.cos(t) - 1)

    X = np.column_stack((x.reshape(-1,1),
                         y.reshape(-1,1),
                         z.reshape(-1,1)))
    X += noise * np.random.randn(1, n_samples).reshape(-1,1)
    t = np.squeeze(t)
    
    return X, y_gaussian_blobs

X, y = get_blobs_on_scurve()


# In[ ]:


hv.extension('matplotlib')
(hv.Scatter3D(pd.DataFrame(X, columns=['x','y','z'])
              .assign(cluster=y), kdims=['x','y'], vdims=['z', 'cluster'])
 .opts(color='cluster', title='Blobs along S-curve Manifold'))


# After fitting this model, we can then compute the distances between neighbours along the lattice to visualize the density accross this lattice and Voila. 

# In[ ]:


som = SOM((15, 15), 'grid_2d_graph', learning_rate= 0.2, cooperative_learning_rate=0.1, n_iter=25, n_jobs=None)
som.fit(X)


# In[ ]:


hv.extension('bokeh')
grid = np.zeros(som.shape)
for key in som.graph_.nodes:
    row, column = list(key)
    for neighbour in som.graph_.neighbors(key):
        grid[row, column] += pairwise_distances(X = som.graph_.nodes[key]['weight'].reshape(1, -1),
                                                Y = som.graph_.nodes[neighbour]['weight'].reshape(1, -1))
        
hv.Image(grid).opts(title='Density along Lattices')

