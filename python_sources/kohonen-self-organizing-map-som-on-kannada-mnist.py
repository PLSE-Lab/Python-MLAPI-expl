#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import seaborn as sns
from imageio import imwrite
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cbook import get_sample_data
import random
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import StandardScaler
from PIL import Image, ImageChops


# ## What are SOM?
# In brief, Self-organizing maps are a type of artificial neural network based on competitive learning (at variance to error-correcting learning typical of other NNs). The idea is to iteratively adapt a connected two-dimensional matrix of vectors (or nodes) to the higher-dimensional topology of the input dataset. At each cycle, a node is selected and its elements (the weights) are updated, together with those of its neighbors, to approach a randomly chosen datapoint from the training set. The competitive element comes into play during the update stage, since the closest node (according to a chosen metric) to the extracted datapoint is selected for the weights update at each iteration.
# 
# SOMs are particularly suited for cases where low-dimensional manifolds are hidden in higher dimensions and are often used together and/or competing with other dimensionality reduction methods and in particular Principal Component Analysis (PCA) for which it could be seen as a non-linear generalization: an exhaustive explanation of SOM's advantages and disadvantages, however, is beyond the scope of this notebook, but there are plenty of resources online for those who would like to know more.

# I've used [this](http://https://github.com/fcomitani/SimpSOM) implementation of Kohonen Self-Organizing Maps

# In[ ]:


get_ipython().system('pip install SimpSOM')


# In[ ]:


import SimpSOM as sps


# ## Data preparation

# In[ ]:


np.random.seed(0)

# get part of the dataset
train = pd.read_csv('../input/Kannada-MNIST/train.csv')
train = train.sample(n=600, random_state=0)
labels = train['label']
train = train.drop("label",axis=1)

# check distribution
sns.countplot(labels)

# standardization of a dataset
train_st = StandardScaler().fit_transform(train.values)


# ## Training SOM

# In[ ]:


# build a network 20x20 with a weights format taken from the train_st and activate Periodic Boundary Conditions. 
som = sps.somNet(20, 20, train_st, PBC=True)

# train it with 0.1 learning rate for 10000 epochs
som.train(0.05, 10000)

# print unified distance matrix
som.diff_graph(show=True, printout=True)


# In[ ]:





# ## Visualizing some nodes of the map

# In[ ]:


fig, axs = plt.subplots(10, 10, figsize=(20, 20))
axs = axs.flatten()

some_nodes_indxs = random.sample(range(len(som.nodeList)), len(axs))


for i, ax in enumerate(axs):
    ax.imshow(np.asarray(som.nodeList[some_nodes_indxs[i]].weights).reshape(28,28))
    ax.axis('off')


# In[ ]:


# print picked coordinates
', '.join([f'({i // 20}, {i % 20})' for i in some_nodes_indxs])


# In[ ]:





# In[ ]:





# In[ ]:


som.cluster(train_st, type='qthresh', show=True);


# In[ ]:





# I hope you enjoyed the kernel. In case you are interested in understanding of this algorithm, I would recommend this <a href="https://www.youtube.com/watch?v=lFbxTlD5R98">lecture</a>

# In[ ]:




