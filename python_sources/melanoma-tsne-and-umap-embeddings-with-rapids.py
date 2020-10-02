#!/usr/bin/env python
# coding: utf-8

# [Rapids](https://rapids.ai) is an open-source GPU accelerated Data Science and Machine Learning library, developed and mainatained by [Nvidia](https://www.nvidia.com). It is designed to be compatible with many existing CPU tools, such as Pandas, scikit-learn, numpy, etc. It enables **massive** acceleration of many data-science and machine learning tasks, oftentimes by a factor fo 100X, or more. 
# 
# Rapids is still undergoing developemnt, and as of right now it's not availabel in the Kaggle Docker environment. If you are interested in installing and running Rapids locally on your own machine, then you should [refer to the followong instructions](https://rapids.ai/start.html).
# 
# The purpose of this kernel is to take a look at dimensionality reduction that one gets with t-SNE and UMAP algorithms. We will apply these algorithms to the pixel-level data, in hopes of discerning some patterns.

# In[ ]:





# In[ ]:


get_ipython().run_cell_magic('time', '', '# INSTALL RAPIDS OFFLINE (FROM KAGGLE DATASET). TAKES 1 MINUTE :-)\nimport sys\n!cp ../input/rapids/rapids.0.13.0 /opt/conda/envs/rapids.tar.gz\n!cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz\nsys.path = ["/opt/conda/envs/rapids/lib"] + ["/opt/conda/envs/rapids/lib/python3.6"] + ["/opt/conda/envs/rapids/lib/python3.6/site-packages"] + sys.path\n!cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/')


# In[ ]:


import cudf, cuml
import cupy as cp
import numpy as np
import pandas as pd
import os
from cuml.manifold import TSNE, UMAP
import matplotlib.pyplot as plt
from matplotlib.pyplot import ylim, xlim
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train_df = pd.read_csv("../input/siim-isic-melanoma-classification/train.csv")
train = np.load('../input/siimisic-melanoma-resized-images/x_train_32.npy')


# In[ ]:


train_df.head()


# In[ ]:


train = train.reshape((train.shape[0], 32*32*3))
train.shape


# In[ ]:


get_ipython().run_cell_magic('time', '', 'tsne = TSNE(n_components=2)\ntrain_2D = tsne.fit_transform(train)')


# In[ ]:


plt.scatter(train_2D[:,0], train_2D[:,1], c=train_df['target'].values, s = 0.8)


# We see that the image pixel data definitley has some clustering structure. At first glance it doesn't seem that it's easy to clearly separate the target cases, partly becasue they comprise less than 2% of all the points. However, it seems like the most of them are concentrated in the lower areas, and soem clusters have more of them than the rest.
# 
# Let's now take a look at what UMAP can discern.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'umap = UMAP(n_components=2)\ntrain_2D = umap.fit_transform(train)')


# In[ ]:


plt.scatter(train_2D[:,0], train_2D[:,1], c=train_df['target'].values, s = 0.8)


# Much more interesting and deliniated clusters. Again, no simple clustering of the positive cases, but they do show spatial concentration.
