#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
get_ipython().system('cp ../input/rapids/rapids.0.11.0 /opt/conda/envs/rapids.tar.gz > /dev/null')
get_ipython().system('cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz > /dev/null')
sys.path = ["/opt/conda/envs/rapids/lib/python3.6/site-packages"] + sys.path
sys.path = ["/opt/conda/envs/rapids/lib/python3.6"] + sys.path
sys.path = ["/opt/conda/envs/rapids/lib"] + sys.path 
get_ipython().system('cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/ > /dev/null')


# In[ ]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = [10, 10]

import cuml.manifold    as tsne_rapids
import sklearn.manifold as tsne_sklearn


# In[ ]:


train = pd.read_csv('../input/otto-group-product-classification-challenge/train.csv')
train = train.sample(15000)
train.shape


# In[ ]:


train.head()


# In[ ]:


y = np.array( [int(v.split('_')[1]) for v in train.target.values ] )
train.drop( ['id','target'], inplace=True, axis=1 )


# In[ ]:


get_ipython().run_cell_magic('time', '', 'tsne = tsne_sklearn.TSNE(n_components=2, random_state=2020 )\ntrain_2D_sklearn = tsne.fit_transform( train.values )')


# In[ ]:


plt.scatter(train_2D_sklearn[:,0], train_2D_sklearn[:,1], c = y, s = 0.5)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'tsne = tsne_rapids.TSNE(n_components=2, perplexity=30, random_state=2020 )\ntrain_2D_rapids = tsne.fit_transform( train.values )')


# In[ ]:


plt.scatter(train_2D_rapids[:,0], train_2D_rapids[:,1], c = y, s = 0.5)

