#!/usr/bin/env python
# coding: utf-8

# [Rapids](https://rapids.ai) is an open-source GPU accelerated Data Science and Machine Learning library, developed and mainatained by [Nvidia](https://www.nvidia.com). It is designed to be compatible with many existing CPU tools, such as Pandas, scikit-learn, numpy, etc. It enables **massive** acceleration of many data-science and machine learning tasks, oftentimes by a factor fo 100X, or even more. 
# 
# Rapids is still undergoing developemnt, and as of right now it's not availabel in the Kaggle Docker environment. If you are interested in installing and riunning Rapids locally on your own machine, then you shoudl [refer to the followong instructions](https://rapids.ai/start.html).
# 
# 
# The purpose of this kernel is to take alook at dimensionality reduction that one gets with t-SNE and UMAP algorithms. TReNDS data is very high-dimensional, and it may not be easy to understand what is going on. The set of features that we are using are the ones that have proven useful in some of the top-scoring public kernels.

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


fnc_df = pd.read_csv("../input/trends-assessment-prediction/fnc.csv")
loading_df = pd.read_csv("../input/trends-assessment-prediction/loading.csv")

fnc_features, loading_features = list(fnc_df.columns[1:]), list(loading_df.columns[1:])
df = fnc_df.merge(loading_df, on="Id")


labels_df = pd.read_csv("../input/trends-assessment-prediction/train_scores.csv")
labels_df["is_train"] = True

df = df.merge(labels_df, on="Id", how="left")

test_df = df[df["is_train"] != True].copy()
df = df[df["is_train"] == True].copy()

df.shape, test_df.shape


# In[ ]:


# Giving less importance to FNC features since they are easier to overfit due to high dimensionality.
FNC_SCALE = 1/600

df[fnc_features] *= FNC_SCALE
test_df[fnc_features] *= FNC_SCALE

features = loading_features + fnc_features


# In[ ]:


train_test = np.vstack([df[features], test_df[features]])
train_test.shape


# In[ ]:


get_ipython().run_cell_magic('time', '', 'tsne = TSNE(n_components=2)\ntrain_test_2D = tsne.fit_transform(train_test)')


# In[ ]:


plt.scatter(train_test_2D[:,0], train_test_2D[:,1], s = 0.5)


# There is some structure there, with "central" denser reagion, and more dispersed periphery, but other than that it's hard to see any distinct groupings. 

# In[ ]:



get_ipython().run_cell_magic('time', '', 'tsne = TSNE(n_components=2)\n\ntrain_2D = tsne.fit_transform(df[features].values)')


# In[ ]:


plt.scatter(train_2D[:,0], train_2D[:,1], s = 0.5)


# In[ ]:


df['age'].values


# In[ ]:


plt.scatter(train_2D[:,0], train_2D[:,1], c=df['age'].values, s = 0.5)


# In[ ]:


plt.scatter(train_2D[:,0], train_2D[:,1], c=df['domain1_var1'].values, s = 0.5)


# In[ ]:


plt.scatter(train_2D[:,0], train_2D[:,1], c=df['domain1_var2'].values, s = 0.5)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'umap = UMAP(n_components=2)\ntrain_test_2D = umap.fit_transform(train_test)')


# In[ ]:


plt.scatter(train_test_2D[:,0], train_test_2D[:,1], s = 0.5)


# In[ ]:



get_ipython().run_cell_magic('time', '', 'umap = UMAP(n_components=2)\n\ntrain_2D = umap.fit_transform(df[features].values)')


# In[ ]:


plt.scatter(train_2D[:,0], train_2D[:,1], s = 0.5)


# In[ ]:


plt.scatter(train_2D[:,0], train_2D[:,1], c=df['age'].values, s = 0.5)


# In[ ]:




