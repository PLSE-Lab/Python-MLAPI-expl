#!/usr/bin/env python
# coding: utf-8

# # About this kernel
# 
# In this kernel, I show you:
# 
# * How to embed each tweet into a high-dimensional vector (512d) using *Universal Sentence Encoder (USE)*. This is done in **Tensorflow Hub**.
# * How to project each of those tweets from high-dimension to low-dimension (2d and 3d), using techniques like *t-SNE* and *UMAP*. This will be done using **RAPIDS.ai**, a framework that let you run all kinds of ML algorithms (not DL specific) directly on GPU.
# * Visualize t-SNE and UMAP interactively using 2D and 3D *scatter plots* with **Plotly**. You can hover over each dot to see the content!

# # Step 1: Install RAPIDS.ai
# First install RAPIDS offline (directly taken from [this kernel](https://www.kaggle.com/tunguz/cats-ii-t-sne-and-umap-embeddings-with-rapids/data)):

# In[ ]:


get_ipython().run_cell_magic('time', '', 'import sys\n!cp ../input/rapids/rapids.0.12.0 /opt/conda/envs/rapids.tar.gz\n!cd /opt/conda/envs/ && tar -xzf rapids.tar.gz\nsys.path = ["/opt/conda/envs/rapids/lib"] + ["/opt/conda/envs/rapids/lib/python3.6"] + ["/opt/conda/envs/rapids/lib/python3.6/site-packages"] + sys.path\n!cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/')


# In[ ]:


import cuml
from cuml.manifold import TSNE, UMAP
import plotly.express as px
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


# # Step 2 - Embed sentences into vector with USE in TFHub
# 
# Universal Sentence Encoder (USE) is a practical.

# In[ ]:


train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")


# In[ ]:


get_ipython().run_cell_magic('time', '', "module_url = '/kaggle/input/universalsentenceencoderlarge4/'\nembed = hub.KerasLayer(module_url, trainable=True, name='USE_embedding')")


# In[ ]:


get_ipython().run_cell_magic('time', '', "encodings = embed(train.text)['outputs'].numpy()")


# # Step 3 - Run t-SNE with RAPIDS.ai

# In[ ]:


get_ipython().run_cell_magic('time', '', 'tsne2d = TSNE(n_components=2)\nprojections_2d = tsne2d.fit_transform(encodings)')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'umap3d = UMAP(n_components=3)\nprojections_3d = umap3d.fit_transform(encodings)')


# # Step 4 - Visualize with Plotly

# In[ ]:


labels = train.target.apply(lambda x: 'Real Disaster' if x else 'Not Disaster')


# ## 2D visualization with t-SNE

# In[ ]:


fig = px.scatter(
    x=projections_2d[:, 0], y=projections_2d[:, 1], 
    color=labels, hover_name=train.text, height=700
)
fig.show()


# ## 3D Visualization with UMAP

# In[ ]:


fig = px.scatter_3d(
    x=projections_3d[:, 0], y=projections_3d[:, 1], z=projections_3d[:, 2], 
    color=labels, hover_name=train.text, size_max=2, height=700
)
fig.update_traces(marker=dict(size=3))

fig.show()

