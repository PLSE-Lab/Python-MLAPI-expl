#!/usr/bin/env python
# coding: utf-8

# **Introduction**
# 
# This tutorial will show you, how kepler.lg (Uber) can be used to generate beautilful and meaningful geospatial visualizations on kaggle without custom docker images. So defuault environment needs few installations.
# 
# 1. Install keplergl using pip
# 2. Install latest nodejs
# 3. Install labextensions
# 4. Load the map after committing the kernel.

# **Workspace setting**

# In[ ]:


## Install kelper
get_ipython().system('pip install keplergl')


# In[ ]:


get_ipython().system('jupyter --version')


# In[ ]:


# Install nodejs using conda on the environment (will be needed for installing labextension)   
# Other options - https://anaconda.org/conda-forge/nodejs   
get_ipython().system('conda install -y -c conda-forge/label/cf202003 nodejs')


# In[ ]:


get_ipython().system('node --version')


# In[ ]:


get_ipython().system('python --version')


# In[ ]:


get_ipython().system('jupyter labextension install @jupyter-widgets/jupyterlab-manager keplergl-jupyter')


# In[ ]:


# Load an empty map
from keplergl import KeplerGl
map_1 = KeplerGl()
map_1


# **Commit the kernel, you will see the map loaded, this wont work in edit mode**
# 
# **If you need to see the map on edit mode, use a custom docker image as done in https://www.kaggle.com/rahulvks/kepler-for-jupyter**
# 
# **Just copy paste https://www.kaggle.com/rahulvks/kepler-for-jupyter kernel will also work since it will use the custom docker image he used.**
