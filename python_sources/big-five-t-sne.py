#!/usr/bin/env python
# coding: utf-8

# [Rapids](https://rapids.ai) is an open-source GPU accelerated Data Sceince and Machine Learning library, developed and mainatained by [Nvidia](https://www.nvidia.com). It is designed to be compatible with many existing CPU tools, such as Pandas, scikit-learn, numpy, etc. It enables **massive** acceleration of many data-science and machine learning tasks, oftentimes by a factor fo 100X, or even more. 
# 
# Rapids is still undergoing developemnt, and as of right now it's not availabel in the Kaggle Docker environment. If you are interested in installing and riunning Rapids locally on your own machine, then you should [refer to the followong instructions](https://rapids.ai/start.html).
# 
# In order to install Rapids locally we'll be follow the setup that was inmplemented by Chris Deotte in [the following kernel](https://www.kaggle.com/cdeotte/rapids-data-augmentation-mnist-0-985).

# In[ ]:


get_ipython().run_cell_magic('time', '', '# INSTALL RAPIDS OFFLINE (FROM KAGGLE DATASET). TAKES 1 MINUTE :-)\nimport sys\n!cp ../input/rapids/rapids.0.12.0 /opt/conda/envs/rapids.tar.gz\n!cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz\nsys.path = ["/opt/conda/envs/rapids/lib"] + ["/opt/conda/envs/rapids/lib/python3.6"] + ["/opt/conda/envs/rapids/lib/python3.6/site-packages"] + sys.path\n!cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/')


# In this exercise we'll try to use tSNE for decomposition of dimensionality reduction of the Big Five dataset. The final result is not that exciting - we end up with one big "blob" and a few little remote satelite "dots". But in a way it confirms what has been well know for a while - that personality is really on a continous spectrum, and "typology" is not well founded approach to this subject. 
# 
# Even though the scientific value of this kernel is rather limited, the technical value is still pretty remarkable - thanks to Rapids, it is possible to calculate t-SNE of a million row dataset in just a cuple of minutes. With other algorithms this could potentially take days to do. In other words, Rapids enables you a much faster experimentation than would otherwise be possible. 

# In[ ]:


import cudf, cuml
import cupy as cp
import numpy as np
import pandas as pd
from cuml.manifold import TSNE
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train = pd.read_csv('../input/big-five-personality-test/IPIP-FFM-data-8Nov2018/data-final.csv', sep='\t')


# In[ ]:


X_train = train[train.columns[:50]]
X_train.head()


# In[ ]:


X_train.shape


# In[ ]:


X_train = X_train.dropna()
X_train.shape


# In[ ]:


X_train = X_train.values/5


# In[ ]:


get_ipython().run_cell_magic('time', '', 'tsne = TSNE(n_components=2)\nX_train_2D = tsne.fit_transform(X_train)')


# In[ ]:


# Plot the embedding
plt.scatter(X_train_2D[:,0], X_train_2D[:,1], s = 0.5)

