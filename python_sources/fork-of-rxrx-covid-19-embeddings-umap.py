#!/usr/bin/env python
# coding: utf-8

# [Rapids](https://rapids.ai) is an open-source GPU accelerated Data Sceince and Machine Learning library, developed and mainatained by [Nvidia](https://www.nvidia.com). It is designed to be compatible with many existing CPU tools, such as Pandas, scikit-learn, numpy, etc. It enables **massive** acceleration of many data-science and machine learning tasks, oftentimes by a factor fo 100X, or even more. 
# 
# Rapids is still undergoing developemnt, and as of right now it's not availabel in the Kaggle Docker environment. If you are interested in installing and riunning Rapids locally on your own machine, then you should [refer to the followong instructions](https://rapids.ai/start.html).
# 
# In order to install Rapids locally we'll be follow the setup that was inmplemented by Chris Deotte in [the following kernel](https://www.kaggle.com/cdeotte/rapids-data-augmentation-mnist-0-985).

# In[ ]:


get_ipython().run_cell_magic('time', '', '# INSTALL RAPIDS OFFLINE (FROM KAGGLE DATASET). TAKES 1 MINUTE :-)\nimport sys\n!cp ../input/rapids/rapids.0.13.0 /opt/conda/envs/rapids.tar.gz\n!cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz\nsys.path = ["/opt/conda/envs/rapids/lib"] + ["/opt/conda/envs/rapids/lib/python3.6"] + ["/opt/conda/envs/rapids/lib/python3.6/site-packages"] + sys.path\n!cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/')


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import cudf, cuml
import cupy as cp
import numpy as np
import pandas as pd
from cuml.manifold import UMAP  
import matplotlib.pyplot as plt
from matplotlib.pyplot import ylim, xlim
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().run_cell_magic('time', '', "embeddings = pd.read_csv('../input/rxrx19a/embeddings.csv')")


# In[ ]:


metadata = pd.read_csv('../input/rxrx19a/metadata.csv')
metadata.head()


# In[ ]:


metadata['disease_condition'].unique()


# In[ ]:


embeddings = embeddings[embeddings.columns[1:]].values


# In[ ]:


get_ipython().run_cell_magic('time', '', 'umap = UMAP(n_components=2)\nembeddings_2D = umap.fit_transform(embeddings)')


# In[ ]:


plt.scatter(embeddings_2D[:,0], embeddings_2D[:,1])


# In[ ]:




