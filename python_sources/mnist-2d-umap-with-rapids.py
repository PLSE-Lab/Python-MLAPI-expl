#!/usr/bin/env python
# coding: utf-8

# [Rapids](https://rapids.ai) is an open-source GPU accelerated Data Science and Machine Learning library, developed and mainatained by [Nvidia](https://www.nvidia.com). It is designed to be compatible with many existing CPU tools, such as Pandas, scikit-learn, numpy, etc. It enables **massive** acceleration of many data-science and machine learning tasks, oftentimes by a factor fo 100X, or even more. 
# 
# Rapids is still undergoing developemnt, and as of right now it's not availabel in the Kaggle Docker environment. If you are interested in installing and riunning Rapids locally on your own machine, then you shoudl [refer to the followong instructions](https://rapids.ai/start.html).
# 
# The first successful install of a Rapids library on kaggle was done by [Chris Deotte](https://www.kaggle.com/cdeotte) in the follwiong [Digit Recognizer kernel](https://www.kaggle.com/cdeotte/rapids-gpu-knn-mnist-0-97). An improved install version that uses a Kaggle Dataset for install can be found [here](https://www.kaggle.com/cdeotte/rapids-data-augmentation-mnist-0-985).  In this kerenl we'll follow that approach.
# 
# The purpose of this kernel is to showcase the speedup that one gets with UMAP algorithm between the UMAP package version and the Rapids version. The UMAP version can be found [here](https://www.kaggle.com/tunguz/mnist-2d-umap)

# In[ ]:


get_ipython().run_cell_magic('time', '', '# INSTALL RAPIDS OFFLINE (FROM KAGGLE DATASET). TAKES 1 MINUTE :-)\nimport sys\n!cp ../input/rapids/rapids.0.11.0 /opt/conda/envs/rapids.tar.gz\n!cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz\nsys.path = ["/opt/conda/envs/rapids/lib"] + ["/opt/conda/envs/rapids/lib/python3.6"] + ["/opt/conda/envs/rapids/lib/python3.6/site-packages"] + sys.path\n!cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/')


# In[ ]:


import cudf, cuml
import pandas as pd
import numpy as np
from cuml.manifold import UMAP
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train = pd.read_csv('../input/digit-recognizer/train.csv')
test = pd.read_csv('../input/digit-recognizer/test.csv')


# In[ ]:


y = train['label'].values
train = train[test.columns].values
test = test[test.columns].values


# In[ ]:


train_test = np.vstack([train, test])
train_test.shape


# In[ ]:


get_ipython().run_cell_magic('time', '', 'umap = UMAP()\ntrain_test_2D = umap.fit_transform(train_test)')


# So we get six seconds instead of ten minutes! That's a substantial speedup. A speedup of over 120X in Kaggle Kernels!
# 
# Now let's visualize to see what the output looks like. We'll have to retrain UMAP on train features only, because we want to see how the embeddings correspond to different digits. 

# In[ ]:


get_ipython().run_cell_magic('time', '', 'umap = UMAP()\ntrain_2D = umap.fit_transform(train)')


# In[ ]:


plt.scatter(train_2D[:,0], train_2D[:,1], c = y, s = 0.5)


# Various colors corrrespond to various digits. We see that same digits are fairly well clustered together, which is probably one of the main reasons why this problem is relatively easy for ML models to do. 

# In[ ]:


train_2D = train_test_2D[:train.shape[0]]
test_2D = train_test_2D[train.shape[0]:]

np.save('train_2D', train_2D)
np.save('test_2D', test_2D)


# In[ ]:




