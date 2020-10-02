#!/usr/bin/env python
# coding: utf-8

# [Rapids](https://rapids.ai) is an open-source GPU accelerated Data Science and Machine Learning library, developed and mainatained by [Nvidia](https://www.nvidia.com). It is designed to be compatible with many existing CPU tools, such as Pandas, scikit-learn, numpy, etc. It enables **massive** acceleration of many data-science and machine learning tasks, oftentimes by a factor fo 100X, or even more. 
# 
# Rapids is still undergoing developemnt, and as of right now it's not availabel in the Kaggle Docker environment. If you are interested in installing and riunning Rapids locally on your own machine, then you shoudl [refer to the followong instructions](https://rapids.ai/start.html).
# 
# The first successful install of a Rapids library on kaggle was done by [Chris Deotte](https://www.kaggle.com/cdeotte) in the follwiong [Digit Recognizer kernel](https://www.kaggle.com/cdeotte/rapids-gpu-knn-mnist-0-97). An improved install version that uses a Kaggle Dataset for install can be found [here](https://www.kaggle.com/cdeotte/rapids-data-augmentation-mnist-0-985).  In this kerenl we'll follow that approach.
# 
# The purpose of this kernel is to showcase the speedup that one gets with t-SNE and UMAP algorithm between the CPU versions and the Rapids version. The t-SNE Sklearn version can be found [here](https://www.kaggle.com/tunguz/fashion-mnist-2d-t-sne/), and the UMAP version can be found [here](https://www.kaggle.com/tunguz/fashion-mnist-2d-umap/)

# In[ ]:


get_ipython().run_cell_magic('time', '', '# INSTALL RAPIDS OFFLINE (FROM KAGGLE DATASET). TAKES 1 MINUTE :-)\nimport sys\n!cp ../input/rapids/rapids.0.12.0 /opt/conda/envs/rapids.tar.gz\n!cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz\nsys.path = ["/opt/conda/envs/rapids/lib"] + ["/opt/conda/envs/rapids/lib/python3.6"] + ["/opt/conda/envs/rapids/lib/python3.6/site-packages"] + sys.path\n!cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/')


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cudf, cuml
from cuml.manifold import UMAP, TSNE
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv('../input/fashionmnist/fashion-mnist_train.csv')
test = pd.read_csv('../input/fashionmnist/fashion-mnist_test.csv')


# In[ ]:


train_test = np.vstack([train, test])
train_test.shape


# In[ ]:


y = train_test[:,0]
train_test = train_test[:,1:]


# In[ ]:


get_ipython().run_cell_magic('time', '', 'tsne = TSNE(n_components=2)\ntrain_test_2D = tsne.fit_transform(train_test)')


# In[ ]:


plt.scatter(train_test_2D[:,0], train_test_2D[:,1], c = y, s = 0.5)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'umap = UMAP(n_components=2)\ntrain_test_2D = umap.fit_transform(train_test)')


# In[ ]:


plt.scatter(train_test_2D[:,0], train_test_2D[:,1], c = y, s = 0.5)


# In[ ]:




