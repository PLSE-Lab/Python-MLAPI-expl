#!/usr/bin/env python
# coding: utf-8

# [Rapids](https://rapids.ai) is an open-source GPU accelerated Data Science and Machine Learning library, developed and mainatained by [Nvidia](https://www.nvidia.com). It is designed to be compatible with many existing CPU tools, such as Pandas, scikit-learn, numpy, etc. It enables **massive** acceleration of many data-science and machine learning tasks, oftentimes by a factor fo 100X, or even more. 
# 
# Rapids is still undergoing developemnt, and as of right now it's not availabel in the Kaggle Docker environment. If you are interested in installing and riunning Rapids locally on your own machine, then you shoudl [refer to the followong instructions](https://rapids.ai/start.html).
# 
# The first successful install of a Rapids library on kaggle was done by [Chris Deotte](https://www.kaggle.com/cdeotte) in the follwiong [Digit Recognizer kernel](https://www.kaggle.com/cdeotte/rapids-gpu-knn-mnist-0-97). An improved install version that uses a Kaggle Dataset for install can be found [here](https://www.kaggle.com/cdeotte/rapids-data-augmentation-mnist-0-985).  In this kerenl we'll follow that approach.
# 
# The purpose of this kernel is to showcase the speedup that one gets with t-SNE and UMAP algorithms that one gets with Rapids. Each one of those algorithsm can provide th embedding in about a minute, which is just incredible for a combined datast of 1,000,000 datapoints. The starting datasets are the processed and target-encoded datasets that can be found [here](https://www.kaggle.com/tunguz/multi-cat-encodings)

# In[ ]:


get_ipython().run_cell_magic('time', '', '# INSTALL RAPIDS OFFLINE (FROM KAGGLE DATASET). TAKES 1 MINUTE :-)\nimport sys\n!cp ../input/rapids/rapids.0.12.0 /opt/conda/envs/rapids.tar.gz\n!cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz\nsys.path = ["/opt/conda/envs/rapids/lib"] + ["/opt/conda/envs/rapids/lib/python3.6"] + ["/opt/conda/envs/rapids/lib/python3.6/site-packages"] + sys.path\n!cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/')


# In[ ]:




# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cudf, cuml
from cuml.manifold import UMAP, TSNE
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train = pd.read_csv('../input/multi-cat-encodings/X_train_te.csv')
test = pd.read_csv('../input/multi-cat-encodings/X_test_te.csv')
sample_submission = pd.read_csv('../input/cat-in-the-dat-ii/sample_submission.csv')
y = np.load('/kaggle/input/multi-cat-encodings/target.npy')
features = test.columns


# In[ ]:


train = train[features].values
test = test[features].values
train_test = np.vstack([train, test])
train_test.shape


# In[ ]:


get_ipython().run_cell_magic('time', '', 'tsne = TSNE(n_components=2)\ntrain_test_2D = tsne.fit_transform(train_test)')


# In[ ]:


train_2D = train_test_2D[:train.shape[0]]
test_2D = train_test_2D[train.shape[0]:]

np.save('train_tsne_2D', train_2D)
np.save('test_tsne_2D', test_2D)


# In[ ]:


plt.scatter(train_2D[:,0], train_2D[:,1], c = y, s = 0.5)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'umap = UMAP(n_components=2)\ntrain_test_2D = umap.fit_transform(train_test)')


# In[ ]:


train_2D = train_test_2D[:train.shape[0]]
test_2D = train_test_2D[train.shape[0]:]

np.save('train_umap_2D', train_2D)
np.save('test_umap_2D', test_2D)


# In[ ]:


plt.scatter(train_2D[:,0], train_2D[:,1], c = y, s = 0.5)


# In[ ]:




