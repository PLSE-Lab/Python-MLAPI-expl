#!/usr/bin/env python
# coding: utf-8

# [Rapids](https://rapids.ai) is an open-source GPU accelerated Data Science and Machine Learning library, developed and mainatained by [Nvidia](https://www.nvidia.com). It is designed to be compatible with many existing CPU tools, such as Pandas, scikit-learn, numpy, etc. It enables **massive** acceleration of many data-science and machine learning tasks, oftentimes by a factor fo 100X, or even more. 
# 
# Rapids is still undergoing developemnt, and as of right now it's not availabel in the Kaggle Docker environment. If you are interested in installing and riunning Rapids locally on your own machine, then you shoudl [refer to the followong instructions](https://rapids.ai/start.html).
# 
# The first successful install of a Rapids library on kaggle was done by [Chris Deotte](https://www.kaggle.com/cdeotte) in the follwiong [Digit Recognizer kernel](https://www.kaggle.com/cdeotte/rapids-gpu-knn-mnist-0-97). An improved install version that uses a Kaggle Dataset for install can be found [here](https://www.kaggle.com/cdeotte/rapids-data-augmentation-mnist-0-985).  In this kerenl we'll follow that approach.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().run_cell_magic('time', '', '# INSTALL RAPIDS OFFLINE (FROM KAGGLE DATASET). TAKES 1 MINUTE :-)\nimport sys\n!cp ../input/rapids/rapids.0.12.0 /opt/conda/envs/rapids.tar.gz\n!cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz\nsys.path = ["/opt/conda/envs/rapids/lib"] + ["/opt/conda/envs/rapids/lib/python3.6"] + ["/opt/conda/envs/rapids/lib/python3.6/site-packages"] + sys.path\n!cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/')


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import cudf, cuml
import cupy as cp
from cuml.linear_model import LogisticRegression


# In[ ]:


train = pd.read_csv('../input/digit-recognizer/train.csv')
submission = pd.read_csv('../input/digit-recognizer/sample_submission.csv')


# In[ ]:


train.head()


# In[ ]:


submission.head()


# In[ ]:


y = train['label'].values


# In[ ]:


y


# In[ ]:


train_tsne = np.load('../input/mnist-2d-t-sne-with-rapids/train_2D.npy')
test_tsne = np.load('../input/mnist-2d-t-sne-with-rapids/test_2D.npy')


# In[ ]:


train_x, val_x, train_y, val_y = train_test_split(train_tsne, y, test_size=0.10)


# In[ ]:


clf = LogisticRegression(C = 0.1)
clf.fit(train_x, train_y.astype('float32'))


# In[ ]:


preds = clf.predict(val_x)


# In[ ]:


np.mean(cp.array(val_y) == preds.values.astype('int64'))


# In[ ]:


train_umap = np.load('../input/mnist-2d-umap-with-rapids/train_2D.npy')
test_umap = np.load('../input/mnist-2d-umap-with-rapids/test_2D.npy')


# In[ ]:


train_x, val_x, train_y, val_y = train_test_split(train_umap, y, test_size=0.10)


# In[ ]:


clf = LogisticRegression(C = 12)
clf.fit(train_x, train_y.astype('float64'))
preds = clf.predict(val_x)
np.mean(cp.array(val_y) == preds.values.astype('int64'))


# In[ ]:


test_preds = clf.predict(test_umap)


# In[ ]:


train_y.astype('float32')


# In[ ]:


train_both = np.hstack([train_umap, train_tsne])
test_both = np.hstack([test_umap, test_tsne])


# In[ ]:


train_x, val_x, train_y, val_y = train_test_split(train_both, y, test_size=0.10)


# In[ ]:


clf = LogisticRegression(C = 1)
clf.fit(train_x, train_y.astype('float64'))
preds = clf.predict(val_x)
np.mean(cp.array(val_y) == preds.values.astype('int64'))


# In[ ]:


#test_preds = clf.predict(test_both)


# In[ ]:


cp.asnumpy(test_preds.values.astype('int64'))


# In[ ]:


submission['Label'] = cp.asnumpy(test_preds.values.astype('int64'))


# In[ ]:


submission.to_csv('submission.csv', index=False)


# In[ ]:


submission.head()


# In[ ]:




