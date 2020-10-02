#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


get_ipython().system('nvidia-smi')


# In[ ]:


get_ipython().system('ls ../input/')


# In[ ]:


import sys
get_ipython().system('cp ../input/rapids/rapids.0.14.0 /opt/conda/envs/rapids.tar.gz')
get_ipython().system('cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz > /dev/null')
sys.path = ["/opt/conda/envs/rapids/lib/python3.7/site-packages"] + sys.path
sys.path = ["/opt/conda/envs/rapids/lib/python3.7"] + sys.path
sys.path = ["/opt/conda/envs/rapids/lib"] + sys.path 
get_ipython().system('cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/')


# In[ ]:


from sklearn.datasets import make_moons
import pandas


# In[ ]:


X, y  = make_moons(n_samples=int(1e2), noise=0.05, random_state=0)
type(X)


# In[ ]:


get_ipython().run_line_magic('time', "X = pandas.DataFrame({'fea%d'%i: X[:, i]for i in range(X.shape[1])})")


# In[ ]:


# print(y.tolist())
X['class'] = y.tolist()
# X.head()
# fig = px.scatter(X, x="fea0", y="fea1")
# fig.show()
ax2 = X.plot.scatter(x='fea0', y ='fea1')


# In[ ]:


from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps = 0.3, min_samples = 5)


# In[ ]:


get_ipython().run_line_magic('time', "dbscan.fit(X[['fea0', 'fea1']])")


# In[ ]:


get_ipython().run_line_magic('time', 'y_hat = dbscan.fit_predict(X)')


# In[ ]:


X['predict_class'] = y_hat.tolist()


# In[ ]:


X.head()


# In[ ]:


ax3 = X.plot.scatter(x='fea0', y='fea1', c='predict_class', colormap='viridis')


# # Using the same analogy with RAPIDs

# In[ ]:


from sklearn.datasets import make_moons
import cudf


# In[ ]:


X, y = make_moons(n_samples=int(1e2),noise=0.05, random_state=0)


# In[ ]:


get_ipython().run_line_magic('time', "X = cudf.DataFrame({'fea%d'%i: X[:, i]for i in range(X.shape[1])})")


# In[ ]:


# print(y.tolist())
X['class'] = y.tolist()
# X.head()
# fig = px.scatter(X, x="fea0", y="fea1")
# fig.show()
ax2 = X.to_pandas().plot.scatter(x='fea0', y ='fea1')


# In[ ]:


from cuml import DBSCAN
dbscan = DBSCAN(eps = 0.3, min_samples = 5)


# In[ ]:


get_ipython().run_line_magic('time', "dbscan.fit(X[['fea0', 'fea1']])")


# In[ ]:


get_ipython().run_line_magic('time', 'y_hat = dbscan.fit_predict(X)')


# In[ ]:


X['predict_class'] = y_hat.tolist()


# In[ ]:


ax3 = X.to_pandas().plot.scatter(x='fea0', y='fea1', c='predict_class', colormap='viridis')

