#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train = pd.read_csv('../input/Kannada-MNIST/train.csv')
test = pd.read_csv('../input/Kannada-MNIST/test.csv')


# In[ ]:


train.head()


# In[ ]:


y = train['label'].values
train = train[test.columns[1:]].values
test = test[test.columns[1:]].values


# In[ ]:


train_test = np.vstack([train, test])
train_test.shape


# In[ ]:


get_ipython().run_cell_magic('time', '', 'tsne = TSNE(n_components=2)\ntrain_test_2D = tsne.fit_transform(train_test)')


# In[ ]:


train_2D = train_test_2D[:train.shape[0]]
test_2D = train_test_2D[train.shape[0]:]


# In[ ]:


plt.scatter(train_2D[:,0], train_2D[:,1], c = y, s = 0.5)


# In[ ]:


np.save('train_2D', train_2D)
np.save('test_2D', test_2D)

