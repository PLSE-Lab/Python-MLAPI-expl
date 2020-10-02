#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


# In[ ]:


X_unlabeled = pd.read_csv('../input/dataset_X.csv', index_col='id')
X_unlabeled.head()


# In[ ]:


X_unlabeled.describe()


# Overview of variable pair dependencies.

# In[ ]:


sns.pairplot(X_unlabeled.sample(10000), diag_kind='hist')
plt.show()


# **Conclusion**:
# - Don't see anything

# Compress using TSNE to see if we find some pattern in the data... 

# In[ ]:


def plot_2d(df): 
    data = (
        pd.DataFrame(
            {'x': df[:, 0], 
             'y': df[:, 1]})
    )

    sns.lmplot(
        data=data, 
        x='x', 
        y='y', 
        fit_reg=True)

    plt.show()


# In[ ]:


X_embedded = TSNE(n_components=2).fit_transform(X_unlabeled.sample(5000).values)


# In[ ]:


plot_2d(X_embedded)


# **Hypothesis**:
# - Data is (almost) completely useless.
# - Maybe we can ask for dots on circle?

# Let's look more closely at a specific pair.

# In[ ]:


sns.jointplot(
    data=X_unlabeled.sample(100000), 
    x='d4', 
    y='d5', 
    kind='kde')
plt.show()


# **Conclusion**:
# - This notebook proved (almost) completely useless.
# - Maybe we don't need all features.
# - From describe it seems like 4, 5, 8 and 10 are similar features, and also 1, 2, 3, 6, 7 and 9 are from a slightly different distribution. 

# In[ ]:


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X=X_unlabeled)


# In[ ]:


plot_2d(X_pca[0:10000, :])

