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


import matplotlib.pyplot as plt
from sklearn.datasets import load_sample_image


# In[ ]:


img = load_sample_image('flower.jpg')


# In[ ]:


plt.imshow(img)


# In[ ]:


img = img/255


# In[ ]:


X = img.reshape((-1, 3)).copy()


# In[ ]:


from sklearn.cluster import KMeans, MiniBatchKMeans


# In[ ]:


kmeans = KMeans(n_clusters=32)


# In[ ]:


kmeans.fit(X)


# In[ ]:


X.shape


# In[ ]:


mnkmeans = MiniBatchKMeans(n_clusters=32)


# In[ ]:


mnkmeans.fit(X)


# In[ ]:


new_img = mnkmeans.cluster_centers_[mnkmeans.labels_]


# In[ ]:


new_img = new_img.reshape((427, 640, 3))


# In[ ]:


new_img.shape


# In[ ]:


fg, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(new_img)
axes[1].imshow(img)


# In[ ]:


kmn = MiniBatchKMeans(n_clusters=64)
kmn.fit(img.reshape(-1, 3))
new_img = kmn.cluster_centers_[kmn.labels_]
fg, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(new_img.reshape(427, 640 ,3))
axes[1].imshow(img)


# In[ ]:


kmn = MiniBatchKMeans(n_clusters=16)
kmn.fit(img.reshape(-1, 3))
new_img = kmn.cluster_centers_[kmn.labels_]
fg, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(new_img.reshape(427, 640 ,3))
axes[1].imshow(img)


# In[ ]:


kmn = MiniBatchKMeans(n_clusters=8)
kmn.fit(img.reshape(-1, 3))
new_img = kmn.cluster_centers_[kmn.labels_]
fg, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(new_img.reshape(427, 640 ,3))
axes[1].imshow(img)


# In[ ]:


kmn = MiniBatchKMeans(n_clusters=4)
kmn.fit(img.reshape(-1, 3))
new_img = kmn.cluster_centers_[kmn.labels_]
fg, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(new_img.reshape(427, 640 ,3))
axes[1].imshow(img)


# In[ ]:


kmn = MiniBatchKMeans(n_clusters=256)
kmn.fit(img.reshape(-1, 3))
new_img = kmn.cluster_centers_[kmn.labels_]
fg, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(new_img.reshape(427, 640 ,3))
axes[1].imshow(img)


# In[ ]:


kmn = MiniBatchKMeans(n_clusters=2)
kmn.fit(img.reshape(-1, 3))
new_img = kmn.cluster_centers_[kmn.labels_]
fg, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(new_img.reshape(427, 640 ,3))
axes[1].imshow(img)


# In[ ]:




