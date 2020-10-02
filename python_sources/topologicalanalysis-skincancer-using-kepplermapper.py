#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from skimage.io import imread

import cv2


# In[ ]:


mal_images = glob('../input/skin-cancer-malignant-vs-benign/data/train/malignant/*')
ben_images = glob('../input/skin-cancer-malignant-vs-benign/data/train/benign/*')


# In[ ]:


len(mal_images)


# In[ ]:


mal_images[0:5]


# In[ ]:


print(len(ben_images))
ben_images[0:5]


# In[ ]:


benign=pd.DataFrame()
labels = []
for imagePath in ben_images:
  column_name=imagePath.split('/')[-1].split('.')[0]
  image=cv2.imread(imagePath)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  benign[column_name]= image.flatten()
  labels.append('0')


# In[ ]:


benign.shape


# In[ ]:


benign=benign.transpose()


# In[ ]:


benign['label']=labels


# In[ ]:


labels_2 = []
malignant=pd.DataFrame()

for imagePath in mal_images:
  column_name=imagePath.split('/')[-1]
  image=cv2.imread(imagePath)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  # image=cv2.resize(image, (300, 300),interpolation=cv2.INTER_AREA)
  malignant[column_name]=image.flatten()
  labels_2.append('1')


# In[ ]:


malignant=malignant.transpose()


# In[ ]:


malignant['label']=labels_2


# In[ ]:


df = pd.concat([benign, malignant])


# In[ ]:


df['label'].value_counts()


# In[ ]:


Xtrain=df.drop(['label'],axis=1)


# In[ ]:


get_ipython().system('pip install kmapper')


# In[ ]:


import kmapper as km

import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn import datasets


# In[ ]:


# Initialize to use t-SNE with 2 components (reduces data to 2 dimensions). Also note high overlap_percentage.
mapper_full = km.KeplerMapper(verbose=2)

# Fit and transform data
projected_data_full = mapper_full.fit_transform(Xtrain,
                                      projection=sklearn.manifold.TSNE(perplexity=50))


# In[ ]:


# Create the graph (we cluster on the projected data and suffer projection loss)
graph_full = mapper_full.map(projected_data_full,
                   clusterer=sklearn.cluster.DBSCAN(eps=0.3, min_samples=15),
                   cover=km.Cover(35, 0.4))


# In[ ]:


Y=df['label']


# In[ ]:


# Matplotlib examples
km.draw_matplotlib(graph_full)
plt.show()


# In[ ]:


# Tooltips with the target y-labels for every cluster member
mapper_full.visualize(graph_full,
                 title="Skin Cancer Mapper with  Labels ",
                 path_html="/kaggle/working/skin_cancer_ylabel_images.html",
                 custom_tooltips=Y)


# In[ ]:





# In[ ]:




