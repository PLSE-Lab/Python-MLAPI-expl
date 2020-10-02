#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('../input/Iris.csv')
df['Species'].loc[101:102]


# In[ ]:


#See info about dataset
df.info()


# In[ ]:


# See how many classes are available totally in dataset
df['Species'].unique()


# In[ ]:


from sklearn.preprocessing import LabelEncoder

#Encode text classes to numeric classes
encoder = LabelEncoder().fit(df['Species'])


# In[ ]:


# Get hot encoded numeric classes from text classes
hot_encoded_labels = pd.DataFrame(encoder.transform(df['Species']))


# In[ ]:


# Assign required features to features dataframe
features = df.drop(['Species','Id'], axis=1)


# In[ ]:


features.head()


# In[ ]:


# Draw scatter_matrix to to see relevance of features
pd.plotting.scatter_matrix(features, figsize=(10,10))
x = 1


# In[ ]:


# Applying unsupervised learning techniques on features
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


# In[ ]:


# Create two models KMeans and GaussianMixture
clf = KMeans(n_clusters=3).fit(features)
gclf = GaussianMixture(n_components=3).fit(features)


# In[ ]:


# Predict features i.e.,predict inputs based on formed clusters
preds = clf.predict(features)
gpreds = gclf.predict(features)


# In[ ]:


def draw_clusters(features, preds):
    fig, axis = plt.subplots(2,2,figsize=(10,8))
    fea_matrix=  [('SepalLengthCm','SepalWidthCm'), ('PetalLengthCm', 'PetalWidthCm'),                 ('SepalLengthCm','PetalWidthCm'), ('PetalLengthCm','SepalWidthCm')]

    for i,row in enumerate(axis):
        for j,col in enumerate(row):
            x,y = fea_matrix[i+j]
            col.set_xlabel(x)
            col.set_ylabel(y)
            col.scatter(features[x],features[y], alpha=0.5,  c=preds,cmap='viridis',s=20 )
    plt.show()


# In[ ]:


#Clusters formed based on KMeans model
draw_clusters(features, preds)


# In[ ]:


#Clusters formed based on GaussianMixture Model
draw_clusters(features, gpreds)


# In[ ]:


#Clusters of actual output labels.
draw_clusters(features, list(hot_encoded_labels[0]))


# >  Hence both KMeans and GaussianMixture Model are performing best on this dataset, as we can see the graphs.

# In[ ]:




