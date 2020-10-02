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
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv("../input/answers.csv")


# In[ ]:


df.head(3)


# # K-Means Clustering

# ### EDA and data preprocessing
# Get only test 1 answers.

# Values according to question numbers are our feature set. It looks like categorical features but it is not categorical. Value column can be threated as continius. There fore k-means clustering can be applied.

# In[ ]:


#test1_soru_ids = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,18,19,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,182,200,202]
#df2 = df[(df['soru_id'].isin(test1_soru_ids))]
#df2.head()


# In[ ]:


df2 = df.pivot_table(index="user_id",columns="soru_id", values="value")
df2.shape


# In[ ]:


df2.head()


# In[ ]:


# Select rows whose count of non NaN valued columns are greater than 35 
df2 = df2[(df2.apply(lambda x: x.count(), axis=1)>35)]


# In[ ]:


# add a mean column
df2["mean"] = df2.mean(axis = 1, skipna = True)
df2.head()


# In[ ]:


# fill NaN fields with mean of that column
df3 = df2.fillna(df2.mean())


# ## Determining the k number with elbow method
# We are looking for an sharp elbow while ploting SSE vs. k
# 
# k: number of centeroids
# 
# SSE: Sum of squared errors. Sum of squared distances(eg. Euclidian Distance) from each sample to it's closest centeroid.
# 
# SSE tends to decrease as we increase k. SSE will decrease until 0 if we get k value with number of samples. Because then each data point has it's own cluster. Then the distance will be zero. The goal is select the smallest k value which makes largest drop on SSE.

# In[ ]:


sse = {}
for k in range(1, 15):
    km = KMeans(n_clusters=k, max_iter=1000).fit(df3)
    #print(data["clusters"])
    sse[k] = km.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()


# Sharpest elbow is on k=2 above image. It looks like there is 2 obvious clusters. Maybe these clusters are just conservative and liberal. But I will chose k=4 :)

# In[ ]:


kmeans = KMeans(n_clusters=4, random_state=1).fit(df3)
df3["clusters"] = kmeans.labels_
df3


# In[ ]:




