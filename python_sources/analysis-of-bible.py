#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import spacy
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import math
import time
import plotly.express as px
import sys
from pandasql import sqldf
pysqldf = lambda q: sqldf(q, globals())
import os
RS = 123


# In[ ]:


data = pd.read_csv("../input/bible/t_asv.csv")


# ## To-do
# 1. Cluster the bible according to the book
# 2. Cluster the bible using K-Means (Doc2Vec of verses)
# 3. Sentiment of each bible chapter
# 4. Text completion using Transformers package

# In[ ]:


documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(data['t'])]


# In[ ]:


model = Doc2Vec(documents, vector_size=10, workers=4)


# In[ ]:


vector = [model.infer_vector([i]) for i in list(data['t'])]
vector = np.array(vector)


# ### Cluster the Bible according to the book

# In[ ]:


perplex = math.sqrt(vector.shape[0])
RS = 123
time_start = time.time()
tsne = TSNE(perplexity = perplex, learning_rate = 100, n_iter = 700, random_state=RS).fit_transform(vector)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))


# In[ ]:


columns = list(data.columns)
columns.extend(['comp1', 'comp2'])
data_filter = np.concatenate((data.to_numpy(),tsne), axis = 1)
data_filter = pd.DataFrame(data_filter, columns = columns)
data_filter.head(2)


# In[ ]:


fig = px.scatter(data_filter, x="comp1", y="comp2", hover_data=["t"], color="b")
fig.show()


# According to this, bible apparently has only one theme

# ### Cluster the bible using ML algorithms

# In[ ]:


sse = {}
for k in range(1, 21):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(vector)
    clusters = kmeans.labels_
    #print(data["clusters"])
    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()


# Cannot form any significant cluster because no elbow arises. Hence we go for GaussianMixture model

# In[ ]:





# In[ ]:


n_components = np.arange(1, 51)
models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(vector) for n in n_components]
plt.plot(n_components, [m.bic(vector) for m in models], label='BIC')
plt.plot(n_components, [m.aic(vector) for m in models], label='AIC')
plt.legend(loc='best')
plt.xlabel('n_components');


# We see a minima at cluster = 20 and hence we divide the data in 20 clusters

# In[ ]:


gmm = GaussianMixture(n_components=20)
gmm.fit(vector)
labels = gmm.predict(vector)


# In[ ]:


perplex = math.sqrt(vector.shape[0])
RS = 123
time_start = time.time()
tsne = TSNE(perplexity = perplex, learning_rate = 100, n_iter = 700, random_state=RS).fit_transform(vector)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))


# In[ ]:





# In[ ]:


columns = list(data.columns)
columns.extend(['comp1', 'comp2', 'labels'])
data_filter = np.concatenate((data.to_numpy(),tsne, labels.reshape(31103,1)), axis = 1)
data_filter = pd.DataFrame(data_filter, columns = columns)
data_filter.head(2)


# In[ ]:


fig = px.scatter(data_filter, x="comp1", y="comp2", hover_data=["t"], color="labels")
fig.show()


# There are small, overlapping clusters of themes in bible
