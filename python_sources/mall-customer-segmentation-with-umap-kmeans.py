#!/usr/bin/env python
# coding: utf-8

# # Visualization and Segmentation of Mall Customer Data
# 
# ### In this Kernel, we visualize UMAP embeddings for the Mall Customer Dataset, and then find and describe Clusters in the embedded datapoints.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import itertools as it

import sklearn
from sklearn import preprocessing
from umap import UMAP
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import plotly.express as px


# ### Read Dataset

# In[ ]:


mall = pd.read_csv('/kaggle/input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')
mall.sample(10)


# Since Gender is a binary categorical feature, we can transform it into a number for easier treatment, without losing information.

# In[ ]:


mall['Gender Numeric'] = mall['Gender'].map({'Male': -1, 'Female': 1})
mall


# ### Sklearn Pipeline: MinMaxScale our features

# In[ ]:


numeric_features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Gender Numeric']

numeric_transformer = Pipeline(steps=[('scaler', preprocessing.MinMaxScaler())])

preprocessor = ColumnTransformer(
    transformers=[('num', numeric_transformer, numeric_features)])


# UMAP is an outstanding dimensionality reduction technique, which we will use to have a "compact" 2-D representation of our 4-D dataset.

# In[ ]:


umap = UMAP(n_components=2, random_state=2020)
pipe = Pipeline(steps=[('preprocessor', preprocessor), ('uamp', umap)])
umap_out = pipe.fit_transform(mall)


# Let's see how many clusters we would need for our efforts. For that, we calculate the inertia (residuals) for the fit of several number of clusters.

# In[ ]:


sse = []
nclusters = list(range(1,11))
for k in nclusters:
    kmeans = KMeans(n_clusters=k)
    clusters = kmeans.fit_predict(umap_out)
    sse.append(kmeans.inertia_)
    
sb.pointplot(nclusters, sse).set_title('Inertia');


# According to this result, setting more than 4 clusters does not improve the fit much more. But, as we will see later, setting 6 will be a better choice.

# In[ ]:


kmeans = KMeans(n_clusters = 6, random_state = 2020)
clusters = kmeans.fit_predict(umap_out)


# Add clusters to a *results* dataset

# In[ ]:


df_umap = pd.DataFrame(data = umap_out, columns = ['Embedding 1', 'Embedding 2'])
df_clusters = pd.DataFrame(data = clusters, columns = ['Clusters']).apply(lambda x: 'C'+x.astype(str))

results = pd.concat([mall, df_umap, df_clusters], axis = 1)


# In[ ]:


results


# # Let's see what we did!

# In[ ]:


fig = px.scatter(results, x = 'Embedding 1', y='Embedding 2',
                    color= 'Clusters',
                    hover_data = ['Age', 'Gender', 'Annual Income (k$)', 'Spending Score (1-100)'],
                    width=600, height=600)
fig.show()


# We can see 3 clusters, one for each gender, featuring traits such as:
# - [C4 & C5] People in their 20s with high Spending Score.
# - [C1 & C3] High income & high Spending Score, Ages 30-40.
# - [C0 & C2] Mostly older people (>40) with medium and low Spending Scores. Some outliers show young people (<25) with very low Spending Score.
# 
# If we set 4 clusters, we would end up fusing pairs of these clusters, and much of these human interpretations!

# A final thought: I believe that adding Gender as a feature to our embeddings didn't add much, since there are clear parallels between the proposed clusters. It might have been better to just set 3 clusters, and dismiss that feature. Of course, in the real world this might be a bad idea as well! (take for example, I am targeting potential customers of gender-defined apparel?)

# In[ ]:




