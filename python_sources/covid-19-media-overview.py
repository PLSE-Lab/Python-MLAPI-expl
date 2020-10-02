#!/usr/bin/env python
# coding: utf-8

# # Getting started with the Covid-19 Public Media Dataset

# This notebook shows how the [Covid-19 Public Media Dataset by Anacode](https://www.kaggle.com/jannalipenkova/covid19-public-media-dataset) can be used for data and text analysis. In the first section, we perform some aggregations on the metadata to get an initial overview. In the second section, we apply linguistic preprocessing and dive deeper into the text data. Specifically, we apply clustering to identify major themes in the data which can then be used for further analysis. 
# 
# For those who are impatient, please check [this interactive chart](https://anacode.de/wordpress/wp-content/uploads/2020/04/covid19media_clusters5.html) for the final clustering of the data. 

# ## 1. Initial overview
# 
# Let's first load the data and aggregate on some of the metadata - namely the domain, topic_area and date columns. For simplicity, we are relying on pandas built-in plotting functions.

# In[ ]:


import os
import pandas as pd


# In[ ]:


# load data file; please change the path to the data file if needed
df = pd.read_csv("/kaggle/input/covid19-public-media-dataset/covid19_articles.csv")
print(len(df))
df.head()


# In[ ]:


# aggregate by domains
domain_stats = df.domain.value_counts(ascending=True)
pd.DataFrame(domain_stats).plot.barh(figsize=(8, 8), legend=False)


# All data sources are in English, and the most prominent data sources such as express.co.uk, cnbc and theguardian are focussed on UK and US author- and readership. Therefore, we expect a corresponding bias towards these countries in the analysis.

# In[ ]:


# aggregate by topic areas
topic_area_stats = df.topic_area.value_counts(ascending=True)
pd.DataFrame(topic_area_stats).plot.barh(figsize=(8, 3), legend=False)


# In[ ]:


# aggregate dates
date_stats = df.date.value_counts()
date_stats.sort_index(inplace=True)
date_stats.plot()


# In this aggregation, note the typical weekly slumps in this chart that normally occur on weekends.

# ## 2. Preprocessing the text data
# 
# We mainly rely on spacy for preprocessing and apply the following steps:
# 
# - Tokenization
# - Lemmatization
# - POS-based filtering function words, leaving only content words
# - Stopword filtering based on a custom list
# 
# Be patient - preprocessing will take a while.

# In[ ]:


import spacy
NLP = spacy.load("en_core_web_sm")


# In[ ]:


RELEVANT_POS_TAGS = ["PROPN", "VERB", "NOUN", "ADJ"]
CUSTOM_STOPWORDS = ["say", "%", "will", "new", "would", "could", "other", 
                    "tell", "see", "make", "-", "go", "come", "can", "do", 
                    "such", "give", "should", "must", "use"]

def preprocess(text):
    doc = NLP(text)
    relevant_tokens = " ".join([token.lemma_.lower() for token in doc if token.pos_ in RELEVANT_POS_TAGS and token.lemma_.lower() not in CUSTOM_STOPWORDS])
    return relevant_tokens


# In[ ]:


from tqdm import tqdm
tqdm.pandas()
processed_content = df["content"].progress_apply(preprocess)
df["processed_content"] = processed_content


# ## 3. Clustering
# 
# In this section, we apply PCA, T-SNE and K-means clustering for clustering of the preprocessed texts. We close the section with an interactive plot in bokeh and some observations about the clusters. This section is inspired by the notebook published by Maksim Eren for Covid-19 Literature Clustering (https://www.kaggle.com/maksimeren/covid-19-literature-clustering).

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
from sklearn.manifold import TSNE
import numpy as np


# In[ ]:


def vectorize(text, maxx_features):    
    vectorizer = TfidfVectorizer(max_features=maxx_features)
    X = vectorizer.fit_transform(text)
    return X


# In[ ]:


texts = df.processed_content.tolist()
texts[0]


# In[ ]:


X = vectorize(texts, 2 ** 10)
X.shape


# In[ ]:


pca = PCA(n_components=0.95, random_state=42)
X_reduced= pca.fit_transform(X.toarray())
X_reduced.shape


# In[ ]:


distortions = []
K = range(8, 20)
for k in K:
    k_means = KMeans(n_clusters=k, random_state=42).fit(X_reduced)
    k_means.fit(X_reduced)
    distortions.append(sum(np.min(cdist(X_reduced, k_means.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])


# In[ ]:


import matplotlib.pyplot as plt
X_line = [K[0], K[-1]]
Y_line = [distortions[0], distortions[-1]]

# Plot the elbow
plt.plot(K, distortions, 'b-')
plt.plot(X_line, Y_line, 'r')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('Finding optimal k using the elbow method')
plt.show()


# We choose 11 as number of clusters and fit the data accordingly.

# In[ ]:


k = 11
kmeans = KMeans(n_clusters=k, random_state=42)
y_pred = kmeans.fit_predict(X_reduced)
df['y'] = y_pred


# To reduce the dimensionality of our data to 2-dimensional space, we apply t-SNE and plot the result.

# In[ ]:


tsne = TSNE(verbose=1, perplexity=100, random_state=42)
X_embedded = tsne.fit_transform(X.toarray())


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
import seaborn as sns

# sns settings
sns.set(rc={'figure.figsize':(10,10)})

# colors
palette = sns.color_palette("bright", 1)

# plot
sns.scatterplot(X_embedded[:,0], X_embedded[:,1], palette=palette)
plt.title('t-SNE without clusters')
plt.savefig("tsne_covid19media_unlabelled.png")
plt.show()


# Now, we use the clusters generated by k-means to color clustered areas in the t-SNE reduced dataset:

# In[ ]:


from matplotlib import pyplot as plt
import seaborn as sns

# sns settings
sns.set(rc={'figure.figsize':(10,10)})

# colors
palette = sns.hls_palette(k, l=.4, s=.9)

# plot
sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=y_pred, legend='full', palette=palette)
plt.title('t-SNE with {} clusters'.format(k))
plt.savefig("tsne_covid19media_labelled.png")
plt.show()


# To provide more transparency into the data, you can further use the bokeh interactive library to plot an interactive version of the chart which shows individual articles on mousehover, as described in [Maksim Eken's notebook on Covid-19 Literature Clustering](https://www.kaggle.com/maksimeren/covid-19-literature-clustering). Please check [this final plot](https://anacode.de/wordpress/wp-content/uploads/2020/04/covid19media_clusters5.html) where we also provided manual labels to the clusters.

# In[ ]:




