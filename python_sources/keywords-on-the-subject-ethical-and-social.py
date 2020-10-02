#!/usr/bin/env python
# coding: utf-8

# # What has been published about ethical and social science considerations?
# ## COVID-19 Open Research Dataset Challenge (CORD-19)
# 
# Let's try to find what people are talking in the COVID-19 papers about ethical and social science.
# In this aproach we are going to try a unsupervisioned method to find articles that match our need.
# This can be useful to save time when we have a lot of data.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import os


# #### First, load the data

# In[ ]:


metadata = pd.read_csv('/kaggle/input/CORD-19-research-challenge/2020-03-13/all_sources_metadata_2020-03-13.csv')


# In[ ]:


metadata.head()


# In[ ]:


metadata.info()


# #### Let's see if we can find anything using the papers abstract

# In[ ]:


metadata_filter = metadata[metadata.abstract.str.contains('ethics|ethical|social science|multidisciplinary research', 
                                                          regex= True, na=False)].reset_index(drop=True)


# In[ ]:


len(metadata_filter)


# #### We found 155 articles, lets take a look into one.

# In[ ]:


metadata_filter.abstract[1]


# #### We can read all 155 of then, or we can use unsupervisioned learning to find patterns inside the text.
# #### First let's prepare the data, removing anything that is not a word, and removing common words (stopwords)

# In[ ]:


REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    return text

metadata_filter['clean_abstract'] = metadata_filter['abstract'].apply(clean_text)


# #### Now lets transform the words into a word matrix

# In[ ]:


vectorizer = TfidfVectorizer(stop_words='english', max_features=2000)
X = vectorizer.fit_transform(metadata_filter['clean_abstract'])


# #### Let's see if we have too many different subjects before getting the keywords. We use TSNE to reduce the word matrix to 2 dimensions, so we can plot the results.

# In[ ]:


tsne = TSNE(perplexity=4, random_state=42)

X_tsne = tsne.fit_transform(X)
X_tsne = pd.DataFrame(data=X_tsne, columns=['D1', 'D2'])


# In[ ]:


fig, ax = plt.subplots(figsize=(12,8))
sns.scatterplot(ax=ax,x = 'D1', y = 'D2', data=X_tsne, alpha=0.7)
plt.show()


# #### We can see that we may have a central cluster and other sparse elements. Let's see if this is a real cluster

# In[ ]:


kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X)


# In[ ]:


X_tsne['CLUSTER'] = kmeans.predict(X)

fig, ax = plt.subplots(figsize=(12,8))
sns.scatterplot(ax=ax,x = 'D1', y = 'D2', hue = 'CLUSTER', data=X_tsne, alpha=0.7)
plt.show()


# #### We have a little separation (not the best one... maybe if we had more data it could be better) but let's work with it. Let's add the cluster information into the metadata DF

# In[ ]:


metadata_filter['cluster'] = kmeans.predict(X)
metadata_filter.groupby('cluster')['cluster'].count()


# #### Now let's try to find the subject to each cluster. We count how many time a word appear and run a LDA to group important words

# In[ ]:


X1 = metadata_filter.loc[metadata_filter['cluster'] == 0, 'clean_abstract']
X2 = metadata_filter.loc[metadata_filter['cluster'] == 1, 'clean_abstract']

tf_vectorizer1 = CountVectorizer(max_features=2000, stop_words='english')
tf_vectorizer2 = CountVectorizer(max_features=2000, stop_words='english')

X1 = tf_vectorizer1.fit_transform(X1)
X2 = tf_vectorizer2.fit_transform(X2)

lda1 = LatentDirichletAllocation(n_components=5, random_state=42)
lda2 = LatentDirichletAllocation(n_components=5, random_state=42)

lda1.fit(X1)
lda2.fit(X2)


# ### Now let's print the main topics and their keywords

# In[ ]:


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)


# In[ ]:


print_top_words(lda1, tf_vectorizer1.get_feature_names(), 5)


# In[ ]:


print_top_words(lda2, tf_vectorizer2.get_feature_names(), 5)


# #### We can see that the first cluster (the bigger one) has more subjects that we want, while the second has more clinical data.
# #### We went from 29500 articles to 116, so this method could be useful to optimize the search for important articles. We could use the full text, or add more words to the regex search, but for now is a good first try.
