#!/usr/bin/env python
# coding: utf-8

# ## Reference ##
# 
# - [Document Clustering with Python](http://brandonrose.org/clustering)
# - ["Building Machine Learning Systems with Python"](https://www.packtpub.com/big-data-and-business-intelligence/building-machine-learning-systems-python) by Richert and Coelho
# - [sklearn User Guide: Text feature extraction](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
#     - [CountVectorizer API](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html#sklearn.feature_extraction.text.CountVectorizer)
#     - [TfidfVectorizer API](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html#sklearn.feature_extraction.text.TfidfVectorizer)

# In[ ]:


import numpy as np
import pandas as pd
import nltk
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

raw_data = pd.read_csv('../input/imdb-data/IMDB-Movie-Data.csv')
raw_data.set_index('Rank', inplace=True)
raw_data.head()


# ## Titles and Descriptions ##

# In[ ]:


data = raw_data[['Title', 'Description']]
print(data.shape)
data.head()


# Let's take a look at some of these descriptions.

# In[ ]:


descriptions = data['Description'].tolist()
descriptions[:3]


# ## Words and Stems ##
# 
# A tokenizer can break a text phrase into individual tokens. In our case, we'll consider the following:
# 
# - Ignore punctuation.
# - Ignore words containing numbers.
# - Break hyphenated words into two words.

# In[ ]:


token_pattern = '(?u)\\b[a-zA-Z][a-zA-Z]+\\b'
token_expression = re.compile(token_pattern)
description_tokenizer = lambda description: token_expression.findall(description)


# The stem of a word is similar to the "root" of a word. A stem lets us map similar words to the same stem.
# 
# The first thing we'll build is a map from each word stem to a corresponding word.

# In[ ]:


words = [
    word \
    for description in descriptions \
    for word in description_tokenizer(description)
]
stemmer = nltk.stem.snowball.SnowballStemmer('english')
stems = (stemmer.stem(word) for word in words)

word_for_stem =     pd.DataFrame({ 'stem': stems, 'word': words })     .set_index('stem')     .sort_index()
word_for_stem = word_for_stem.loc[~word_for_stem.index.duplicated(keep='first')]
word_for_stem.head()


# ## Preliminary Features ##
# 
# Now we'll turn each of the film descriptions into a feature vector.

# In[ ]:


vectorizer = TfidfVectorizer()
description_vectors = vectorizer.fit_transform(descriptions)
print(description_vectors.shape)

feature_names = vectorizer.get_feature_names()
feature_names[:6]


# Lets try again but only consider words that have all letters. Ignore words with numbers.

# In[ ]:


vectorizer = TfidfVectorizer(tokenizer=description_tokenizer)
description_vectors = vectorizer.fit_transform(descriptions)
print(description_vectors.shape)

feature_names = vectorizer.get_feature_names()
feature_names[:6]


# We'll define some methods for converting the words in an n-gram to/from stems.

# In[ ]:


stemmer = nltk.stem.snowball.SnowballStemmer('english')

def stems_for(ngram):
    stems = (stemmer.stem(word) for word in ngram.split())
    return ' '.join(stems)

def words_for(stemmed_phrase):
    words = (word_for_stem.loc[stem][0] for stem in stemmed_phrase.split())
    return ' '.join(words)


# Now we can map similar words to the same stem.

# In[ ]:


class CustomVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        standard_analyzer = super().build_analyzer()
        stemmer = nltk.stem.snowball.SnowballStemmer('english')
        return lambda description: (
            stems_for(ngram) for ngram in standard_analyzer(description)
        )

vectorizer = CustomVectorizer(tokenizer=description_tokenizer)
description_vectors = vectorizer.fit_transform(descriptions)
print(description_vectors.shape)

feature_names = vectorizer.get_feature_names()
[words_for(stems) for stems in feature_names[:6]]


# Ignore words that don't carry a lot of meaning.

# In[ ]:


vectorizer = CustomVectorizer(tokenizer=description_tokenizer, stop_words='english')
description_vectors = vectorizer.fit_transform(descriptions)
print(description_vectors.shape)

feature_names = vectorizer.get_feature_names()
[words_for(stems) for stems in feature_names[:6]]


# In addition to considering words individually, lets look at pairs and triplets of words.

# In[ ]:


vectorizer = CustomVectorizer(
    tokenizer=description_tokenizer, 
    stop_words='english',
    ngram_range=(1, 3)
)
description_vectors = vectorizer.fit_transform(descriptions)
print(description_vectors.shape)

feature_names = vectorizer.get_feature_names()
[words_for(stems) for stems in feature_names[:6]]


# ## Selected Features ##
# 
# If an n-gram shows up in almost none of the descriptions, lets ignore it.

# In[ ]:


vectorizer = CustomVectorizer(
    tokenizer=description_tokenizer, 
    stop_words='english',
    ngram_range=(1, 3),
    min_df=16
)
description_vectors = vectorizer.fit_transform(descriptions)
print(description_vectors.shape)

feature_names = vectorizer.get_feature_names()
[words_for(stems) for stems in feature_names[:6]]


# ## Clustering ##
# 
# Now we'll try to group the descriptions into five clusters.

# In[ ]:


clusterer = KMeans(n_clusters=5, random_state=1)
cluster_indices = pd.Series(
    clusterer.fit_predict(description_vectors), 
    name='Cluster Index'
)

clustered_data = data.join(cluster_indices)
clustered_data.head()


# Let's take a look at the center of the cluster that _"Guardians of the Galaxy"_ is a member of.

# In[ ]:


cluster_center = clusterer.cluster_centers_[4]
top_stem_indices = cluster_center.argsort()[::-1][:8]
top_stems = [feature_names[index] for index in top_stem_indices]
word_for_stem.loc[top_stems]


# What other movies are also part of this cluster?

# In[ ]:


clustered_data.loc[clustered_data['Cluster Index'] == 4]


# Some strange outliers there like _"La La Land"_ and _"Annie"_.
# 
# The cluster seems to be pretty big: 579 out of the 1000 films. Lets see how many films are in the other clusters.

# In[ ]:


cluster_indices.value_counts()

