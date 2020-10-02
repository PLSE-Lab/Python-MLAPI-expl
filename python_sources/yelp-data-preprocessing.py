#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import time
from scipy.sparse import save_npz
from nltk.tokenize import word_tokenize, ToktokTokenizer

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.model_selection import train_test_split


# In[ ]:


print(os.listdir("../input"))


# In[ ]:


train_df = pd.read_csv('../input/yelp-reviews-2015/yelp_review_full_csv/yelp_review_full_csv/train.csv', header=None, names=['Score', 'Text'])
test_df = pd.read_csv('../input/yelp-reviews-2015/yelp_review_full_csv/yelp_review_full_csv/test.csv', header=None, names=['Score', 'Text'])


# In[ ]:


print(train_df.shape)
print(test_df.shape)
train_df.head()


# ## Save target

# In[ ]:


y_train = train_df['Score'].values
y_test = test_df['Score'].values
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)


# ## BoW

# In[ ]:


bow_vec = CountVectorizer(max_features=50000)
bow_vec.fit(train_df['Text'].values)


# In[ ]:


train_bow = bow_vec.transform(train_df['Text'].values)
test_bow = bow_vec.transform(test_df['Text'].values)
save_npz('X_train.npz', train_bow)
save_npz('X_test.npz', test_bow)
save_npz('train_bow.npz', train_bow)
save_npz('test_bow.npz', test_bow)


# ## Bigrams

# In[ ]:


bigrams_vec = CountVectorizer(max_features=50000, ngram_range=(1,2))
bigrams_vec.fit(train_df['Text'].values)


# In[ ]:


train_bigrams = bigrams_vec.transform(train_df['Text'].values)
test_bigrams = bigrams_vec.transform(test_df['Text'].values)
save_npz('train_bigrams.npz', train_bigrams)
save_npz('test_bigrams.npz', test_bigrams)


# ### Trigrams

# ## 3-grams with HashingVectorizer

# In[ ]:


hash_vec = HashingVectorizer(n_features=100000, ngram_range=(1,3))
train_hash = hash_vec.fit_transform(train_df['Text'].values)
test_hash = hash_vec.fit_transform(test_df['Text'].values)


# In[ ]:


save_npz('train_hash.npz', train_hash)
save_npz('test_hash.npz', test_hash)


# ## FastText

# In[ ]:


from gensim.models import KeyedVectors
fasttext_path = '../input/fasttext-wikinews/wiki-news-300d-1M.vec'
keyed_vec = KeyedVectors.load_word2vec_format(fasttext_path)


# In[ ]:


def verify(df):
    mean_vectors = []

    for document in df["Text"]:
        tokens = word_tokenize(document)

#     embedding = np.vstack(mean_vectors)
#     return embedding


# In[ ]:


def fasttext_embedding(arr):
    mean_vectors = []

    for document in arr:
        tokens = word_tokenize(document)
        vectors = [keyed_vec.get_vector(token) for token in tokens if token in keyed_vec.vocab]

        if vectors:
            mean_vec = np.vstack(vectors).mean(axis=0)
            mean_vectors.append(mean_vec)
        else:
            mean_vectors.append(np.zeros(300))

    embedding = np.vstack(mean_vectors)
    return embedding


# In[ ]:


start = time.time()
train_fasttext = fasttext_embedding(train_df["Text"].values)
test_fasttext = fasttext_embedding(test_df["Text"].values)
print(time.time()-start)
np.save('train_fasttext.npy', train_fasttext)
np.save('test_fasttext.npy', test_fasttext)

