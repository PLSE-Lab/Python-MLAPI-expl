#!/usr/bin/env python
# coding: utf-8

# # Starter Code for fastText English Word Vectors Embedding
# 
# This kernel intends to be a starter code for anyone using the fastText Embedding. It uses Gensim to create a `KeyedVector` object (behavior similar to a dictionary). An example of tokenizing the data is also given.

# In[ ]:


import os

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import nltk
from gensim.models import KeyedVectors
from sklearn.datasets import fetch_20newsgroups


# In[ ]:


print(os.listdir('../input'))


# In[ ]:


FILE_PATH = '../input/fasttext-wikinews/wiki-news-300d-1M.vec'


# In[ ]:


# Let's read the first few lines 
with open(FILE_PATH) as f:
    for _ in range(5):
        print(f.readline()[:80])


# ## Load the embedding

# In[ ]:


# This may take a few mins
keyed_vec = KeyedVectors.load_word2vec_format(FILE_PATH)


# In[ ]:


for word in ['hello', '!', '2', 'Turing', 'foobarz', 'hi!']:
    print(word, "is in the vocabulary:", word in keyed_vec.vocab)


# ### Retrieving a vector with the KeyedVector

# In[ ]:


word_vec = keyed_vec.get_vector('foobar')
print(word_vec.shape)
print(word_vec[:25])


# ### Creating Keras Embeddings

# In[ ]:


keras_embedding = keyed_vec.get_keras_embedding()
keras_embedding.get_config()


# ## Applied Example: Prediction with scikit-learn

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


# In[ ]:


def mean_fasttext(arr, embedding_dim=300):
    '''
    Create the average of the fasttext embeddings from each word in a document. 
    Very slow function, needs to be optimized for larger datasets
    '''
    mean_vectors = []
    for document in arr:
        tokens = nltk.tokenize.word_tokenize(document)
        vectors = [keyed_vec.get_vector(token) for token in tokens if token in keyed_vec.vocab]
        if vectors:
            mean_vec = np.vstack(vectors).mean(axis=0)
            mean_vectors.append(mean_vec)
        else:
            mean_vectors.append(np.zeros(embedding_dim))
    embedding = np.vstack(mean_vectors)
    return embedding


# In[ ]:


data_sample = pd.read_csv('../input/quora-insincere-questions-classification/train.csv', nrows=6000)
train_sample = data_sample[:5000]
test_sample = data_sample[5000:]
train_sample.head()


# In[ ]:


X_train = mean_fasttext(train_sample["question_text"].values)
X_test = mean_fasttext(test_sample["question_text"].values)
y_train = train_sample['target'].values
y_test = test_sample['target'].values
print(X_train.shape)
print(y_train.shape)


# In[ ]:


model = LogisticRegression(solver='lbfgs')
model.fit(X_train, y_train)
print("Train Score:", f1_score(y_train, model.predict(X_train)))
print("Test Score:", f1_score(y_test, model.predict(X_test)))

