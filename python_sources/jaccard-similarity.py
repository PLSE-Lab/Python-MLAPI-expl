#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import json
from sklearn import preprocessing
from datasketch import MinHash, MinHashLSHForest


# # A nearest neighbor based approach
# 
# I've noticed a lot of approaches take a very NLP approach to this problem, however, this really doesn't seem to me like an NLP problem (I haven't looked into the data too much, so this may be incorrect). For each recipie, we are given a set of ingredients used and then we have to determine which cuisine the recipie belongs to. Given that we are given sets of ingredients, which are typically a word or two, there is no real structure here that NLP algorithms can take.
# 
# Given that we're working with sets, a natural choice of metric is the Jaccard similarity, combined with a simple k-nearest neighbors classifier. This, to me, seems much simpler than fancy GloVe or word2vec embeddings, combined with neural nets or SVMs.
# 
# In this notebook, I implement a simple weighted voting scheme with k-nearest neighbors using the Jaccard similarity index. I use an arbitrary LSH approximate nearest neighbor library because I did not want to wait for the exact nearest neighbors to be computed.
# 
# First, we have to build up the LSH index. We will use this to query the nearest neighbor for every test point.

# In[ ]:


with open('../input/train.json') as file:
    training_data = json.load(file)

lsh = MinHashLSHForest(num_perm=128)
train, y_train_str = {}, {}
for entry in training_data:
    # Compute min-hash and add to LSH
    min_hash = MinHash(num_perm=128)
    for e in entry['ingredients']:
        min_hash.update(e.encode('utf-8'))
    lsh.add(entry['id'], min_hash)
    
    # We may want to use the raw data later on
    train[entry['id']] = entry['ingredients']
    y_train_str[entry['id']] = entry['cuisine']

lsh.index()


# In order to make working with numpy simpler, we convert the target labels (cuisine) to integers.

# In[ ]:


label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(list(y_train_str.values()))
y_train = {eid: label_encoder.transform([y_train_str[eid]])[0] for eid in y_train_str}


# In[ ]:


def jaccard_similarity(x, y):
    return len(set(x) & set(y)) / len(set(x) | set(y))


# We will use an arbitrary number of neighbors for demonstrative purposes.

# In[ ]:


N_NEAREST_NEIGHBORS = 100


# To predict on the test set, the K nearest neighbors are found. Each neighbor then votes for its cuisine with the Jaccard similarity to the test point. The cusine with the most votes is assigned to the test point.

# In[ ]:


with open('../input/test.json') as file:
    test_data = json.load(file)
    
predictions = {}
for entry in test_data:
    # Create the min-hash for the given item and find its nearest neighbors
    min_hash = MinHash(num_perm=128)
    for e in entry['ingredients']:
        min_hash.update(e.encode('utf-8'))
        
    nearest_neighbors = lsh.query(min_hash, N_NEAREST_NEIGHBORS)
    similarities, cuisines, cuisines_str = [], [], []
    for nn in nearest_neighbors:
        similarities.append(jaccard_similarity(train[nn], entry['ingredients']))
        cuisines.append(y_train[nn])
        cuisines_str.append(y_train_str[nn])
    
    weighted_votes = np.bincount(cuisines, weights=similarities)
    prediction_idx = np.argmax(weighted_votes)
    prediction = label_encoder.classes_[prediction_idx]
    
    predictions[entry['id']] = prediction


# We will use pandas to write our predictions to a csv.

# In[ ]:


predictions_df = pd.DataFrame(list(predictions.items()), columns=['id', 'cuisine'])
predictions_df.to_csv('predictions.csv', index=False)


# This approach doesn't boast particularly high accuracy, but given its simplicity, it doesn't do too poorly either. I merely tried to demonstrate that there is no need to instantly jump to the most sophisticated methods like neural nets before exploring simpler options.
# 
# This approach could probably be improved upon as well to achieve better results e.g. the LSH could likely be tuned for higher recall, we could use a weighted min-hashes using tf-idf weights, so that rarer words would have a larger impact in nearest neighbor search.
# 
# As stated, the end goal of this notebook is not to achieve the best results, but to show that somewhat decent results can be achieved with minimal effort and using a very conceptually and programatically simple technique.
