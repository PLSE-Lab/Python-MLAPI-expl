#!/usr/bin/env python
# coding: utf-8

# SITQ: Learning to Hash for Maximum Inner Product Search
# =============================================
# 
# SITQ is a fast algorithm for approximate Maximum Inner Product Search (MIPS). It can find items which are likely to maximize inner product against a query in sublinear time.
# 
# About package
# ---------------------
# 
# `sitq` package provides two classes: `Sitq` and `Mips`. 
# 
# `Sitq` is the core of this package. It offers methods which convert a vector into a signature. 
# Signature is binary array with arbitrary number of bits. `Sitq` needs to train parameters with all items In order to calculate signatures for items and queries. 
# The important property of SITQ is: items whose signatures are similar to query signature are likely to maximize inner product among all items.
# 
# `Mips` offers methods for searching items which are likely to maximize inner product against an arbitrary query.
# It uses `Sitq` internally for fast approximate search.
# 
# More information is available at https://github.com/shiroyagicorp/sitq
# 
# About this notebook
# ----------------------------
# 
# This notebook compares performance of SITQ and brute-force for recommendation using `MovieLens 20M Dataset`.
# 

# Preparation
# ----------------

# In[ ]:


get_ipython().system('pip install -U sitq')


# In[ ]:


# Load ml-20m dataset

from sklearn.model_selection import train_test_split

def iter_data(path='../input/rating.csv'):
    with open(path, 'rt') as f:
        for line in f:
            (user_id, movie_id, _, _) = line.split(',')
            yield (user_id, movie_id)
        
uids_train, uids_test, iids_train, iids_test =     train_test_split(*zip(*iter_data()), test_size=0.2, random_state=0)


# In[ ]:


# Train recommender model

import implicit
import numpy as np
from scipy import sparse

uid_idxs = {uid: idx for idx, uid in enumerate(set(uids_train))}
iid_idxs = {iid: idx for idx, iid in enumerate(set(iids_train))}

def train_als():
    X = sparse.coo_matrix((np.ones(len(uids_train)), 
                           ([iid_idxs[iid] for iid in iids_train],
                            [uid_idxs[uid] for uid in uids_train])),
                          shape=(len(iid_idxs), len(uid_idxs))
                         ).tocsr()
    model = implicit.als.AlternatingLeastSquares(factors=16)
    model.fit(X)
    return model, X

als, X_train = train_als()


# In[ ]:


# Stats

print('Number of train data: {}'.format(len(uids_train)))
print('Number of test data: {}'.format(len(uids_test)))
print('')
print('Number of trained users: {}'.format(len(als.user_factors)))
print('Number of trained items: {}'.format(len(als.item_factors)))


# In[ ]:


# Define functions for evaluation

from collections import defaultdict
from functools import lru_cache, partial
from timeit import default_timer as timer

def _get_precision(recommended_items, true_items):
    return len(np.intersect1d(recommended_items, true_items)) / len(recommended_items)

def _get_recall(recommended_items, true_items):
    return len(np.intersect1d(recommended_items, true_items)) / len(true_items)

@lru_cache()
def _get_idcg(topn):
    return sum(1 / np.log2(i + 2) for i in range(topn))

def _get_ndcg(recommended_items, true_items):
    dcg = sum(1 / np.log2(i + 2)
              for i, item in enumerate(recommended_items) if item in true_items)
    return dcg / _get_idcg(len(true_items))

def _get_recommendations(recommender):
    recommendations_by_user = []
    for user_vector, train_items in zip(als.user_factors, X_train.T):
        train_item_idxs = train_items.nonzero()[1]
        recommendations_by_user.append(recommender(user_vector, excluded_idxs=train_item_idxs))
    return recommendations_by_user
    
def evaluate(recommender):
    start = timer()
    recommendations_by_user = _get_recommendations(recommender)
    end = timer()
    print('Time to recommend: {:.1f} sec'.format(end - start))
    
    iid_idxs_by_uid_idx = defaultdict(list)
    for (uid, iid) in zip(uids_test, iids_test):
        uid_idx = uid_idxs.get(uid)
        iid_idx = iid_idxs.get(iid)
        if uid_idx is None or iid_idx is None:
            continue
        iid_idxs_by_uid_idx[uid_idx].append(iid_idx)
        
    precision = recall = ndcg = 0
    for (uid_idx, _iid_idxs) in iid_idxs_by_uid_idx.items():
        recommended_idxs = recommendations_by_user[uid_idx]
        precision += _get_precision(recommended_idxs, _iid_idxs)
        recall += _get_recall(recommended_idxs, _iid_idxs)
        ndcg += _get_ndcg(recommended_idxs, _iid_idxs)
    print('precision: {:.3f}'.format(precision / len(iid_idxs_by_uid_idx)))
    print('recall: {:.3f}'.format(recall / len(iid_idxs_by_uid_idx)))
    print('ndcg: {:.3f}'.format(ndcg / len(iid_idxs_by_uid_idx)))


# Benchmark
# ----------------

# In[ ]:


# brute-force

def recommend_by_als(query_vector, excluded_idxs, limit):
    _limit = limit + len(excluded_idxs)
    scores = als.item_factors.dot(query_vector)
    item_idxs = np.argpartition(scores, -_limit)[-_limit:]
    item_idxs = item_idxs[np.argsort(scores[item_idxs])[::-1]]
    return np.setdiff1d(item_idxs, excluded_idxs, assume_unique=True)[:limit]

def evaluate_als():
    evaluate(partial(recommend_by_als, limit=10))


# In[ ]:


evaluate_als()


# In[ ]:


# SITQ

from sitq import Mips

def recommend_by_sitq(mips, query_vector, excluded_idxs, limit, distance):
    item_idxs, _ = mips.search(query_vector, 
                               limit=(limit + len(excluded_idxs)),
                               distance=distance,
                               require_items=True,
                               sort=True)
    return np.setdiff1d(item_idxs, excluded_idxs, assume_unique=True)[:limit]

def evaluate_sitq(signature_size, distance):
    start = timer()
    mips = Mips(signature_size=signature_size)
    mips.fit(als.item_factors)
    end = timer()
    print('Time to train SITQ: {:.3f} sec'.format(end - start))
    evaluate(partial(recommend_by_sitq, mips, limit=10, distance=distance))


# In[ ]:


evaluate_sitq(signature_size=8, distance=0)

