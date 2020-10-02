#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import os
train=pd.read_csv('../input/utkml/train_final.csv')#.fillna(99)
test=pd.read_csv('../input/utkml/test_final.csv')#.fillna(99)
total=train.append(test,ignore_index=True)


# In[ ]:


total.sort_values('user_id')


# In[ ]:


datacol=total[['user_id','JOKE:5']]
datacol.columns=['user_id','rating']
datacol['item_id']=0

data=datacol.dropna()
for ci in range(2,141):
    colnm=train.columns[ci]
    datacol=total[['user_id',colnm]]
    datacol.columns=['user_id','rating']
    datacol['item_id']=ci-1
    data=data.append(datacol.dropna())
data

from scipy.sparse import coo_matrix

datacoo=coo_matrix((data.rating, ( data.item_id,data.user_id)) )

datacoo


# # test implicit ALS
# using Alternating Least Squares

# In[ ]:


import implicit

# initialize a model
model = implicit.als.AlternatingLeastSquares(factors=50)

# train the model on a sparse matrix of item/user/confidence weights
model.fit(datacoo)

# recommend items for a user
user_items = datacoo.T.tocsr()
recommendations = model.recommend(1, user_items)
print('recommended ',recommendations)
# find related items
related = model.similar_items(1)
print(related)


# In[ ]:


ratings=datacoo
import argparse
import codecs
import logging
import time

import numpy as np
import tqdm

from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from implicit.datasets.movielens import get_movielens
from implicit.lmf import LogisticMatrixFactorization
from implicit.nearest_neighbours import (BM25Recommender, CosineRecommender,
                                         TFIDFRecommender, bm25_weight)

log = logging.getLogger("implicit")


# # ALS on BM25
# probably the best solution lets try

# In[ ]:


# read in the input data file
start = time.time()
output_filename='output.txt'
model_name='bpr'
min_rating=-10.0,
titles=train.columns[1:]
# remove things < min_rating, and convert to implicit dataset
# by considering ratings as a binary preference only
ratings.data[ratings.data < min_rating] = 0
ratings.eliminate_zeros()
ratings.data = np.ones(len(ratings.data))

log.info("read data file in %s", time.time() - start)

# generate a recommender model based off the input params
if model_name == "als":
    model = AlternatingLeastSquares()

    # lets weight these models by bm25weight.
    print("weighting matrix by bm25_weight")
    ratings = (bm25_weight(ratings, B=0.9) * 5).tocsr()

elif model_name == "bpr":
    model = BayesianPersonalizedRanking()

elif model_name == "lmf":
    model = LogisticMatrixFactorization()

elif model_name == "tfidf":
    model = TFIDFRecommender()

elif model_name == "cosine":
    model = CosineRecommender()

elif model_name == "bm25":
    model = BM25Recommender(B=0.2)

else:
    raise NotImplementedError("TODO: model %s" % model_name)

# train the model
print("training model %s", model_name)
start = time.time()
model.fit(ratings)
print("trained model '%s' in %s", model_name, time.time() - start)
log.debug("calculating top movies")

user_count = np.ediff1d(ratings.indptr)
to_generate = sorted(np.arange(len(titles)), key=lambda x: -user_count[x])

print("calculating similar movies")
with tqdm.tqdm(total=len(to_generate)) as progress:
    with codecs.open(output_filename, "w", "utf8") as o:
        for movieid in to_generate:
            print(movieid,model.similar_items(movieid))
            # if this movie has no ratings, skip over (for instance 'Graffiti Bridge' has
            # no ratings > 4 meaning we've filtered out all data for it.
            if ratings.indptr[movieid] != ratings.indptr[movieid + 1]:
                title = titles[movieid]
                for other, score in model.similar_items(movieid):
                    #o.write("%s\t%s\t%s\n" % (title, titles[other], score))
                    try:
                        print(title,titles[other],score)
                        print()
                    except:
                        print(title,other,score)
            progress.update(1)



# # test best jokes for first 10 users

# In[ ]:


for xi in range(10):
    recommendations = model.recommend(xi, user_items)
    print('USER',xi,train.iloc[xi].sort_values(ascending=False)[:3])
    for ri,prob in recommendations:
        print('recommended ',ri,titles[ri],prob)


# In[ ]:


train.iloc[1]


# # fill in submission

# In[ ]:


test['predictions']='np.nan'
for xi in range(len(train),len(total)):

    testxi=xi-len(train)
    testuserid=test.iloc[testxi]['user_id']
    recommendations = model.recommend(testuserid, user_items)    
    test.iat[testxi,141]=titles[recommendations[0][0]]
    if xi/1000==int(xi/1000):
        print('USER',testxi,total.iloc[xi].sort_values(ascending=False)[:3])
        for ri,prob in recommendations:
            print('recommended ',ri,titles[ri],prob)
    


# In[ ]:


test[['user_id','predictions']].to_csv('submit.csv',index=False)

