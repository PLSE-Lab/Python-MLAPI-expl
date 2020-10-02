#!/usr/bin/env python
# coding: utf-8

# ## Instacart dataset exploratory
# * Forked from : https://www.kaggle.com/goodvc/instacart-product2vec-clustering-using-word2vec 
#      * Fixed paths + minor changes
# * Instacart kaggle : https://www.kaggle.com/c/instacart-market-basket-analysis#prizes
# * blog post : https://tech.instacart.com/3-million-instacart-orders-open-sourced-d40d29ead6f2
# * data dictionary : https://gist.github.com/jeremystan/c3b39d947d9b88b3ccff3147dbcf6c6b
# * dataset file list 
# 
# 
# 

# In[ ]:


get_ipython().system(' pip install gensim')


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import gensim
from gensim.models import Word2Vec

import os
print(os.listdir("../input"))

pd.options.display.max_rows = 15
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style="whitegrid", palette="colorblind", font_scale=1, rc={'font.family':'NanumGothic'} )

def toReadable(v):
    value = round(v,2) if isinstance(v, float) else v

    if value < 1000:
        return str(value)
    elif value<1000000:
        return str(round(value/1000,1))+'K'
    elif value>=1000000:
        return str(round(value/1000000,1))+'M'
    return value


# ---
# ## Load Dataset

# In[ ]:


IDIR = '../input/'


# In[ ]:


raw_order_ds = pd.read_csv(IDIR + 'orders.csv', dtype={
        'order_id': np.int32,
        'user_id': np.int32,
        'eval_set': 'category',
        'order_number': np.int16,
        'order_dow': np.int8,
        'order_hour_of_day': np.int8,
        'days_since_prior_order': np.float32})
print("raw_order_ds shape",raw_order_ds.shape)

order_product_ds = priors = pd.read_csv(IDIR + 'order_products__prior.csv', dtype={
            'order_id': np.int32,
            'product_id': np.uint16,
            'add_to_cart_order': np.int16,
            'reordered': np.int8})
print("order_product_ds shape",order_product_ds.shape)

product_ds = pd.read_csv(IDIR + 'products.csv')
print("orders shape",product_ds.shape)

order_product_cnt_ds = order_product_ds.groupby('order_id').count()[['product_id']]
order_product_cnt_ds.columns = ['product_cnt']

## join product count 
order_ds = raw_order_ds.merge(order_product_cnt_ds, left_on='order_id', right_index=True)


# ### Dataset Summery
# Let's look at the simple stats of a dataset

# In[ ]:


total_user = len(order_ds.user_id.unique())
total_order = len(order_ds)
total_ordered_product = len(order_product_ds)
unique_products = len(order_product_ds.product_id.unique())

print("total user = {}".format(toReadable(total_user)))
print("total order = {} ({} orders per a user )".format(toReadable(total_order), toReadable(total_order/total_user) ))
print("total product = ", toReadable(unique_products))
print("total ordered product  = {} ({} orders per a product )".format(
    toReadable(total_ordered_product), toReadable(total_ordered_product/unique_products) ))


# ---
# ## Clustering similar product by user's order informaiton
# * Traing product2vec using word2vec 
#  * word = product_id
#  * scentence = user's order = [product_id1, product_id2, ... ]
# * clustering by trained product vector
# * Use only products ordered more than 100 times
# 
# * Do not Filter out orders with only 1 item in then in this case ? 

# In[ ]:


merge_order_product_ds = order_product_ds.merge(order_ds, on='order_id' )


# In[ ]:


order_product_list = merge_order_product_ds    .sort_values(['user_id','order_id','add_to_cart_order'])[['order_id','product_id']]    .values.tolist()

product_corpus = []
sentence = []
new_order_id = order_product_list[0][0]
for (order_id, product_id) in order_product_list:
    if new_order_id != order_id:
        product_corpus.append(sentence)
        sentence = []
        new_order_id = order_id
    sentence.append(str(product_id))


# In[ ]:


model = Word2Vec(product_corpus, window=9, size=100, workers=4, min_count=50)
# model.save('./resource/prod2vec.100d.model')
# model = Word2Vec.load('./resource/prod2vec.100d.model')


# In[ ]:


def toProductName(id):
    return product_ds[product_ds.product_id==id]['product_name'].values.tolist()[0]
toProductName(24852)


# In[ ]:


def most_similar_readable(model, product_id):
    similar_list = [(product_id,1.0)]+model.wv.most_similar(str(product_id))
    
    return [( toProductName(int(id)), similarity ) for (id,similarity) in similar_list]


# ### What is the most similar?
# * most similar to banana(24852) is ..

# In[ ]:


pd.DataFrame(most_similar_readable(model, 24852), columns=['product','similarity'])


# * most similar to Drinking Water(27845) is ..

# In[ ]:


pd.DataFrame(most_similar_readable(model, 27845), columns=['product','similarity'])


# * most similar to Organic Whole Milk(40939) is .. 

# In[ ]:


pd.DataFrame(most_similar_readable(model, 40939), columns=['product','similarity'])


# In[ ]:


pd.DataFrame(most_similar_readable(model, 48697), columns=['product','similarity'])


# ### Product2Vec works well !!! 

# * Create 500 clusters as similar products
# * using kmeans ( k=500 )

# In[ ]:


from __future__ import division
import random
import numpy as np
from scipy.spatial.distance import cdist  # $scipy/spatial/distance.py
    # http://docs.scipy.org/doc/scipy/reference/spatial.html
from scipy.sparse import issparse  # $scipy/sparse/csr.py

__date__ = "2018-09-01"
    # X sparse, any cdist metric: real app ?
    # centres get dense rapidly, metrics in high dim hit distance whiteout
    # vs unsupervised / semi-supervised svm
#...............................................................................
def kmeans( X, centres, delta=.001, maxiter=10, metric="euclidean", p=2, verbose=1 ):
    """ centres, Xtocentre, distances = kmeans( X, initial centres ... )
    in:
        X N x dim  may be sparse
        centres k x dim: initial centres, e.g. random.sample( X, k )
        delta: relative error, iterate until the average distance to centres
            is within delta of the previous average distance
        maxiter
        metric: any of the 20-odd in scipy.spatial.distance
            "chebyshev" = max, "cityblock" = L1, "minkowski" with p=
            or a function( Xvec, centrevec ), e.g. Lqmetric below
        p: for minkowski metric -- local mod cdist for 0 < p < 1 too
        verbose: 0 silent, 2 prints running distances
    out:
        centres, k x dim
        Xtocentre: each X -> its nearest centre, ints N -> k
        distances, N
    see also: kmeanssample below, class Kmeans below.
    """
    if not issparse(X):
        X = np.asanyarray(X)  # ?
    centres = centres.todense() if issparse(centres)         else centres.copy()
    N, dim = X.shape
    k, cdim = centres.shape
    if dim != cdim:
        raise ValueError( "kmeans: X %s and centres %s must have the same number of columns" % (
            X.shape, centres.shape ))
    if verbose:
        print ("kmeans: X %s  centres %s  delta=%.2g  maxiter=%d  metric=%s" % (
            X.shape, centres.shape, delta, maxiter, metric) )
    allx = np.arange(N)
    prevdist = 0
    for jiter in range( 1, maxiter+1 ):
        D = cdist_sparse( X, centres, metric=metric, p=p )  # |X| x |centres|
        xtoc = D.argmin(axis=1)  # X -> nearest centre
        distances = D[allx,xtoc]
        avdist = distances.mean()  # median ?
        if verbose >= 2:
            print("kmeans: av |X - nearest centre| = %.4g" % avdist)
        if (1 - delta) * prevdist <= avdist <= prevdist         or jiter == maxiter:
            break
        prevdist = avdist
        for jc in range(k):  # (1 pass in C)
            c = np.where( xtoc == jc )[0]
            if len(c) > 0:
                centres[jc] = X[c].mean( axis=0 )
    if verbose:
        print ("kmeans: %d iterations  cluster sizes:" % jiter, np.bincount(xtoc))
    if verbose >= 2:
        r50 = np.zeros(k)
        r90 = np.zeros(k)
        for j in range(k):
            dist = distances[ xtoc == j ]
            if len(dist) > 0:
                r50[j], r90[j] = np.percentile( dist, (50, 90) )
        print ("kmeans: cluster 50 % radius", r50.astype(int))
        print ("kmeans: cluster 90 % radius", r90.astype(int))
            # scale L1 / dim, L2 / sqrt(dim) ?
    return centres, xtoc, distances
#...............................................................................
def kmeanssample( X, k, nsample=0, **kwargs ):
    """ 2-pass kmeans, fast for large N:
        1) kmeans a random sample of nsample ~ sqrt(N) from X
        2) full kmeans, starting from those centres
    """
        # merge w kmeans ? mttiw
        # v large N: sample N^1/2, N^1/2 of that
        # seed like sklearn ?
    N, dim = X.shape
    if nsample == 0:
        nsample = max( 2*np.sqrt(N), 10*k )
    Xsample = randomsample( X, int(nsample) )
    pass1centres = randomsample( X, int(k) )
    samplecentres = kmeans( Xsample, pass1centres, **kwargs )[0]
    return kmeans( X, samplecentres, **kwargs )

def cdist_sparse( X, Y, **kwargs ):
    """ -> |X| x |Y| cdist array, any cdist metric
        X or Y may be sparse -- best csr
    """
        # todense row at a time, v slow if both v sparse
    sxy = 2*issparse(X) + issparse(Y)
    if sxy == 0:
        return cdist( X, Y, **kwargs )
    d = np.empty( (X.shape[0], Y.shape[0]), np.float64 )
    if sxy == 2:
        for j, x in enumerate(X):
            d[j] = cdist( x.todense(), Y, **kwargs ) [0]
    elif sxy == 1:
        for k, y in enumerate(Y):
            d[:,k] = cdist( X, y.todense(), **kwargs ) [0]
    else:
        for j, x in enumerate(X):
            for k, y in enumerate(Y):
                d[j,k] = cdist( x.todense(), y.todense(), **kwargs ) [0]
    return d

def randomsample( X, n ):
    """ random.sample of the rows of X
        X may be sparse -- best csr
    """
    random.seed(100)    
    sampleix = random.sample( range( X.shape[0] ), int(n) )
    return X[sampleix]

def nearestcentres( X, centres, metric="euclidean", p=2 ):
    """ each X -> nearest centre, any metric
            euclidean2 (~ withinss) is more sensitive to outliers,
            cityblock (manhattan, L1) less sensitive
    """
    D = cdist( X, centres, metric=metric, p=p )  # |X| x |centres|
    return D.argmin(axis=1)

def Lqmetric( x, y=None, q=.5 ):
    # yes a metric, may increase weight of near matches; see ...
    return (np.abs(x - y) ** q) .mean() if y is not None         else (np.abs(x) ** q) .mean()

#...............................................................................
class Kmeans:
    """ km = Kmeans( X, k= or centres=, ... )
        in: either initial centres= for kmeans
            or k= [nsample=] for kmeanssample
        out: km.centres, km.Xtocentre, km.distances
        iterator:
            for jcentre, J in km:
                clustercentre = centres[jcentre]
                J indexes e.g. X[J], classes[J]
    """
    def __init__( self, X, k=0, centres=None, nsample=0, **kwargs ):
        self.X = X
        if centres is None:
            self.centres, self.Xtocentre, self.distances = kmeanssample(
                X, k=k, nsample=nsample, **kwargs )
        else:
            self.centres, self.Xtocentre, self.distances = kmeans(
                X, centres, **kwargs )

    def __iter__(self):
        for jc in range(len(self.centres)):
            yield jc, (self.Xtocentre == jc)


# In[ ]:


def clustering(model, k=500, delta=0.00000001, maxiter=200):
    movie_vec = model.wv.syn0
    centres, index2cid, dist = kmeanssample(movie_vec, k, 
                                                   metric = 'cosine', 
                                                   delta = delta, 
                                                   nsample = 0, maxiter = maxiter,)
    clustered_ds = pd.DataFrame( [ (a, b, c) for a, b, c in zip(model.wv.index2word, index2cid, dist )],
                 columns=['product_id', 'cid', 'dist'] ).sort_values(['cid','dist'], ascending=True)

    prod2cid = { product_id:cid for product_id,cid in zip(model.wv.index2word, index2cid) }

    return (centres, index2cid, dist, clustered_ds, prod2cid)


# In[ ]:


(centres, index2cid, dist, clustered_ds, prod2cid) = clustering(model)


# In[ ]:


clustered_ds.product_id = clustered_ds.product_id.apply(pd.to_numeric)


# In[ ]:


def idToProductDesc(id):
    return product_ds[product_ds.product_id==id][['product_name','aisle_id']].values.tolist()[0]
    
def getProductNames(product_id_list):
    return [ idToProductDesc(int(product_id)) for  product_id in product_id_list ]

import urllib
def printClusterMembers(cluster_id, topn=10):
    members = getProductNames(clustered_ds[clustered_ds.cid==cluster_id].product_id[:topn].tolist())
    for member in members:
        print("{aisle} / {name} ".format( 
            aisle=member[1], name=member[0], q=urllib.parse.quote_plus(member[0]) ) 
        )


# ### Clustered Result
# ### Let's look at clustered product 

# * Cluster ID = 0 th

# In[ ]:


printClusterMembers(1, topn=10)


# * Cluster ID = 100 th

# In[ ]:


printClusterMembers(100, topn=10)


# * Cluster ID = 200 th

# In[ ]:


printClusterMembers(200, topn=10)


# * Cluster ID = 300 th

# In[ ]:


printClusterMembers(300, topn=10)


# * Cluster ID = 400 th

# In[ ]:


printClusterMembers(400, topn=10)


# * Cluster ID = 499 th

# In[ ]:


printClusterMembers(499, topn=10)


# ### It looks goooood!! 

# ----
# ### Order time trend of clustered product.

# * Extract representative keywords from each cluster.
# * Reprosentative keywords : 3 words and max 15 latters
# * Sort by popular order hour 

# In[ ]:


# product_reorder_ds.groupby('aisle_id').agg({'product_name':                                           lambda x: })
from collections import defaultdict
import operator

def popularWords(names, topn=2):
    wordFrequency = defaultdict(int)
    def updateWords(words):
        for word in words :
            if len(word)>1:
                wordFrequency[word] += 1
    names.apply(lambda x: updateWords(x.split()))
    tops = sorted(wordFrequency.items(), key=operator.itemgetter(1),reverse=True)[:topn]
    return " ".join([n[0] for n in tops])


# In[ ]:


clusterIdToKeywords = { cid: popularWords(sub_ds.product_name,3) for cid, sub_ds in clustered_ds.merge(product_ds, on='product_id').groupby('cid')}


# #### Hour of Day Trend Per cluster 

# In[ ]:


product_hod_ds = merge_order_product_ds.pivot_table(index='product_id', columns='order_hour_of_day', values='order_id', aggfunc=len, fill_value=0)

orderByHotHour = clustered_ds.merge(product_hod_ds, left_on='product_id', right_index=True)    .groupby('cid').sum()[np.arange(0,24)].idxmax(axis=1).sort_values().index


# In[ ]:


sns.set(style="whitegrid", palette="colorblind", font_scale=1, rc={'font.family':'NanumGothic'} )

def drawHODCluster(ncols, nrows, startClusterNumber, step):
    fig, axes = plt.subplots(ncols=ncols, nrows = nrows, figsize=(ncols*2.5,nrows*2), sharex=True, sharey=True)

    for cid, ax  in enumerate(axes.flatten()):
        cid = startClusterNumber + (cid*step)
        if cid>=500:
            break
        cid = orderByHotHour[cid]

        product_id_list = clustered_ds[clustered_ds.cid==cid].product_id.values
        tmp_ds = product_hod_ds.loc[product_id_list].T
        hot_hour = tmp_ds.sum(axis=1).argmax()
        normalized_ds =(tmp_ds/tmp_ds.max())
        title = "{cid}th {n} products \n({keyword})".format(cid=cid, n=normalized_ds.shape[1],  keyword=clusterIdToKeywords[cid][:23])
        normalized_ds.plot(linewidth=.3, legend=False, alpha=.4, ax=ax, title=title, color='r' if hot_hour<13 else 'k')
        ax.plot((hot_hour,hot_hour),(1,0), '-.', linewidth=1, color='b')
        ax.text(hot_hour,0,"{h}h(hot)".format(h=hot_hour),color='b')

    fig.tight_layout()


# In[ ]:


ncols, nrows=(6,4)
step = 3
for n in np.arange(0,500,ncols*nrows*step):
    drawHODCluster(ncols, nrows, n, step)


# #### Hour of Day Trend Per cluster 

# In[ ]:


product_dow_ds = merge_order_product_ds.pivot_table(index='product_id', columns='order_dow', values='order_id', aggfunc=len, fill_value=0)

orderByHotDay = clustered_ds.merge(product_dow_ds, left_on='product_id', right_index=True)    .groupby('cid').sum()[np.arange(0,6)].idxmax(axis=1).sort_values().index


# In[ ]:


def drawDOWCluster(ncols, nrows, startClusterNumber, step):
    sns.set(style="whitegrid", palette="colorblind", font_scale=1, rc={'font.family':'NanumGothic'} )
    week_day = "Sun Mon Tue Wed Thu Fri Sat".split()
    fig, axes = plt.subplots(ncols=ncols, nrows = nrows, figsize=(ncols*2.5,nrows*2), sharex=True, sharey=True)

    for cid, ax  in enumerate(axes.flatten()):
        cid = startClusterNumber + (cid*step)
        if cid>=500:
            break
        cid = orderByHotDay[cid]    
        product_id_list = clustered_ds[clustered_ds.cid==cid].product_id.values
        tmp_ds = product_dow_ds.loc[product_id_list].T
        hot_day = tmp_ds.sum(axis=1).argmax()
        normalized_ds =(tmp_ds/tmp_ds.max())
        normalized_ds.index = week_day
        title = "{cid}th \n({keyword})".format(cid=cid, h=hot_day,  keyword=clusterIdToKeywords[cid][:23])
        normalized_ds.plot(kind='bar', linewidth=.1, legend=False, alpha=.4, ax=ax, title=title, color='r' if hot_day in(0,6) else 'k')
        ax.plot((hot_day,hot_day),(1,0), '-.', linewidth=2, color='b')
        # ax.text(hot_day+.3,-.5,"{h}".format(h=week_day[hot_day]),color='b')
    
    fig.tight_layout()


# In[ ]:


ncols, nrows=(6,4)
step = 3
for n in np.arange(0,500,ncols*nrows*step):
    drawDOWCluster(ncols, nrows, n, step)


# In[ ]:





# In[ ]:




