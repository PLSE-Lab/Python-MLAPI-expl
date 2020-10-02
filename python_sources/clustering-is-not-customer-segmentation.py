#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system(' pip install --upgrade pip')
get_ipython().system(' pip install hvplot')


# In third year at University we were introduced to a number of methods in Cluster Analysis: Kmeans, PAM, DBSCAN, Mixture Models and Hierarchical Clustering.  In Cluster Analysis, people in the data sciences try to find subsets or groups in the data which may inform analysis on how that data was generated.  At the time, Cluster Analysis was argued to impact many industries and research domains, from Finance (where clusetrings may identify shares with similar risk profiles or market states), to Linguistics (where researchers may use it to identify how languages have diverged over time) and Marketing (where analysts look to Segment Customers to target compaigns and products).  Typically, data used in Cluster Analysis has many dimensions, like age, gender, income and product spend.  This data can be neatly seperated in this high-dimensional space into grounds of well defined bell-shaped blobs, overlappings concentric circles or inter-locking crescents. This makes clustering hard and many clustering methods try to make trade-offs in their ablity to cluster data with strange shapes, varying densities or unkown numbers of clusters.  While clustering methods can get more and more complex in the way they make these trade-offs the problem for your Chief Marketing Officer is: is these are not actionable!
#   
# In Marketing, Customer Segmentation is not just a practice in grouping customers, it is a practise in identifying a group of uniqie and seperable customers around which you can think strategically about product promotion, development and pricing.  **Segmentation != Clustering **
#   
# So lets get back to clustering. Why is clustering so different? Well in traditional clustering, what we get are groups of clusters which are difficult to describe. You can describe them maybe by their means or their medoids, but in many cases this can be misleading.  In the half-moons dataset, there exist two interlocking semi-circles which represent two clusters in the data.  Using hierarchical cluster, or many methods, you can serperate these clusters but you cannot easily descibe in the original feature-space actionable rules which would be easy enough to have a student night, as the rule would have to look like over this age, over this income proportional to age, unless over this age then this income inverely proportional to age- which yes, is confusing.  Why bother clustering at all!
# 
# Now yes, there are amazing methods in Biclustering which cluster across overservations and features and they are fantastic for this problem but they can scale poorly to data and struggle with continuous data.  We need something fast, scallable, understandable which mirrors the business case: not clustering.  
# 
# What I have been looking at it 'clustering' trees.  I am not the first to think about this, and there are many papers across industries on clustering trees. In clustering trees what we are going to do is look at each feature individually and try and find a split point which maximizes some clustering score- say silhouette scores.  Not my implementation for this is really not as flexible or robust as I would want, but is enough for a prototype without tryign to get into some cython and piggy-back off sklearn. 
# 

# In[ ]:


from IPython.display import Image
Image('/kaggle/input/segmentation-examples/ABCDE_Facet.jpg')


# In[ ]:


Image('/kaggle/input/segmentation-examples/FGE_Facet.jpg')


# ![Scikit-Learn Clustering Becnhmarks](https://scikit-learn.org/stable/_images/sphx_glr_plot_cluster_comparison_001.png)  
# *Clustering benchmarks provided in the [Scikit-Learn Docuementation](https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html). *

# In[ ]:


import pandas as pd
import numpy as np 
from functools import partial
from toolz.curried import *
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.preprocessing import StandardScaler
from scipy.optimize import fminbound
from typing import Tuple
from itertools import chain
import hvplot.pandas


def yield_rule(X: np.ndarray, D: np.ndarray) -> Tuple[float, float]:
    not_isnan_mask = np.any(~np.isnan(X), axis=-1)
    criterion = partial(silhouette_score, X = D[not_isnan_mask, :][:, not_isnan_mask], metric = 'precomputed', njobs=-1)

    xopt_ = []
    fval_ = []

    for c in range(X.shape[1]):
        xopt, fval, _, _= fminbound(func = lambda x: -criterion(labels = X[not_isnan_mask,c] > x), 
                                    x1=np.nanquantile(X[not_isnan_mask,c], 0.1), x2=np.nanquantile(X[:,c], 0.9), 
                                    full_output=True)

        xopt_.append(xopt)
        fval_.append(-fval)

    split_feature = np.argmax(fval_)
    score = fval_[split_feature]
    split_point= xopt_[split_feature]
    
    return  split_feature, split_point

def yeild_split(X: np.ndarray, split_feature: int, split_point: float):
        
    left_ = np.where(X[:, [split_feature]] > split_point, X, np.full_like(X, np.nan, dtype=np.float))
    right_ = np.where(X[:, [split_feature]] <= split_point, X, np.full_like(X, np.nan, dtype=np.float))
    
    return left_, right_

def rule_split_chain(x, D):
    split_feature, split_point = yield_rule(x, D)
    return yeild_split(x, split_feature, split_point)


# So the data we are looking at is going to be a Kaggle dataset of mall customers. In it we have data on Gender, Age, Annual Income and Spending Score.  We want to find simple rules which describe the customer segments present in the data.  

# In[ ]:


data = pd.read_csv('/kaggle/input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')
data


# In[ ]:


depth = 3

X = StandardScaler().fit_transform(data.loc[:,['Age','Annual Income (k$)','Spending Score (1-100)']].to_numpy())
D = pairwise_distances(X, n_jobs=-1)
splitter = compose_left(lambda t: chain(*t), 
                        map(partial(rule_split_chain, D=D)))
leaves = pipe(reduce(lambda x, _: splitter(x), [[[X]] for _ in range(depth)]), lambda x: chain(*x), list)
labels = np.stack((np.any(~np.isnan(branch), axis=-1) for branch in leaves)).T.astype(float).argmax(-1)
silhouette_score(labels=labels, X = D, metric = 'precomputed', njobs=-1)


# For sure the model would require pruning, as I am kepping the tree symetrical throughout, and some hyperparameter tuning. But I would love to get poeples opinion on when Data Science != Business Problem.  

# In[ ]:


((data.loc[:,['Annual Income (k$)','Spending Score (1-100)']]
 .assign(label = labels.astype(str))
 .hvplot.scatter(x='Annual Income (k$)',y='Spending Score (1-100)', color='label'))

+ 

(data.loc[:,['Age','Spending Score (1-100)']]
 .assign(label = labels.astype(str))
 .hvplot.scatter(x='Age',y='Spending Score (1-100)', color='label'))

+ 

(data.loc[:,['Annual Income (k$)','Age']]
 .assign(label = labels.astype(str))
 .hvplot.scatter(x='Annual Income (k$)',y='Age', color='label'))

+

(data.loc[:,['Gender','Age']]
 .assign(label = labels.astype(str))
 .hvplot.scatter(x='Gender',y='Age', color='label'))).cols(2).opts(title='Tree segmentation')


# In[ ]:




