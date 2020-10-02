#!/usr/bin/env python
# coding: utf-8

# # Summrary
# We try to implement a version of $k$NN from scratch, and explore what it means for computing the Area Under the Receiver Operating Characteristic Curve (ROC), or AUC score. The general format of the training follows from Chris's template.
# 
# ### Reference:
# * [Logistic Regression - [0.800]](https://www.kaggle.com/cdeotte/logistic-regression-0-800)

# In[126]:


import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set()

from tqdm import tqdm_notebook

import warnings
warnings.filterwarnings('ignore')


# In[2]:


get_ipython().run_cell_magic('time', '', "train = pd.read_csv('../input/train.csv')\ntest = pd.read_csv('../input/test.csv')\n\ncols = [c for c in train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]")


# # $k$-NN implementation
# 
# ### Simple $k$-NN
# Given a $k$ and a new test sample $\mathbf{x}^{(0)}$, the $k$NN classifier first identifies the neighbors $k$ points in the training data that are closest to $\mathbf{x}^{(0)}$, whose indices are represented by $\mathcal{N}_0$. Notice that the distance choice $\text{dist}(\mathbf{x}^{(0)}, \mathbf{x})$ here is not unique, usually we use $L^2$-distance (Euclidean).
# 
# $k$NN then estimates the conditional probability for class $j$ by computing the fraction of points in $\mathcal{N}_0$ whose target label actually equal $j$:
# 
# $$
# P\big(y= j| \mathbf{x}^{(0)} \big)\approx  \frac{1}{k} \sum_{i\in \mathcal{N}_0} 1\{ y^{(i)} = j\}.
# $$
# 
# The indicator function $1\{ y^{(i)} = j\}$ can be viewed as one vote from $i$-th sample. Finally, $k$NN applies Bayesian rule and classifies the test sample $\mathbf{x}^{(0)}$ to the class with the largest estimated probability (most votes).
# 
# ### Weighted voting
# 
# Inverse distance-weighted voting: closer neighbors get higher "votes". The class of each of the $k$ neighbors is multiplied by a weight proportional to the inverse of the distance from that point to the given test point. This ensures that nearer neighbors contribute more to the final vote than the more distant ones. For the new sample $\mathbf{x}^{(0)}$, then the vote function $V(\mathbf{x}^{(i)})$ for $i=1,\dots, k$ for these $k$ neighbors are defined as
# $$
# V(\mathbf{x}^{(i)}) = \begin{cases}
# \infty & \text{ if } \text{dist}(\mathbf{x}^{(0)}, \mathbf{x}^{(i)}) = 0,
# \\[1em]
# \displaystyle\frac{1}{\text{dist}(\mathbf{x}^{(0)}, \mathbf{x}^{(i)})} & \text{ otherwise }.
# \end{cases}
# $$
# Then we sum the votes for each class among these $k$ neighbors and classify the newcoming sample $\mathbf{x}^{(0)}$ into the class with the highest vote.
# 

# In[86]:


def get_knn_proba(X_train, y_train, X_test, k=5, tol=1e-8):
    '''
    Get the distance weighted voting using k nearest neighbors
    '''
    
    if type(y_train) is pd.Series:
        y_train = y_train.values
    
    num_classes = len(np.unique(y_train))
    
    # vectorized L^2 distance matrix
    dist = -2 * np.dot(X_test, X_train.T)             + np.sum(X_train**2, axis=1)             + np.sum(X_test**2, axis=1).reshape(-1,1)
    
    # if distance is too small, rescale it
    dist[dist <= tol] = tol
    
    # sort by columns for each row, then return the first k columns' indices
    index_knn = np.argsort(dist,axis = 1)[:,:k] 
    
    # the above are the indices, this is computing the inverse distances
    dist_inv_knn = 1/np.sort(dist,axis = 1)[:,:k] 
    
    # retrieving the labels of these k neighbors
    label_knn = y_train[index_knn]
    
    # computing the vote
    vote = np.zeros((X_test.shape[0], num_classes))
    for j in range(num_classes):
        vote[:,j] = np.sum(dist_inv_knn*(label_knn==j), axis=1)
        
    # normalize the vote to become a probability
    proba = vote/np.sum(vote,axis=1)[:,np.newaxis]
    
    return proba


# In[176]:


oof = np.zeros(len(train))
preds = np.zeros(len(test))
num_neighbors = 9

for i in tqdm_notebook(range(512)):

    train2 = train[train['wheezy-copper-turtle-magic']==i]
    test2 = test[test['wheezy-copper-turtle-magic']==i]
    idx1 = train2.index
    idx2 = test2.index
    train2.reset_index(drop=True,inplace=True)

    data = pd.concat([pd.DataFrame(train2[cols]), pd.DataFrame(test2[cols])])
    data2 = VarianceThreshold(threshold=2).fit_transform(data[cols])

    train3 = data2[:train2.shape[0]]; test3 = data2[train2.shape[0]:]

    skf = StratifiedKFold(n_splits=17, random_state=i)
    for train_index, cv_index in skf.split(train2, train2['target']):
        
        # the kNN probabilities
        oof[idx1[cv_index]] = get_knn_proba(train3[train_index,:], 
                            train2.loc[train_index]['target'],
                            train3[cv_index,:], k=num_neighbors)[:,1]
        preds[idx2] += get_knn_proba(train3[train_index,:], 
                            train2.loc[train_index]['target'],
                            test3, k=num_neighbors)[:,1]/ skf.n_splits 


# # Compute the ROC curve
# 
# The ROC curve is created by plotting the true positive rate (TPR) against the false positive rate (FPR) at various threshold settings. Here in this competition, say the label is $1$ for positive, $0$ is negative, then  the FPR can be computed by
# $$
# \text{FPR} = \frac{\# \text{ of negative samples classified as positive}}{\text{Total }\# \text{ of negative samples}},
# $$
# and TPR similarly is
# $$
# \text{TPR} = \frac{\# \text{ of positive samples classified as positive}}{\text{Total }\# \text{ of positive samples}}.
# $$
# Hence we run a threshold through a linear space approximately in $(0,1)$ to get these two rates.

# In[177]:


num_threshold = 1000
threshold = np.linspace(0,1,num=num_threshold)
num_pos = (train['target']==1).sum()
num_neg = (train['target']==0).sum()

# # a non-vectorized implementation of the code below
# FPR = np.zeros(num_threshold)
# TPR = np.zeros(num_threshold)
# for i, p in tqdm_notebook(enumerate(threshold)):
#     TPR[i] = ((train['target']==1)*(oof>=p)).sum()/num_pos
#     FPR[i] = ((train['target']==0)*(oof>=p)).sum()/num_neg

TPR = ((train['target']==1).values[np.newaxis,:]*(oof>=threshold[:,np.newaxis])       ).sum(axis=1)/num_pos
FPR = ((train['target']==0).values[np.newaxis,:]*(oof>=threshold[:,np.newaxis])       ).sum(axis=1)/num_neg


# ## Visualize the AUC
# We can simply use the midpoint rule to approximate this area.

# In[182]:


_, ax = plt.subplots(figsize=(10,6))
ax.plot(FPR, TPR, linewidth=3, color='k')
ax.fill_between(FPR, TPR, color="red", alpha=0.2)
ax.annotate('Area Under the Curve (ROC)', xy=(0.1, 0.4), fontsize=30)
ax.set_xlabel('False positive rate')
ax.set_ylabel('True positive rate');


# In[187]:


# midpoint rule
print('Approximated AUC: {:.6}'.format(np.abs((np.diff(FPR)*(TPR[:-1] + TPR[1:])/2).sum())) )


# In[186]:


# sklearn's version is more accurate
auc = roc_auc_score(train['target'], oof)
print(f'AUC: {auc:.6}')


# In[185]:


sub = pd.read_csv('../input/sample_submission.csv')
sub['target'] = preds
sub.to_csv('submission.csv', index=False)

