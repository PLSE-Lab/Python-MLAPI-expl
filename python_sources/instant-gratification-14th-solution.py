#!/usr/bin/env python
# coding: utf-8

# First of all, congrats to all the winners! I'll share what I've did till today.

# # 1. Magic Split
# 
# Like discovered in the early stage of this competition, splitting the data with `wheezy-copper-turtle-magic` and modelling each subsets gave our score a great boost. (https://www.kaggle.com/cdeotte/logistic-regression-0-800)

# # 2. Dataset Structure
# Chris found the structure of the dataset. It was generated with 512 different distributions, and the features that have std smaller than 2 were useless. (https://www.kaggle.com/c/instant-gratification/discussion/92930)

# # 3. make\_classification
# 
# mhviraf discovered that the each of the 512 data subsets has many similarities with the data generated with sklearn's make\_classification function. (https://www.kaggle.com/mhviraf/synthetic-data-for-next-instant-gratification)

# ---
# 
# **From now on, I'll share methods that were not shared on the forum.**
# 
# # 4. Digging deeper into make\_classification
# 
# I looked carefully into the source code of sklearn's `make_classification` function. I observed that standard deviation of the features gather around 1 and 3.7 only when `n_redundant` and `n_repeated` parameters are 0. So I set `n_features` = `n_informative` + `n_useless`.
# 
# 1. Generates `n_clusters_per_class` cluster centroids per class in the `n_features` feature space.
# 2. Creates `n_features` gaussian distributions.
# 3. Add `n_clusters_per_class` cluster centroids to `n_informative` features for each class. For example, if `n_clusters_per_class` is 2 and `n_class` is 2(pos, neg), 4 centroids are generated, then two are assigned to pos class and the other two are assigned to the neg class. Among two centroids that are assigned to pos class, first centroid is added to half of the positive samples and the second centroid is added to the other half of the positive samples. It is the same of negative samples.
# 4. Dot some random gaussian distributed matrix with the `n_informative` features.
# 5. Flip labels according to `flip_y` ratio.
# 6. Shuffle the feature order.
# 
# **So to sum up, the positive examples were sampled in different `n_clusters_per_class` gaussian distributions and the negative examples were sampled in different `n_clusters_per_class` gaussian distributions.**
# 
# Therefore if we find this `n_clusters_per_class` and find the centroids, we might come up with a perfect classifier, which classifies everything correctly except for the `flip_y` examples.

# # 5. Gaussian Mixture
# 
# I found sklearn's `GaussianMixture` will hopefully find the means(cluster centroids) and the covariances of the gaussian distribution that generated the dataset. (https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html)
# 
# After playing with Gaussian Mixture, I found that it seperates the data well when I set `n_components` to 6; 3 for positive class and 3 for negative class. (I needed to set `n_init` param to about 20 to achieve concrete estimation.)
# 
# Also, after experimenting with make\_classification function, I found out that if I had about over 900 samples, GaussianMixture with `n_components=6` and `n_init=20` will always perfectly find out the underlying distribution. It means that I will not be able to perfectly classify only with exposed data(train & public test), but can perfectly classify when I actually submit the code, since with private test set added, number of examples per each 512 subsets will be around 1024.
# 
# By this time, I achieved 0.97443 and was pretty confident that I made a perfect classifier.

# # 6. Roc Auc
# 
# But then I thought of the evaluation metric; `Roc Auc`. If I made predictions to just 0 and 1, it gave inferior result compared to the prediction with ordered numbers (like raddar did here https://www.kaggle.com/c/instant-gratification/discussion/94671#latest-554496), since there existed flipped targets.
# 
# So I generated many randomly generated but ordered submissions, and could climb up the public leaderboard to 0.97495.

# # 7. Underfitting Public Leaderboard
# 
# By then, I was thinking that the private leaderboard will depend soley on luck; where the predictions of flipped labels were placed in the ordering of predictions. Then I came up with this final idea.
# 
# If the public score is good, it means that flipped samples in the public test set are placed in 'good' positions. Then it is more likely that the flipped samples in the private test set placed in 'bad' positions. Reversely, if the public score is bad, it means that flipped samples in the public test set are not filling up the 'good' positions, so it is more likely that the flipped samples in the private test to be placed in 'good' positions, which results in good private leaderboard score.
# 
# I experimented this hypothesis with synthetic data I made with `make_classification` and confirmed that public leaderboard score and private leaderboard score have significant negative correlation.
# 
# So ironically, the submission that underfits the public leaderboard overfits the private leaderboard.
# 
# However sadly it turned out that it was not that significant and I ended up in 14th place. My best private sub would have achieved 0.976300...

# # Code to Reproduce the Result

# In[ ]:


import numpy as np
import pandas as pd

from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal

from sklearn.datasets import make_classification

import matplotlib.pyplot as plt

np.random.seed(0)


# In[ ]:


train = pd.read_csv('../input/train.csv', index_col='id')
test = pd.read_csv('../input/test.csv', index_col='id')
data = pd.concat([train, test], sort=False)


# In[ ]:


methods = ['diff', 'discrete']
methods += ['random'+str(i) for i in range(1, 100)]
pred = pd.DataFrame(index=test.index, columns=methods)

for i in range(512):
    
    cur = data[data['wheezy-copper-turtle-magic']==i]
    train_idx = [i for i in cur.index if i in train.index]
    test_idx = [i for i in cur.index if i in test.index]
    
    y_data = cur['target']
    x_data = cur.loc[:, cur.std()>2]
    cur_test = x_data.loc[test_idx]
    
    gm = GaussianMixture(6, n_init=20, max_iter=10000, tol=1e-5).fit(x_data)
    tmp = pd.DataFrame([gm.predict(x_data), y_data.values]).dropna(axis=1)
    pos_val_counts = tmp.loc[:, tmp.loc[1]==1].loc[0].value_counts()
    neg_val_counts = tmp.loc[:, tmp.loc[1]==0].loc[0].value_counts()
    pos_cluster = pos_val_counts.iloc[:3].index.astype(int)
    neg_cluster = neg_val_counts.iloc[:3].index.astype(int)
    pos_mns = [multivariate_normal(gm.means_[i], gm.covariances_[i]) for i in pos_cluster]
    neg_mns = [multivariate_normal(gm.means_[i], gm.covariances_[i]) for i in neg_cluster]
    pos_pred = np.max([pos_mn.logpdf(cur_test) for pos_mn in pos_mns], axis=0)
    neg_pred = np.max([neg_mn.logpdf(cur_test) for neg_mn in neg_mns], axis=0)
    pred.loc[cur_test.index, 'diff'] = pos_pred - neg_pred
    pred.loc[cur_test.index, 'discrete'] = pd.Series(gm.predict(cur_test)).replace(pos_cluster, -1).replace(neg_cluster, 0).abs().values

for i in range(1, 100):
    pred['random'+str(i)] = pred['discrete']
    ascending = np.arange(2, len(pred)+2)
    n_zeros = (pred['discrete']==0).sum()
    n_ones = (pred['discrete']==1).sum()
    zeros = ascending[:n_zeros].copy()
    ones = ascending[-n_ones:].copy()
    np.random.shuffle(zeros)
    np.random.shuffle(ones)
    pred['random'+str(i)][pred['discrete']==0] = zeros
    pred['random'+str(i)][pred['discrete']==1] = ones


# In[ ]:


pred.head(20)


# In[ ]:


for i in methods:
    sub = pd.DataFrame({'id': pred.index, 'target': pred[i].values})
    sub.to_csv(i+'.csv', index=False)
    sub.head(20)


# Two subs with lowest public scores were `random73` which got public score of 0.97385, and `random10` which got public score of 0.97396.
# 
# However the best private sub was `random33`.
