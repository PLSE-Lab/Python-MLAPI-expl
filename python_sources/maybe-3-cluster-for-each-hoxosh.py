#!/usr/bin/env python
# coding: utf-8

# * Use BGMM to see how many clusters[](http://) are there.  
# https://scikit-learn.org/stable/modules/generated/sklearn.mixture.BayesianGaussianMixture.html
# 

# Use BGMM(BayesianGaussianMixtureModel), n_components = 4  
# If 2 clusters (1 cluster for each class), clusters will shrink. 

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.mixture import BayesianGaussianMixture
from sklearn.datasets import make_classification
from sklearn.feature_selection import VarianceThreshold


# In[ ]:


# READ DATA
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
# INITIALIZE VARIABLES
cols = [c for c in train.columns if c not in ['id', 'target']]
cols.remove('wheezy-copper-turtle-magic')


# # Part 1
# make 4 clusters and see distribution

# Competition dataset

# In[ ]:


cnt_compe_data = np.zeros(512*4)
for i in tqdm(range(512)):
    # ONLY TRAIN WITH DATA WHERE WHEEZY EQUALS I
    train2 = train[train['wheezy-copper-turtle-magic']==i]
    test2 = test[test['wheezy-copper-turtle-magic']==i]
    idx1 = train2.index; idx2 = test2.index
    train2.reset_index(drop=True,inplace=True)

    # FEATURE SELECTION (USE APPROX 40 OF 255 FEATURES)
    sel = VarianceThreshold(threshold=1.5).fit(train2[cols])
    train3 = sel.transform(train2[cols])
    test3 = sel.transform(test2[cols])

    # FIT BGMM
    X = np.concatenate([train3, test3], axis=0)
    gmm = BayesianGaussianMixture(n_components=4, verbose=0, max_iter=10000)
    gmm.fit(X)
    clusters = gmm.predict(X)
    cnt_compe_data[i*4:(i+1)*4] = pd.value_counts(clusters).values


# In[ ]:


plt.hist(cnt_compe_data, bins=np.arange(0, 800, 20))
plt.show()


# * Simulation 1 (n_clusters_per_class = 1)

# In[ ]:


np.random.seed(71)
cnt_dummy_2 = np.zeros(512*4)
for i in tqdm(range(512)):
    X_dummy, y_dummy = make_classification(
        n_samples=768, n_features=512, n_informative=40, n_redundant=0, n_repeated=0, 
        n_classes=2, n_clusters_per_class=1, flip_y=0.05
    )

    sel_sim = VarianceThreshold(threshold=1.5).fit(X_dummy)
    X_dummy = sel_sim.transform(X_dummy)

    gmm = BayesianGaussianMixture(n_components=4, verbose=0, max_iter=10000)
    gmm.fit(X_dummy)
    clusters_sim = gmm.predict(X_dummy)
    cnt_dummy_2[i*4:(i+1)*4] = pd.value_counts(clusters_sim).values


# In[ ]:


plt.hist(cnt_dummy_2, bins=np.arange(0, 800, 20))
plt.show()


# * Simulation 2 (n_clusters_per_class = 2)

# In[ ]:


np.random.seed(71)
cnt_dummy_4 = np.zeros(512*4)
for i in tqdm(range(512)):
    X_dummy, y_dummy = make_classification(
        n_samples=768, n_features=512, n_informative=40, n_redundant=0, n_repeated=0, 
        n_classes=2, n_clusters_per_class=2, flip_y=0.05
    )

    sel_sim = VarianceThreshold(threshold=1.5).fit(X_dummy)
    X_dummy = sel_sim.transform(X_dummy)

    gmm = BayesianGaussianMixture(n_components=4, verbose=0, max_iter=10000)
    gmm.fit(X_dummy)
    clusters_sim = gmm.predict(X_dummy)
    cnt_dummy_4[i*4:(i+1)*4] = pd.value_counts(clusters_sim).values


# In[ ]:


plt.hist(cnt_dummy_4, bins=np.arange(0, 800, 20))
plt.show()


# * Simulation 3 (n_clusters_per_class = 3)

# In[ ]:


np.random.seed(71)
cnt_dummy_6 = np.zeros(512*4)
for i in tqdm(range(512)):
    X_dummy, y_dummy = make_classification(
        n_samples=768, n_features=512, n_informative=40, n_redundant=0, n_repeated=0, 
        n_classes=2, n_clusters_per_class=3, flip_y=0.05
    )

    sel_sim = VarianceThreshold(threshold=1.5).fit(X_dummy)
    X_dummy = sel_sim.transform(X_dummy)

    gmm = BayesianGaussianMixture(n_components=4, verbose=0, max_iter=10000)
    gmm.fit(X_dummy)
    clusters_sim = gmm.predict(X_dummy)
    cnt_dummy_6[i*4:(i+1)*4] = pd.value_counts(clusters_sim).values


# In[ ]:


plt.hist(cnt_dummy_6, bins=np.arange(0, 800, 20))
plt.show()


# OH...

# # Part 2
# 6 clusters

# competition dataset

# In[ ]:


cnt_compe_data = np.zeros(512*6)
for i in tqdm(range(512)):
    # ONLY TRAIN WITH DATA WHERE WHEEZY EQUALS I
    train2 = train[train['wheezy-copper-turtle-magic']==i]
    test2 = test[test['wheezy-copper-turtle-magic']==i]
    idx1 = train2.index; idx2 = test2.index
    train2.reset_index(drop=True,inplace=True)

    # FEATURE SELECTION (USE APPROX 40 OF 255 FEATURES)
    sel = VarianceThreshold(threshold=1.5).fit(train2[cols])
    train3 = sel.transform(train2[cols])
    test3 = sel.transform(test2[cols])

    # FIT BGMM
    X = np.concatenate([train3, test3], axis=0)
    gmm = BayesianGaussianMixture(n_components=6, verbose=0, max_iter=10000)
    gmm.fit(X)
    clusters = gmm.predict(X)
    cnt_compe_data[i*6:(i+1)*6] = pd.value_counts(clusters).values


# In[ ]:


plt.hist(cnt_compe_data, bins=np.arange(0, 800, 20))
plt.show()


# * Simulation 1 (n_clusters_per_class = 1)

# In[ ]:


np.random.seed(71)
cnt_dummy_2 = np.zeros(512*6)
for i in tqdm(range(512)):
    X_dummy, y_dummy = make_classification(
        n_samples=768, n_features=512, n_informative=40, n_redundant=0, n_repeated=0, 
        n_classes=2, n_clusters_per_class=1, flip_y=0.05
    )

    sel_sim = VarianceThreshold(threshold=1.5).fit(X_dummy)
    X_dummy = sel_sim.transform(X_dummy)

    gmm = BayesianGaussianMixture(n_components=6, verbose=0, max_iter=10000)
    gmm.fit(X_dummy)
    clusters_sim = gmm.predict(X_dummy)
    cnt_dummy_2[i*6:(i+1)*6] = pd.value_counts(clusters_sim).values


# In[ ]:


plt.hist(cnt_dummy_2, bins=np.arange(0, 800, 20))
plt.show()


# * Simulation 2 (n_clusters_per_class = 2)

# In[ ]:


np.random.seed(71)
cnt_dummy_4 = np.zeros(512*6)
for i in tqdm(range(512)):
    X_dummy, y_dummy = make_classification(
        n_samples=768, n_features=512, n_informative=40, n_redundant=0, n_repeated=0, 
        n_classes=2, n_clusters_per_class=2, flip_y=0.05
    )

    sel_sim = VarianceThreshold(threshold=1.5).fit(X_dummy)
    X_dummy = sel_sim.transform(X_dummy)

    gmm = BayesianGaussianMixture(n_components=6, verbose=0, max_iter=10000)
    gmm.fit(X_dummy)
    clusters_sim = gmm.predict(X_dummy)
    cnt_dummy_4[i*6:(i+1)*6] = pd.value_counts(clusters_sim).values


# In[ ]:


plt.hist(cnt_dummy_4, bins=np.arange(0, 800, 20))
plt.show()


# * Simulation 3 (n_clusters_per_class = 3)

# In[ ]:


np.random.seed(71)
cnt_dummy_6 = np.zeros(512*6)
for i in tqdm(range(512)):
    X_dummy, y_dummy = make_classification(
        n_samples=768, n_features=512, n_informative=40, n_redundant=0, n_repeated=0, 
        n_classes=2, n_clusters_per_class=3, flip_y=0.05
    )

    sel_sim = VarianceThreshold(threshold=1.5).fit(X_dummy)
    X_dummy = sel_sim.transform(X_dummy)

    gmm = BayesianGaussianMixture(n_components=6, verbose=0, max_iter=10000)
    gmm.fit(X_dummy)
    clusters_sim = gmm.predict(X_dummy)
    cnt_dummy_6[i*6:(i+1)*6] = pd.value_counts(clusters_sim).values


# In[ ]:


plt.hist(cnt_dummy_6, bins=np.arange(0, 800, 20))
plt.show()


# In[ ]:




