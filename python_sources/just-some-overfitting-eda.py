#!/usr/bin/env python
# coding: utf-8

# ## Overview
# 
# The purpose of this kernel is to take a look at the data, come up with some insights, and attempt to create a predictive model or two. This notebook is still **very** raw. I will work on it as my very limited time permits, and hope to expend it in the upcoming days and weeks.
# 
# ## Packages
# 
# First, let's load a few useful Python packages. This section will keep growing in subsequent versions of this EDA.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.linear_model import Ridge, LogisticRegression
import time
from sklearn import preprocessing
import warnings
import datetime
warnings.filterwarnings("ignore")
import gc
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import describe, rankdata
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error, roc_auc_score
import xgboost as xgb


# Now let's look at the data

# In[ ]:



import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#Loading Train and Test Data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
print("{} observations and {} features in train set.".format(train.shape[0],train.shape[1]))
print("{} observations and {} features in test set.".format(test.shape[0],test.shape[1]))


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train.target.describe()


# Since the target is binary, we see that almost 2/3 of the target belongs to the positive class, 0r 0.64 x 250 = 160.

# In[ ]:


train[train.columns[2:]].describe()


# In[ ]:


test[test.columns[1:]].describe()


# Seems that **all** of the features are numerical, with approximately 0 mean and 1.0 standard deviation. That's very interesting. 
# 
# Let's now look at the distributions of various features in the train set:

# In[ ]:


plt.figure(figsize=(12, 5))
plt.hist(train['0'].values, bins=20)
plt.title('Histogram 0 train counts')
plt.xlabel('Count')
plt.ylabel('Target')
plt.show()


# In[ ]:


plt.figure(figsize=(12, 5))
plt.hist(train['1'].values, bins=20)
plt.title('Histogram 1 train counts')
plt.xlabel('Count')
plt.ylabel('Target')
plt.show()


# In[ ]:


plt.figure(figsize=(12, 5))
plt.hist(train['123'].values, bins=20)
plt.title('Histogram 123 counts')
plt.xlabel('Count')
plt.ylabel('Target')
plt.show()


# In[ ]:


plt.figure(figsize=(12, 5))
plt.hist(test['0'].values, bins=20)
plt.title('Histogram 0 test counts')
plt.xlabel('Count')
plt.ylabel('Target')
plt.show()


# In[ ]:


plt.figure(figsize=(12, 5))
plt.hist(test['1'].values, bins=20)
plt.title('Histogram 1 test counts')
plt.xlabel('Count')
plt.ylabel('Target')
plt.show()


# In[ ]:


plt.figure(figsize=(12, 5))
plt.hist(test['123'].values, bins=20)
plt.title('Histogram 123 test counts')
plt.xlabel('Count')
plt.ylabel('Target')
plt.show()


# Based on this **very** limited subsample, seems that all of the features are approaximately normally distributed.

# Now we'll join train and test numerical featues into a single dataset, and explore how features are correlated with each other. 

# In[ ]:


train_test = pd.concat([train[train.columns[2:]], test[test.columns[1:]]])


# In[ ]:


train_test.shape


# In[ ]:


corr = train_test.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)


# Huh! That looks like all the features are almost completely uncorrelated!

# Next, we'll do some unviariate estimations of the target. We'll try to find the features that best predict the target by themselves. 

# In[ ]:


AUCs = []
Ginis = []


for i in range(300):
    AUC = roc_auc_score(train.target.values, train[str(i)].values)
    AUCs.append(AUC)
    Gini = 2*AUC - 1
    Ginis.append(Gini)


# In[ ]:


np.sort(np.abs(Ginis))[::-1]


# In[ ]:


np.argsort(np.abs(Ginis))[::-1]


# In[ ]:


[Ginis[k] for k in np.argsort(np.abs(Ginis))[::-1][:13]]


# Now we'll take a look at the top most-correlated featues and see how good of an AUC they oir their opposites get with the target. 

# In[ ]:


roc_auc_score(train.target.values,train['33'].values)


# In[ ]:


roc_auc_score(train.target.values,train['65'].values)


# In[ ]:


roc_auc_score(train.target.values,-train['217'].values)


# In[ ]:


roc_auc_score(train.target.values,-train['117'].values)


# In[ ]:


roc_auc_score(train.target.values,-train['91'].values)


# In[ ]:


roc_auc_score(train.target.values,-train['295'].values)


# In[ ]:


roc_auc_score(train.target.values,train['24'].values)


# In[ ]:


roc_auc_score(train.target.values,train['199'].values)


# In[ ]:


roc_auc_score(train.target.values,-train['80'].values)


# In[ ]:


roc_auc_score(train.target.values,-train['73'].values)


# In[ ]:


roc_auc_score(train.target.values,-train['194'].values)


# Based on these values, we'll create a **very** simple blend and see how well it does:

# In[ ]:


roc_auc_score(train.target.values,0.146*train['33'].values + 0.12*train['65'].values-0.06*train['217'].values-0.05*train['117'].values
             -0.05*train['91'].values-0.05*train['295'].values+0.05*train['24'].values+0.05*train['199'].values-
             0.05*train['80'].values- 0.05*train['73'].values-0.05*train['194'].values)


# In[ ]:


preds = (0.146*test['33'].values + 0.12*test['65'].values-0.06*test['217'].values-0.05*test['117'].values
             -0.05*test['91'].values-0.05*test['295'].values+0.05*test['24'].values+0.05*test['199'].values-
             0.05*test['80'].values- 0.05*test['73'].values-0.05*test['194'].values)
preds = rankdata(preds)/preds.shape[0]
preds


# Not bad! AUC of 0.903 is pretty good for any predictive model. But is it overfitting? There is only one way to find out! Let's submit it and see how it performs on public LB.

# In[ ]:


sample_submission = pd.read_csv('../input/sample_submission.csv')
sample_submission['target'] = preds
sample_submission.to_csv('submission.csv', index=False)


# So as expected, it **does** overfit - we "only" get 0.814 on public LB. Still, this is not bad - it beats several "benchmark" algorithms, while using **NO** machine learning!!!

# In[ ]:


pca = PCA(n_components=0.99)
pca.fit(train_test.values)


# In[ ]:


pca.n_components_


# In[ ]:


pca = PCA(n_components=0.9)
pca.fit(train_test.values)
pca.n_components_


# So PCA doesn't seem to help much here.
# 
# Let's take a look at clustering. We'll try to fit the KMeans clustering on the entire dataset, and try to see what the optimal number of clusters is.

# In[ ]:


Sum_of_squared_distances = []
K = range(1,15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(train_test)
    Sum_of_squared_distances.append(km.inertia_)


# In[ ]:


plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()


# There really doesn't seem to be an optimal number of clusters. Sems that this dataset is as scale-free adn uncorrelated as they come!

# In[ ]:


NN = 80

train_pred = 0

gini_list = [Ginis[k] for k in np.argsort(np.abs(Ginis))[::-1][:NN]]

for i in range(NN):
    if gini_list[i] > 0:
        train_pred += train[str(np.argsort(np.abs(Ginis))[::-1][i])].values
    else:
        train_pred -= train[str(np.argsort(np.abs(Ginis))[::-1][i])].values
        
roc_auc_score(train.target.values, train_pred)


# In[ ]:


test_pred = 0

gini_list = [Ginis[k] for k in np.argsort(np.abs(Ginis))[::-1][:NN]]

for i in range(NN):
    if gini_list[i] > 0:
        test_pred += test[str(np.argsort(np.abs(Ginis))[::-1][i])].values
    else:
        test_pred -= test[str(np.argsort(np.abs(Ginis))[::-1][i])].values


# In[ ]:


test_pred = rankdata(test_pred)/test_pred.shape[0]
test_pred


# In[ ]:


sample_submission['target'] = test_pred
sample_submission.to_csv('submission_80.csv', index=False)


# In[ ]:




