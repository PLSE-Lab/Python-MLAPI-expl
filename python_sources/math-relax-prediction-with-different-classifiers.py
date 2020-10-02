#!/usr/bin/env python
# coding: utf-8

# Idk what I am doing here

# In[ ]:


get_ipython().run_line_magic('pylab', 'inline')
get_ipython().run_line_magic('matplotlib', 'inline')
import json
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from time import time
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[ ]:


df = pd.read_csv("../input/eeg-data.csv")

# convert to arrays from strings
df.raw_values = df.raw_values.map(json.loads)
df.eeg_power = df.eeg_power.map(json.loads)

df.head(2)


# In[ ]:


relax = df[df.label == 'relax']
math = df[(df.label == 'math1') |
          (df.label == 'math2') |
          (df.label == 'math3') |
          (df.label == 'math4') |
          (df.label == 'math5') |
          (df.label == 'math6') |
          (df.label == 'math7') |
          (df.label == 'math8') |
          (df.label == 'math9') |
          (df.label == 'math10') |
          (df.label == 'math11') |
          (df.label == 'math12') ]

len(relax)
len(math)


# In[ ]:


def vectors_labels (list1, list2):
    def label (l):
        return lambda x: l
    X = list1 + list2
    y = list(map(label(0), list1)) + list(map(label(1), list2))
    return X, y


# In[ ]:


train_data = pd.DataFrame(columns=["delta", "theta", "low_alpha", "high_alpha",                                  "low_beta", "high_beta", "low_gamma", "mid_gamma"])
for id in math["id"]:
    one_math = math[math['id']==id]
    one_relax = relax[relax['id']==id]
    X, y = vectors_labels(one_math.eeg_power.tolist(), one_relax.eeg_power.tolist())
    X = np.matrix(X)
    y = np.array(y)
    data = pd.DataFrame(data=X,
                        index=y,
                        columns=["delta", "theta", "low_alpha", "high_alpha", \
                                 "low_beta", "high_beta", "low_gamma", "mid_gamma"])
    frames = [train_data, data]
    train_data = pd.concat(frames)
train_data.shape


# In[ ]:


train_data.index.value_counts()


# In[ ]:


train_data.info()


# In[ ]:


train_data.describe().T


# In[ ]:


import seaborn as sns
corr = train_data.corr()
figsize(8, 6)
sns.heatmap(corr)


# In[ ]:


plots = train_data.hist()


# In[ ]:


scaler = StandardScaler()
data = scaler.fit_transform(train_data)
pca = PCA(n_components=2)
pca_data = pca.fit_transform(train_data)
plot_data = train_data
plot_data["relax"] = train_data.index
scatter(pca_data[:, 0], pca_data[:, 1],        c=plot_data["relax"].apply(lambda relax: 'red' if relax else 'green'))


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(
       train_data, train_data.index, test_size=0.33, random_state=42)
X_train.shape


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=100)
forest = RandomForestClassifier(max_depth=3, random_state=42, criterion="entropy")
knn.fit(X_train, y_train)
forest.fit(X_train, y_train)


# In[ ]:


np.mean(cross_val_score(knn, X_train, y_train, cv=None, scoring="roc_auc"))


# In[ ]:


np.mean(cross_val_score(forest, X_train, y_train, cv=None, scoring="roc_auc"))


# In[ ]:


accuracy_score(y_test, knn.predict(X_test))


# In[ ]:


accuracy_score(y_test, forest.predict(X_test))


# In[ ]:


# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


# specify parameters and distributions to sample from
param_dist = {"max_depth": [3, None],
              "max_features": [1,2,3,4,5,6,7,8],
              "min_samples_split": sp_randint(2, 8),
              "min_samples_leaf": sp_randint(1, 8),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

# run randomized search
n_iter_search = 20
random_search = RandomizedSearchCV(forest, param_distributions=param_dist,
                                   n_iter=n_iter_search)

start = time()
random_search.fit(X_train, y_train)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(random_search.cv_results_)


# In[ ]:


train_sizes, train_scores, valid_scores = learning_curve(
    knn, X_train, y_train, train_sizes=[500, 1000, 1500, 3000, 5000, 10000, 15000])
pl.plot(train_sizes, train_scores)
pl.plot(train_sizes, valid_scores)
pl.show


# In[ ]:


train_sizes, train_scores, valid_scores = learning_curve(
    forest, X_train, y_train, train_sizes=[10, 30, 50, 70, 100, 150], cv=5)
pl.plot(train_sizes, train_scores, label="training score", c="green")
pl.plot(train_sizes, valid_scores, label="validation score", c="red")
pl.legend()
pl.show


# Evaluation

# In[ ]:




