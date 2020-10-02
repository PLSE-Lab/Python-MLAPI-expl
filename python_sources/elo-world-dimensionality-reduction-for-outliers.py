#!/usr/bin/env python
# coding: utf-8

# This is a kernel to explore the outliers for the [Elo Merchant Category Recommendation Competition](https://www.kaggle.com/c/elo-merchant-category-recommendation) . It can also serve as a reall good introduction to **dimensionality reduction** for visualisation and model training purposes.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA,KernelPCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn import metrics

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# Most of the features are obtained from [You're Going to Want More Categories](https://www.kaggle.com/peterhurford/you-re-going-to-want-more-categories-lb-3-737) which I believe is a greate starter kernel for anyone wanting to join in the fun! I have also added a few features of my own from the merchants dataset which seemed largely unexplored but to no avail!~

# In[ ]:


complete_features = pd.read_csv("../input/cached-features/complete_features.csv", parse_dates=["first_active_month"])
complete_features_test = pd.read_csv("../input/cached-features/complete_features_test.csv", parse_dates=["first_active_month"])


# In[ ]:


target = complete_features['target']
drops = ['card_id', 'first_active_month', 'target']
# to_remove = [c for c in complete_features if 'new' in c]
use_cols = [c for c in complete_features.columns if c not in drops]
features = list(complete_features[use_cols].columns)


# So the process here will be pretty simple I am using PCA ([Principal Components Analysis ](https://en.wikipedia.org/wiki/Principal_component_analysis)) to reduce the dimensionality to the first two Principal Components. A good way to think about this process is that it reduces our features into two uncorrelated features which are ideal for visualisations. These two features (=projections) carry the most variance of our dataset. Going a bit meta, PCA finds linear projections in our feature space to which the variance is maximized.

# In[ ]:


complete_features[features] = complete_features[features].fillna(0)
complete_features_test[features] = complete_features_test[features].fillna(0)
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(complete_features[features])
print(reduced_data.shape)
print(pca.explained_variance_ratio_)
plt.plot(reduced_data[:,0],reduced_data[:,1],'.b')
plt.show()
reduced_df = pd.DataFrame({'x' : reduced_data[:,0], 'y' : reduced_data[:,1], 'c': complete_features.target.apply(lambda x : x<-20).astype(int)})
plt.scatter(x=reduced_df['x'], y=reduced_df['y'],c=reduced_df['c'], s=1)


# So this visualisation here needs a bit of thinking. We first see that data points are spread almost uniformly across this 2D space. Most importantly that our outliers are not easily distinguishable as they are spread across the whole space and not concentrated anywhere. I make the following observations :
# * We are potentially lacking signals that the top competitors have identified and are able to produce a better distinction between the outliers and non-outliers as the outliers are essentially what produce most of the error
# * We have added too much noise with our features and we need to eliminate a number of them
# 
# **BUT** a drawback of PCA is that it performs LINEAR projections so let's try using something differen't called Kernel PCA ([Kernel Principal Components Analysis](https://en.wikipedia.org/wiki/Kernel_principal_component_analysis) which takes advantages of the kernel trick to produce non-linear projections. However, as computing the Kernel Matrix is expensive we can only do this for a sample of our data. 

# In[ ]:


kpca = KernelPCA(n_components=2, kernel='rbf')
np.random.seed(3)
s = pd.concat([complete_features[complete_features['target']>-20].sample(2207),complete_features[complete_features['target']<-20]]).reset_index(drop='index')

reduced_data_d = kpca.fit_transform(s[features])
print(reduced_data_d.shape)
plt.plot(reduced_data_d[:,0],reduced_data_d[:,1],'.b')


# In[ ]:


reduced_df_d = pd.DataFrame({'x' : reduced_data_d[:,0], 'y' : reduced_data_d[:,1], 'c': s.target.apply(lambda x : x<-20).astype(int)})
plt.scatter(x=reduced_df_d['x'], y=reduced_df_d['y'],c=reduced_df_d['c'], s=1)


# Now this seems a bit better but it is still noisy. Let's try and build some **classifiers** to see how well we can classifiy outliers. I used SVC and RandomForests which to get a sense of what is going on

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(
    reduced_df_d[['x','y']].values, s.target.apply(lambda x : -1 if x<-20 else 1).values, test_size=0.2, random_state=42)


# In[ ]:


svm = SVC(C=1,gamma=1.3, tol=0.0001, kernel='rbf', degree=5)
svm.fit(X_train, y_train)
rf = RandomForestClassifier(n_estimators=100,max_depth=5)
rf.fit(X_train, y_train)


# In[ ]:



def make_meshgrid(x, y, h=.002):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 0.1, x.max() + 0.1
    y_min, y_max = y.min() - 0.1, y.max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    if not isinstance(clf, RandomForestClassifier):
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        out = ax.contourf(xx, yy, Z, **params)
        return out
    else:
        for tree in clf.estimators_:
            Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            plt.contourf(xx, yy, Z, alpha=1. /len(clf.estimators_), cmap=plt.cm.coolwarm)

    


# Let's plot the boundaries for SVMs

# In[ ]:


fig, ax = plt.subplots(figsize=(12,10))
X0, X1 = X_train[:,0],X_train[:,1]
xx, yy = make_meshgrid(X0, X1)

plot_contours(ax, svm, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
ax.scatter(X0, X1, c=y_train, cmap=plt.cm.coolwarm, s=25, edgecolors='k')


# In[ ]:


print(metrics.classification_report(y_train, svm.predict(X_train)))


# Hmmm, SVC is not doing a pretty good job 0 precision and 0 recall for the outlier class

# In[ ]:


fig, ax = plt.subplots(figsize=(12,10))

plot_contours(ax, rf, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=1.0 / len(rf.estimators_))
ax.scatter(X0, X1, c=y_train, cmap=plt.cm.coolwarm, s=25, edgecolors='k')


# In[ ]:


print(metrics.classification_report(y_train, rf.predict(X_train)))


# RandomForest are much better, still not as high as we would like but is still good enough. We can now use this classifier by first transforming the rest of our data using KPCA and adding an extra feature called 'outlier_proba'. I will be extending this kernel later on to do this.

# In[ ]:




