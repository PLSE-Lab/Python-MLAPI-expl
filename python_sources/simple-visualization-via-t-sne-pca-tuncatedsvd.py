#!/usr/bin/env python
# coding: utf-8

# # Simple Visualization via t-SNE/PCA/TuncatedSVD
# Thanks for the [kernel](https://www.kaggle.com/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets), visualization via t-SNE/PCA/TruncatedSVD can be executed easily.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import time
import os
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn import preprocessing
import matplotlib.patches as mpatches

print(os.listdir("../input"))
import seaborn as sns
import warnings
warnings.simplefilter("ignore")
plt.style.use('ggplot')
color_pal = [x['color'] for x in plt.rcParams['axes.prop_cycle']]


# In[ ]:


train_transaction = pd.read_csv('../input/train_transaction.csv')
test_transaction = pd.read_csv('../input/test_transaction.csv')
train_identity = pd.read_csv('../input/train_identity.csv')
test_identity = pd.read_csv('../input/test_identity.csv')


# In[ ]:


# Check imbalance
train_transaction.groupby('isFraud')     .count()['TransactionID']     .plot(kind='barh',
          title='Distribution of Target in Train',
          figsize=(15, 3))
plt.show()


# In[ ]:


# Merge dataset
train = train_identity.merge(train_transaction[['TransactionID','TransactionDT','isFraud']],on=['TransactionID'])
test = test_identity.merge(test_transaction[['TransactionID','TransactionDT']],on=['TransactionID'])
#del train_identity, test_identity, train_transaction, test_transaction


# In[ ]:


label = 'isFraud'
X_train = train.drop(label, axis=1)
y_train = train[label]

X_test = test


# In[ ]:


# Label Encoding
for f in X_train.columns:
    if X_train[f].dtype=='object' or X_test[f].dtype=='object': 
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(X_train[f].values) + list(X_test[f].values))
        X_train[f] = lbl.transform(list(X_train[f].values))
        X_test[f] = lbl.transform(list(X_test[f].values))   


# In[ ]:


# Filling NaN series
X_train = X_train.fillna(-999)


# In[ ]:


# https://www.kaggle.com/pavansanagapati/anomaly-detection-credit-card-fraud-analysis
random_state = 42
n_components = 2

# T-SNE Implementation
t0 = time.time()
X_reduced_tsne = TSNE(n_components=n_components, random_state=random_state).fit_transform(X_train.values)
t1 = time.time()
print("T-SNE took {:.2} s".format(t1-t0))

# PCA Implementation
t0 = time.time()
X_reduced_pca = PCA(n_components=n_components, random_state=random_state).fit_transform(X_train.values)
t1 = time.time()
print("PCA took {:.2} s".format(t1-t0))

# TruncatedSVD
t0 = time.time()
X_reduced_svd = TruncatedSVD(n_components=n_components, algorithm='randomized', random_state=random_state).fit_transform(X_train.values)
t1 = time.time()
print("Truncated SVD took {:.2} s".format(t1-t0))


# In[ ]:


f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24,6))
# labels = ['No Fraud', 'Fraud']
f.suptitle('Clusters using Dimensionality Reduction', fontsize=14)

blue_patch = mpatches.Patch(color='#0A0AFF', label='No Fraud')
red_patch = mpatches.Patch(color='#AF0000', label='Fraud')

# t-SNE scatter plot
ax1.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y_train == 0), cmap='coolwarm', label='No Fraud', linewidths=2)
ax1.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y_train == 1), cmap='coolwarm', label='Fraud', linewidths=2)
ax1.set_title('t-SNE', fontsize=14)
ax1.grid(True)
ax1.legend(handles=[blue_patch, red_patch])

# PCA scatter plot
ax2.scatter(X_reduced_pca[:,0], X_reduced_pca[:,1], c=(y_train == 0), cmap='coolwarm', label='No Fraud', linewidths=2)
ax2.scatter(X_reduced_pca[:,0], X_reduced_pca[:,1], c=(y_train == 1), cmap='coolwarm', label='Fraud', linewidths=2)
ax2.set_title('PCA', fontsize=14)
ax2.grid(True)
ax2.legend(handles=[blue_patch, red_patch])

# TruncatedSVD scatter plot
ax3.scatter(X_reduced_svd[:,0], X_reduced_svd[:,1], c=(y_train == 0), cmap='coolwarm', label='No Fraud', linewidths=2)
ax3.scatter(X_reduced_svd[:,0], X_reduced_svd[:,1], c=(y_train == 1), cmap='coolwarm', label='Fraud', linewidths=2)
ax3.set_title('Truncated SVD', fontsize=14)
ax3.grid(True)
ax3.legend(handles=[blue_patch, red_patch])

plt.show()

