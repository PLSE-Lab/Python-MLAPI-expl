#!/usr/bin/env python
# coding: utf-8

# # Dimensionality Reduction - PCA, LDA, t-SNE, UMAP
# 
# ## Table of Contents
# ### 1. [Introduction](#Introduction)
# ### 2. [Principal Component Analysis](#Principal Component Analysis)
# ### 3. [Linear Discriminant Analysis](#Linear Discriminant Analysis)
# ### 4. [t-Distributed Stochastic Neighbor Embedding](#t-Distributed Stochastic Neighbor Embedding)
# ### 5. [Uniform Manifold Approximation and Projections](#Uniform Manifold Approximation and Projection)

# ## Introduction
# 
# When we deal to a machine learning problem the first operation we should do is to look at data. In fact, a very common problem that leads to build an inaccurate model is the so-called "Curse of Dimensionality". The dataset usually contains features not giving information because they can be very correlated to other dimensions or be irrilevant for a problem. 
# 
# So, the main aim of a dimensionality reduction technique is to decrease the number of features without loss of information; the algorithm projects a d-dimension feature space onto a smaller subspace k (where k < d) in order to reduce memory usage, computational cost and to avoid the curse of dimensionality.
# 
# This notebook presents a theoric summary [WIP], a scratch code (only for PCA and LDA)[WIP] and the principal library implementation for :
# 
#     * Principal Component Analysis
#     * Linear Discriminant Analysis
#     * t-Distributed Stochastic Neighbor Embedding
#     * Uniform Manifold Approximation and Projections
# 

# In[ ]:


import os
import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from numpy import linalg as LA
from scipy import optimize

from umap import UMAP
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from matplotlib import pyplot as plt
import imageio
import tqdm


# ## Dataset: Sign Language MNIST
# 
# The Sign Language MNIST dataset is an American Sign Language letter dataset of hand gestures represent a multi-class problem with 24 classes of letters (excluding J and Z which require motion).

# In[ ]:


fig = plt.figure(figsize=(20,10))
image = imageio.imread("../input/amer_sign2.png")
plt.imshow(image)


# In[ ]:


df = pd.read_csv("../input/sign_mnist_train.csv")
df.head()
df.shape


# In[ ]:


letter2encode = {'A': 0,'B': 1,'C': 2,'D': 3,'E': 4,'F': 5,'G': 6,'H': 7,'I': 8,'K': 9,'L': 10,'M': 11,
                'N': 12,'O': 13,'P': 14,'Q': 15,'R': 16,'S': 17,'T': 18,'U': 19,'V': 20,'W': 21,'X': 22, 'Y': 23}

def fix_label_gap(l):
    if(l>=9):
        return (l-1)
    else:
        return l

def encode(character):
    return letter2encode[character]

df['label'] = df['label'].apply(fix_label_gap)


# In[ ]:


WORD = 'THANKS'

word = np.array(list(WORD))
embedded_word = list(map(encode, word))
print(embedded_word)

reduced_df = df[df['label'].isin(embedded_word)]

reduced_df.shape
X = reduced_df.loc[:, reduced_df.columns != 'label'].values

len(X)
y = reduced_df['label'].values

plt.imshow(X[12].reshape(28,28))


# In[ ]:


X_PCA = PCA(n_components=5).fit_transform(X)
X_LDA = LDA(n_components=5).fit_transform(X,y)
X_TSNE = TSNE().fit_transform(X)
X_UMAP = UMAP(n_neighbors=15,
                      min_dist=0.1,
                      metric='correlation').fit_transform(X)


# In[ ]:


fig = plt.figure(figsize=(50,40))
plt.subplot(2,2,1)
plt.scatter(X_PCA[:,0], X_PCA[:,1], c=y, cmap='Set1')
plt.title("Principal Component Analysis", fontsize=40)
plt.subplot(2,2,2)
plt.scatter(X_UMAP[:,0], X_UMAP[:,1], c=y, cmap='Set1')
plt.title("Uniform Manifold Approximation and Projections", fontsize=40)
plt.subplot(2,2,3)
plt.scatter(X_LDA[:,0], X_LDA[:,1], c=y, cmap='Set1')
plt.title("Linear Discriminant Analysis", fontsize=40)
plt.subplot(2,2,4)
plt.scatter(X_TSNE[:,0], X_TSNE[:,1], c=y, cmap='Set1')
plt.title("t-Distributed Stochastic Neighbor Embedding", fontsize=40)
plt.show()

