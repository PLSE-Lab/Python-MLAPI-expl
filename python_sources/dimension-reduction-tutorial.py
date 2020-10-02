#!/usr/bin/env python
# coding: utf-8

# # Dimention Reduction Techniques (PCA)

# #### A short notebook in which i tried to cover some Linear as well as Non- Linear Dimension Reduction techniques.
# ### **Linear** 
# #### 1. Principal component analysis 
# #### 2. Incremental PCA
# #### 3. SVD (single value decomposition)
# #### 4. Sparse PCA
# #### 5. Random projection (a)-> Gaussian Random Projection  (b)-> Sparse Random Projection
# ### **Non-Linear (Manifold)**
# #### 1. kernel PCA
# #### 2. Isomap
# #### 3. t-SNE
# #### 4. Multi dimension Scaling
# #### 5. Independent Principal Component

# In[ ]:


import numpy as np 
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
print(os.listdir("../input"))


# In[ ]:


train=pd.read_csv("../input/fashion-mnist_train.csv")


# In[ ]:


train.head()


# In[ ]:


train_label=train['label']
del train['label']


# ## Choosing only some portion of data so that dimension reduction algorithms execute fast

# In[ ]:


# Batch data
train=train.iloc[0:1000,:]
train_label=train_label.iloc[0:1000]


# ## Helper function below which we will use to plot figures. 
# ## Will use to Plotting 1st and 2nd Principal component

# In[ ]:


# helper function for plotting
def plotting(xdf,ydf,title):
    xdf=pd.DataFrame(data=xdf, index=train.index)
    xdf=xdf.iloc[:,0:2]
    data=pd.concat((xdf,ydf),axis=1)
    data=data.rename(columns={0:'first',1:'second'})
    sns.lmplot(data=data,x='first',y='second',hue='label',fit_reg=False)
    sns.set_style('darkgrid')
    plt.title(title)


# ## PCA: 
# #### It will try to an axis to retain maximum variance. Newly derived axis called Principal component (PC1). Now next principal component will be orthogonal to all previous one

# In[ ]:


from sklearn.decomposition import PCA
pca=PCA(n_components=2,random_state=132,whiten=False)
train_pca=pca.fit_transform(train)
plotting(train_pca,train_label,'PCA')


# ## Incemental PCA
# #### Sometimes it's hard to load full data to perform dimension reduction. We divide data into batches then apply Incremetal PCA one by one

# In[ ]:


from sklearn.decomposition import IncrementalPCA
ipca=IncrementalPCA(n_components=3)
batch_size=100
for train_batch in np.array_split(train,batch_size):
    ipca.partial_fit(train_batch)
plotting(ipca.transform(train),train_label,'Incremental PCA')


# ## Sparse PCA:
# #### Sparse PCA retain some degree of sparsity controlled by hyper parameter aplha. It is slow to train

# In[ ]:


from sklearn.decomposition import SparsePCA
sparse_pca=SparsePCA(n_components=2,alpha=0.002)
plotting(sparse_pca.fit_transform(train),train_label,'Incremental PCA')


# ## Kernel PCA:
# #### Mainly use when original dataset is not lineraly seperable. Then Kernel use to plot on kernel space then dimension reduction applied on top of it

# In[ ]:


from sklearn.decomposition import KernelPCA
kpca=KernelPCA(n_components=2, kernel='rbf')
plotting(kpca.fit_transform(train),train_label,'Kernel PCA with RBF')


# ## SVD (Single value Decomposition):
# #### Way in which original data dimension reduced and further we can get original data again with some combination of smaller rank matrix

# In[ ]:


from sklearn.decomposition import TruncatedSVD
tsvd=TruncatedSVD(n_components=2, n_iter=5, algorithm='randomized')
plotting(tsvd.fit_transform(train),train_label,'SVD')


# ## Random Projection :
# #### 1. Gaussian Projection : In this projection we don't need to specify number of principal components. It can be controlled by hyper parameter eps
# #### 2. Sparse Projection : Same as Gaussian proejction with retain some sparse space from dataset. number of principal components can be controlled parameter eps. If eps high PC low and eps low PC high

# In[ ]:


from sklearn.random_projection import GaussianRandomProjection
grp=GaussianRandomProjection(n_components='auto', eps=0.5)
plotting(grp.fit_transform(train),train_label,'Gaussian Random Projection')


# In[ ]:


from sklearn.random_projection import SparseRandomProjection
srp=SparseRandomProjection(n_components='auto', eps=0.5, dense_output=False)
plotting(srp.fit_transform(train),train_label,'Gaussian Random Projection')


# ## Isomap:
# #### Calculate pairwise distance between all points where, distance is not calculated by euclidean distance. Parameter n_neighbors use to specify neighbors to choose for pair wise distance calculation

# In[ ]:


from sklearn.manifold import Isomap
isomap= Isomap(n_components=2, n_neighbors=10, n_jobs=-1)
plotting(isomap.fit_transform(train),train_label,'ISOMAP')


# ## t-SNE :
# #### It is mainly use to reduce dimention to 2 or 3 for visualization purpose. It uses probablity distribution of higher order dimension and compare with lower dimension order for dimension reduction

# In[ ]:


from sklearn.manifold import TSNE
tsne=TSNE(n_components=2, learning_rate=300, early_exaggeration=12, init='random')
plotting(tsne.fit_transform(train),train_label,'TSNE')


# ## Multi-dimensional Scaling (MDS):
# #### It tries to understand the similarity between data points of higher dimension then use that learning to reduce dimension to lower dimenion.

# In[ ]:


from sklearn.manifold import MDS 
mds=MDS(n_components=2, max_iter=100, metric=True, n_jobs=-1)
plotting(mds.fit_transform(train),train_label,'MDS')


# ## Independent Component Analysis(ICA):
# #### One of the most important dimension reduction technique is ICA. Since sometimes many signals are blended together as one feature so in order to seperate these we use ICA. It can also revert all signal back to one feature. mainly use in audio, video, signal processing

# In[ ]:


from sklearn.decomposition import FastICA


# In[ ]:


fast_ica=FastICA(n_components=2, max_iter=50, algorithm='parallel')
plotting(fast_ica.fit_transform(train),train_label,'ICA')


# # Thanks ! Dont forget to upvote this kernel :)
# 
