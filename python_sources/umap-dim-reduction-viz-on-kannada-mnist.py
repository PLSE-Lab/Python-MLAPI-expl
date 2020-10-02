#!/usr/bin/env python
# coding: utf-8

# The objective of this notebook is to show how one of the fastest dimensionality reduction techniques UMAP (which is also a great tSNE alternative) groups similar digits of Kannada MNIST dataset. 
# 
# As we know before, the dataset is an MNIST-like dataset with first column as label and 784 (28 x 28) pixel values

# ### UMAP
# 
# Uniform Manifold Approximation and Projection (UMAP) is a dimension reduction technique that can be used for visualisation similarly to t-SNE, but also for general non-linear dimension reduction.

# ### Loading required libraries starting from `umap` 

# In[ ]:


import umap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
plt.figure(figsize=(20,10))


# ### Read Training dataset

# In[ ]:


kannada_train = pd.read_csv("../input/Kannada-MNIST/train.csv")


# ### To reduce the execution time for learning purpose, We'll take only 10K rows randomly sampled out of 60K rows from the training dataset

# In[ ]:


kannada_train_sampled = kannada_train.sample(n = 10000, random_state = 123)


# ### Looking at the label distribution after random sampling

# In[ ]:


kannada_train_sampled.groupby('label')['label'].count()


# ### Visualizing a sample digit

# In[ ]:



image = np.array(kannada_train_sampled.iloc[2,1:785], dtype='float')
pixels = image.reshape((28, 28))
plt.imshow(pixels, cmap='gray')
plt.title(kannada_train_sampled.iloc[2,0])
plt.show()


# `random_state` parameter is set for reproducibility
# 

# In[ ]:


data = kannada_train_sampled.iloc[:, 1:].values.astype(np.float32)
target = kannada_train_sampled['label'].values


# ### UMAP() function for dimensionality reduction. 
# 
# `UMAP()` takes other hyperparameters like 
# 
#     n_neighbors
#     min_dist
#     n_components
#     metric
# 
# by default, `n_components` is set to 2 which makes the reduced dimension into 2

# In[ ]:


get_ipython().run_line_magic('time', 'reduce = umap.UMAP(random_state = 123) #just for reproducibility')
get_ipython().run_line_magic('time', 'embedding = reduce.fit_transform(data)')


# ### Converting the output embedding into a dataframe 
# ### And, Renaming the columns in it for easier access

# Since we have just two dimensions, let's name it x and y

# In[ ]:


df = pd.DataFrame(embedding, columns=('x', 'y'))
df["class"] = target


# Adding the target column for visualization labelling

# ### Kannada MNIST clustered (dimensions reduced - Visulization) with UMAP

# A simple scatter plot with x and y, color from target label

# In[ ]:


plt.scatter(df.x, df.y, s= 5, c=target, cmap='Spectral')
plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
plt.title('Kannada MNIST clustered (dimensions reduced - Visulization) with UMAP', fontsize=20);


# As, we can see `UMAP` has done a pretty good job in identifying distinct groups in the dataset purely based on the dimensions / columns (pixel values).
# 
# UMAP is definitely a good-fast alternative wherever you think tSNE could be useful. 

# ### Reference
# 
# + [`umap` Python Package Github](https://github.com/lmcinnes/umap)
# + [UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction](https://arxiv.org/abs/1802.03426)
