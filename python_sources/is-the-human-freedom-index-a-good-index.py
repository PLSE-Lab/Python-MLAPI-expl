#!/usr/bin/env python
# coding: utf-8

# ## Objective:
# 
# The Human Freedom Index is a measure of how good a country is ranked amongst others countries in terms of freedom across government, society and economics variables. The idea is to implement principal component analysis in order to understand if a single indicator describe all these variables.
# 

# In[ ]:


import numpy as np                # linear algebra
import pandas as pd               # data frames
import seaborn as sns             # visualizations
import matplotlib.pyplot as plt   # visualizations
from sklearn.decomposition import PCA

import os
print(os.listdir("../input"))


# In[ ]:


df = pd.read_csv("../input/hfi_cc_2018.csv")

# Print the head of df
print(df.head())

# Print the info of df
print(df.info())

# Print the shape of df
print(df.shape)

print(np.min(df.year))


# In[ ]:


df.head()


# Preprocessing the database in order to fill the missing values.

# In[ ]:


countries = df.dropna(axis=1, how='all')
countries = countries.fillna(countries.mean())
countries.iloc[:,4:110].head()


# Identifying the correlation between those variables

# In[ ]:


# Compute the correlation matrix
corr=countries.iloc[:,4:110].corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# Principal component analysis the goal is to find a lower dimensionality representation of the variables maintaining most of the variance with a linear combination of them to create principal components which are not correlated.

# In[ ]:


# Create a PCA instance: pca
pca = PCA(n_components=5)

# Fit the pipeline to 'samples'
pca.fit(countries.iloc[:,4:110])

pca_features = pca.transform(countries.iloc[:,4:110])

# Plot the explained variances
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.xticks(features)
plt.show()


# It looks like the first component describe pretty well most of the variables.

# In[ ]:


pd.DataFrame(pca.components_)


# In[ ]:


# Assign 0th column of pca_features: xs
xs = pca_features[:,0]

# Assign 1st column of pca_features: ys
ys = countries['hf_score']

# Scatter plot xs vs ys
plt.scatter(xs, ys)
plt.axis('equal')
plt.show()


# Doing PCA we can identify one feature that describe very well all the variables that we have from a country. The Human Freedom Index was reverse engineered with this approach and now we are able to replicate the index with new data.
