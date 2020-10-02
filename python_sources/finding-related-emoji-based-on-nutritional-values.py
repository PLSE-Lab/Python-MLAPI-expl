#!/usr/bin/env python
# coding: utf-8

# **Welcome, in this notebook I would like to create a model that is capable of finding related emoji based on their nutritional values.  In addition to that I would like to plot a correlation matrix and do PCA (principal component analysis) to detect some possible outliers. **

# In[ ]:


import pandas as pd
import numpy as np

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
get_ipython().run_line_magic('matplotlib', 'notebook')

import seaborn as sns
sns.set()
sns.set(style="white")


# In[ ]:


df = pd.read_csv("../input/Emoji Diet Nutritional Data (g) - EmojiFoods (g).csv")


# In[ ]:


df.head()


# Let's start by moving columns 'name' and 'emoji' from our dataframe to a new variable (python list). Then we convert the dataframe to a numpy array.

# In[ ]:


labels = df[['name', 'emoji']].values
df.drop(['name', 'emoji'], axis=1, inplace=True)
table = np.array(df.values)


# Next we are going to define a similarity measure. For this case I have chosen to use a cosine similarity. 

# In[ ]:


def norm(v):
    # Euclidean norm for 1D np.array
    return (sum(v**2))**0.5

def similarities_of(emoji_index, table):
    a = table[emoji_index]
    results = []
    for row in table:
        # cosine similarity
        result = np.dot(a, row.T) / (norm(a) * norm(row))
        results.append(result)
    return results


# The final print function is just sorting and formating output, so that we can choose how many items we would like to print.

# In[ ]:


def print_results(item_id, table, labels, num_print=10):
    sim = similarities_of(item_id, table)
    # get original indicies of a sorted array 
    indicies = np.argsort(sim)
    indicies_m = indicies[::-1][:num_print]
    indicies_l = indicies[:num_print]
    
    print('Find similar to:')
    print(*labels[indicies_m[0]][::-1])
    
    print('----------------')
    print('The most similar:')
    # skip the first one 
    for i in indicies_m[1:]:
        print(*labels[i][::-1], round(sim[i],2))
    
    print('----------------')
    print('The least similar:')
    for i in indicies_l:
        print(*labels[i][::-1], round(sim[i],2))


# Finally here is our "lazy learning" type of the model. 

# In[ ]:


print_results(11, table, labels, num_print=8)


# Let's continue with PCA to explore and visualize our dataset. 

# In[ ]:


pca = PCA(n_components=4)
pca_result = pca.fit_transform(table)


# In[ ]:


plt.scatter(np.array(pca_result).T[0],np.array(pca_result).T[1])


# In[ ]:


fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(np.array(pca_result).T[0], np.array(pca_result).T[1], np.array(pca_result).T[2])


# In[ ]:


print(pca.explained_variance_ratio_) 


# There are two outliers to be seen. Let's find out which emoji are those. 

# In[ ]:


s = np.array(pca_result).T[0]

list(reversed(sorted(range(len(s)), key=lambda k: s[k])))[:5]


# In[ ]:


print(labels[17], labels[18])


# For some reason it is the potato and the carrot emoji.

# And lastly let us look at the correlation matrix of data fetures.  

# In[ ]:


corr = df.corr()


# In[ ]:


# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap,center=0,square=True,  linewidths=.2, cbar_kws={"shrink": .5})


# **Thanks for reading my notebook.**

# References & used code from:  
# [Seaborn examples - plot correlation matrix](https://seaborn.pydata.org/examples/many_pairwise_correlations.html)  
# [matplotlib  examples - 3D scatterplot](https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html)  
# [sklearn docs - PCA](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
