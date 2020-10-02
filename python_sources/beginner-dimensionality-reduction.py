#!/usr/bin/env python
# coding: utf-8

# ## Dimensionality Reduction Techniques for MINST dataset classification (Beginner alert!)
# 
# 

# ### Intro: Why do we reduce dimensionality?
# 
# The curse of dimensionality is ever present:
# - Too many dimensions/features = complexity
# - Processing power needed increases, as data needed to fill in all those dimensions increases!
# - Reducing the number of dimensions is a good idea, but we need to do it in a way to make sure that not too   much important data is thrown away 
# 
# Lets start by importing the necessary modules/tools
# 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import seaborn as sb
import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


# ### What PCA (Principle Component Analysis) is all about!
# 
# Great video:
# https://www.youtube.com/watch?v=_UVHneBUBW0&list=PLLIH2ZW8RSTg3e8BEoMglp5XSJAWhM1VD&index=2&t=151s
# 
# - Helps extract new "better" set of variables from an existing large set of variables
# - This is done by forming a linear combination of those original variables
# - These new variables are essentially the "Principle Components"
# 
# - First Component tries to explain the maximum variance in the dataset
# - Each component thereafter, continually tries to explain the remaining variance in the dataset (i.e variance that is not explained by all the previous principle components
# 
# 
# First Lets plot some of the images representing digits

# In[ ]:


# load the Data
training = pd.read_csv("../input/train.csv")

target = training["label"]
training = training.drop("label", axis=1)

# set up figure
plt.figure(figsize=(15, 13))
for digit in range(0, 50):
    plt.subplot(5, 10, digit+1)
    data_grid = training.iloc[digit].values.reshape(28, 28)
    plt.imshow(data_grid)
plt.tight_layout(pad=1)
    
    


# ### Sklearn PCA
# 
# - We will use Python's useful sklearn toolkit to apply PCA to the training data
# 

# In[ ]:


# Set up a variable N = number of rows to apply PCA to
N = 10000
X = training[:N].values

# Y = Target Values
Y = target[:N]

# Standardize the values
X_std = StandardScaler().fit_transform(X)

# Call PCA method from sklearn toolkit
pca = PCA(n_components=4)
principle_components = pca.fit_transform(X_std)


# Now, lets plot the results by way of a scatter plot

# In[ ]:


trace = go.Scatter(
    x = principle_components[:, 0],
    y = principle_components[:, 1],
    mode="markers",
    text=Y,
    marker=dict(
        size=8, color=Y, 
        colorscale="Electric", 
        opacity=0.7
    )
)

data = [trace]

layout = go.Layout(
    title="Principle Component Analysis",
    xaxis=dict(
        title="First Principle Component",
        gridwidth=2
    ),
    yaxis=dict(
        title="Second Princple Component",
        gridwidth=2
    )
)

fig = dict(data=data, layout=layout)
py.iplot(fig)


# ### Reviewing the Graph
# 
# - The groupings of points are becoming apparent, but there is no obvious seperation between groups.
# - PCA is actually an unsupervised model, meaning it does not use class labels to train the data
# - This is a reason why the above graph shows not much "clear" distinction between clusters of points
# - The different groups 
# 

# ### KMeans Clustering Technique
# 
# - We can make up for this by applying the popular Kmeans Clustering Technique to the new data that has just passed through the PCA run
# - Basically, it attempts to split a given anonymous dataset into fixed "k" number of clusters
#     - Centroids are initialized, and data generally moves towards the closest centroid (rough explanation)
# 

# In[ ]:


# Use Kmeans method from sklearn.cluster library
kmn = KMeans(init='k-means++', n_clusters=9, random_state=0)
X_kmean = kmn.fit_predict(principle_components)

trace_k = go.Scatter(
    x=principle_components[:,0],
    y=principle_components[:,1],
    mode="markers",
    marker=dict(
        size=8,
        color=X_kmean,
        colorscale="Picnic",
    )
)
data_k = [trace_k]

layout_k = go.Layout(
    title="K-Means Clustering result",
    xaxis=dict(title='First Principle Component', gridwidth=2),
    yaxis=dict(title='Second Principle Component', gridwidth=2)
)

fig_k = dict(data=data_k, layout=layout_k)
py.iplot(fig_k)


# ### Discussion of plot
# - We see that the clusters are much more distinguishable from each other. Success
