#!/usr/bin/env python
# coding: utf-8

# <img src="https://digmet.files.wordpress.com/2014/12/step2-nsa-netvizz.png" width="650px" height="650px"/> 

# # Visualizing High Dimensional Clusters

# ## Contents
# 1. [Introduction:](#1)
# 1. [Imports:](#2)
# 1. [Read the Data:](#3)
# 1. [Exploration/Engineering:](#4)
# 1. [Clustering:](#5)
# 1. [**Method #1:** *Principal Component Analysis* (PCA):](#6)
# 1. [**Method #2:** *T-Distributed Stochastic Neighbor Embedding* (T-SNE):](#7)
# 1. [Conclusion:](#8)
# 1. [Closing Remarks:](#9)

# <a id="1"></a>
# # Introduction:

# In this notebook we will be exploring two different methods that can be used to visualize [clusters](https://en.wikipedia.org/wiki/Cluster_analysis) that were formed on high-dimensional data (data with more than three dimensions).
# 
# First, we will clean our data so that it's in a proper format for clustering, then, we will divide the data into three different clusters using [K-Means Clustering](https://en.wikipedia.org/wiki/K-means_clustering). After that, we will go ahead and visualize our three clusters using our two methods: [Principal Component Analysis](https://en.wikipedia.org/wiki/Principal_component_analysis) (PCA), and [T-Distributed Stochastic Neighbor Embedding](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding) (T-SNE).
# 
# The data we will be using will be the [Forest Cover Type Dataset](https://www.kaggle.com/uciml/forest-cover-type-dataset).

# <a id="2"></a>
# # Imports:

# In[ ]:


#Basic imports
import numpy as np
import pandas as pd

#sklearn imports
from sklearn.decomposition import PCA #Principal Component Analysis
from sklearn.manifold import TSNE #T-Distributed Stochastic Neighbor Embedding
from sklearn.cluster import KMeans #K-Means Clustering
from sklearn.preprocessing import StandardScaler #used for 'Feature Scaling'

#plotly imports
import plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot


# <a id="3"></a>
# # Read the data:

# In[ ]:


#df is our original DataFrame
df = pd.read_csv("../input/covtype.csv")


# <a id="4"></a>
# # Exploration/Engineering:

# This is not a particularly important section of the Kernel as the bulk of the interesting work will be done in the next few sections. Feel free to skim this part, if you want.

# First, we construct a new DataFrame, `X` that we can modify. `X` will begin as a 'copy' of the original DataFrame, `df`.

# In[ ]:


X = df.copy()


# Any missing values?

# In[ ]:


X.isnull().sum()


# Sweet! No missing values. That saves us quite a bit of work.

# In[ ]:


X.head()


# If we look at the columns: `X["Horizontal_Distance_To_Hydrology"]` and `X[Vertical_Distance_To_Hydrology"]`, we see that we can create from them, a new column `X[Distance_To_Hydrology]`, which measures the shortest distance to Hydrology. We can calculate the values of this column through using the equation from the [Pythagorean Theorem](https://en.wikipedia.org/wiki/Pythagorean_theorem).

# In[ ]:


X["Distance_To_Hydrology"] = ( (X["Horizontal_Distance_To_Hydrology"] ** 2) + (X["Vertical_Distance_To_Hydrology"] ** 2) ) ** (0.5)


# Now that we have `X["Distance_To_Hydrology"]`, and because there's nothing extra special about Vertical or Horizontal Distances to Hydrology, we can drop the original two columns:

# In[ ]:


X.drop(["Horizontal_Distance_To_Hydrology","Vertical_Distance_To_Hydrology"], axis=1, inplace=True)


# In[ ]:


X.head()


# Next, if you take a look at the values contained within `X['Cover_Type']`, you'll notice that it contains numerically-encoded [categorical data](https://en.wikipedia.org/wiki/Categorical_variable). If we head over to the column descriptions on the [Forest Cover Type Dataset](https://www.kaggle.com/uciml/forest-cover-type-dataset) page, it says that:
# 
# > *1 = "Spruce/Fir", 2 = "Lodgepole Pine", 3 = "Ponderosa Pine", 4 = "Cottonwood/WIllow", 5 = "Aspen", 6 = "Douglas-fir", and 7 = "Krummholz".*
# 
# We'll relabel our data so that the values in `X['Cover_Type']` are more descriptive of what's really contained within it. We'll also do it so that we can easily apply a [one-hot-encoding](https://www.kaggle.com/dansbecker/using-categorical-data-with-one-hot-encoding) to it, afterwards - so that `X['Cover_Type']` will be properly encoded along with the rest of the categorical data in `X`.

# In[ ]:


X['Cover_Type'].replace({1:'Spruce/Fir', 2:'Lodgepole Pine', 3:'Ponderosa Pine', 4:'Cottonwood/Willow', 5:'Aspen', 6:'Douglas-fir', 7:'Krummholz'}, inplace=True)


# In[ ]:


X.head()


# And now we can 'one-hot-encode' this column:

# In[ ]:


#We use pandas's 'get_dummies()' method
X = pd.get_dummies(X)


# In[ ]:


X.head()


# <a id="5"></a>
# # Clustering:

# Now, before we get into clustering our data, we just need to do one more thing: [feature-scale](https://en.wikipedia.org/wiki/Feature_scaling#Standardization) our [numerical variables](https://www.dummies.com/education/math/statistics/types-of-statistical-data-numerical-categorical-and-ordinal/).
# 
# We need to do this because, while each of our categorical variables hold values of either 0 or 1, some of our numerical variables hold values like 2596 and 2785. If we were to leave our data like this, then K-Means Clustering would not give us such a nice result, since K-Means Clustering measures the [euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance) between data-points. This means that, if we were to leave our numeical variables un-scaled, then most of the distance measured between points would be attributed to the larger numerical variables, rather than any of the categorical variables.
# 
# To fix this problem we will scale all of our numerical variables through the use of sklearn's [StandardScaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) tool. This tool allows us to scale each numerical variable such that each numerical variable's mean becomes 0, and it's variance becomes 1. This is a good way to make sure that all of the numerical variables are on roughly the same scale that the categorical (binary) variables are on.

# But, to make sure we scale only our numerical variables -- and not our categorical variables --, we'll split our current DataFrame, `X`, into two other DataFrames: `numer` and `cater`; feature-scale. `numer`, then recombine the two DataFrames together again into a DataFrame that is suitable for clustering.

# In[ ]:


#numer is the DataFrame that holds all of X's numerical variables
numer = X[["Elevation","Aspect","Slope","Horizontal_Distance_To_Roadways","Hillshade_9am","Hillshade_Noon","Hillshade_3pm","Horizontal_Distance_To_Fire_Points","Distance_To_Hydrology"]]


# In[ ]:


#cater is the DataFrame that holds all of X's categorical variables
cater = X[["Wilderness_Area1","Wilderness_Area2","Wilderness_Area3","Wilderness_Area4","Soil_Type1","Soil_Type2","Soil_Type3","Soil_Type4","Soil_Type5","Soil_Type6","Soil_Type7","Soil_Type8","Soil_Type9","Soil_Type10","Soil_Type11","Soil_Type12","Soil_Type13","Soil_Type14","Soil_Type15","Soil_Type16","Soil_Type17","Soil_Type18","Soil_Type19","Soil_Type20","Soil_Type21","Soil_Type22","Soil_Type23","Soil_Type24","Soil_Type25","Soil_Type26","Soil_Type27","Soil_Type28","Soil_Type29","Soil_Type30","Soil_Type31","Soil_Type32","Soil_Type33","Soil_Type34","Soil_Type35","Soil_Type36","Soil_Type37","Soil_Type38","Soil_Type39","Soil_Type40","Cover_Type_Aspen","Cover_Type_Cottonwood/Willow","Cover_Type_Douglas-fir","Cover_Type_Krummholz","Cover_Type_Lodgepole Pine","Cover_Type_Ponderosa Pine","Cover_Type_Spruce/Fir"]]


# In[ ]:


numer.head()


# In[ ]:


cater.head()


# Okay. Now that we have our separate numerical DataFrame, it's time to feature-scale it:

# In[ ]:


#Initialize our scaler
scaler = StandardScaler()


# In[ ]:


#Scale each column in numer
numer = pd.DataFrame(scaler.fit_transform(numer))


# We'll rename the columns to show that they've been scaled:

# In[ ]:


numer.columns = ["Elevation_Scaled","Aspect_Scaled","Slope_Scaled","Horizontal_Distance_To_Roadways_Scaled","Hillshade_9am_Scaled","Hillshade_Noon_Scaled","Hillshade_3pm_Scaled","Horizontal_Distance_To_Fire_Points_Scaled","Distance_To_Hydrology_Scaled"]


# Now we can re-merge our two DataFrames into a new, scaled `X`.

# In[ ]:


X = pd.concat([numer, cater], axis=1, join='inner')


# In[ ]:


X.head()


# **Time to build our clusters.**

# In this kernel, we will be visualizing only three different clusters on our data. I chose three because I found it to be a good number of clusters to help us visualize our data in a non-complicated way.

# In[ ]:


#Initialize our model
kmeans = KMeans(n_clusters=3)


# In[ ]:


#Fit our model
kmeans.fit(X)


# In[ ]:


#Find which cluster each data-point belongs to
clusters = kmeans.predict(X)


# In[ ]:


#Add the cluster vector to our DataFrame, X
X["Cluster"] = clusters


# Now that we have our clusters, we can begin visualizing our data!

# <a id="6"></a>
# # **Method #1:** *Principal Component Analysis* (PCA):

# Our first method for visualization will be [Principal Component Analysis](https://en.wikipedia.org/wiki/Principal_component_analysis) (PCA). 
# 
# PCA is an algorithm that is used for [dimensionality reduction](https://en.wikipedia.org/wiki/Dimensionality_reduction) - meaning, informally, that it can take in a DataFrame with many columns and return a DataFrame with a *reduced* number of columns that still retains much of the information from the columns of the original DataFrame. The columns of the DataFrame produced from the PCA procedure are called *Principal Components*. We will use these principal components to help us visualize our clusters in 1-D, 2-D, and 3-D space, since we cannot easily visualize the data we have in higher dimensions. For example, we can use two principal components to visualize the clusters in 2-D space, or three principal components to visualize the clusters in 3-D space.

# But first, we will create a seperate, smaller DataFrame, `plotX`, to plot our data with. The reason we create a smaller DataFrame is so that we can plot our data faster, and so that our plots do not turn out looking too messy or over-crowded.

# In[ ]:


#plotX is a DataFrame containing 5000 values sampled randomly from X
plotX = pd.DataFrame(np.array(X.sample(5000)))

#Rename plotX's columns since it was briefly converted to an np.array above
plotX.columns = X.columns


# (The reason we converted `X.sample(5000)` to a numpy array, then back to a pandas DataFrame, is so that the indices of the resulting DataFrame, `plotX`, are *'renumbered'* 0-4999. )

# Now, to visualize our data, we will build three DataFrames from `plotX` using the 'PCA' algorithm. 
# 
# The *first* DataFrame will hold the results of the PCA algorithm with only one principal component. This DataFrame will be used to visualize our clusters in *one dimension* ([**1-D**](#PCA_1D)).
# 
# The *second* DataFrame will hold the two principal components returned by the PCA algorithm with `n_components=2`. This DataFrame will aid us in our visualization of these clusters in *two dimensions* ([**2-D**](#PCA_2D)).
# 
# And the *third* DataFrame will hold the results of the PCA algorithm that returns three principal components. This DataFrame will allow us to visualize the clusters in *three dimensional space* ([**3-D**](#PCA_3D)).

# We initialize our PCA models:

# In[ ]:


#PCA with one principal component
pca_1d = PCA(n_components=1)

#PCA with two principal components
pca_2d = PCA(n_components=2)

#PCA with three principal components
pca_3d = PCA(n_components=3)


# We build our new DataFrames:

# In[ ]:


#This DataFrame holds that single principal component mentioned above
PCs_1d = pd.DataFrame(pca_1d.fit_transform(plotX.drop(["Cluster"], axis=1)))

#This DataFrame contains the two principal components that will be used
#for the 2-D visualization mentioned above
PCs_2d = pd.DataFrame(pca_2d.fit_transform(plotX.drop(["Cluster"], axis=1)))

#And this DataFrame contains three principal components that will aid us
#in visualizing our clusters in 3-D
PCs_3d = pd.DataFrame(pca_3d.fit_transform(plotX.drop(["Cluster"], axis=1)))


# (Note that, above, we performed our PCA's on data that *excluded* the `Cluster` variable.)

# Rename the columns of these newly created DataFrames:

# In[ ]:


PCs_1d.columns = ["PC1_1d"]

#"PC1_2d" means: 'The first principal component of the components created for 2-D visualization, by PCA.'
#And "PC2_2d" means: 'The second principal component of the components created for 2-D visualization, by PCA.'
PCs_2d.columns = ["PC1_2d", "PC2_2d"]

PCs_3d.columns = ["PC1_3d", "PC2_3d", "PC3_3d"]


# We concatenate these newly created DataFrames to `plotX` so that they can be used by `plotX` as columns.

# In[ ]:


plotX = pd.concat([plotX,PCs_1d,PCs_2d,PCs_3d], axis=1, join='inner')


# And we create one new column for `plotX` so that we can use it for 1-D visualization.

# In[ ]:


plotX["dummy"] = 0


# Now we divide our DataFrame, `plotX`, into three new DataFrames. 
# 
# Each of these new DataFrames will hold all of the values contained in exacltly one of the clusters. For example, all of the values contained within the DataFrame, `cluster0` will belong to 'cluster 0', and all the values contained in DataFrame, `cluster1` will belong to 'cluster 1', etc.

# In[ ]:


#Note that all of the DataFrames below are sub-DataFrames of 'plotX'.
#This is because we intend to plot the values contained within each of these DataFrames.

cluster0 = plotX[plotX["Cluster"] == 0]
cluster1 = plotX[plotX["Cluster"] == 1]
cluster2 = plotX[plotX["Cluster"] == 2]


# ## PCA Visualizations:

# In[ ]:


#This is needed so we can display plotly plots properly
init_notebook_mode(connected=True)


# <a id="PCA_1D"></a>
# ### 1-D Visualization:

# The plot below displays our three original clusters on the single *principal component* created for 1-D visualization:

# In[ ]:


#Instructions for building the 1-D plot

#trace1 is for 'Cluster 0'
trace1 = go.Scatter(
                    x = cluster0["PC1_1d"],
                    y = cluster0["dummy"],
                    mode = "markers",
                    name = "Cluster 0",
                    marker = dict(color = 'rgba(255, 128, 255, 0.8)'),
                    text = None)

#trace2 is for 'Cluster 1'
trace2 = go.Scatter(
                    x = cluster1["PC1_1d"],
                    y = cluster1["dummy"],
                    mode = "markers",
                    name = "Cluster 1",
                    marker = dict(color = 'rgba(255, 128, 2, 0.8)'),
                    text = None)

#trace3 is for 'Cluster 2'
trace3 = go.Scatter(
                    x = cluster2["PC1_1d"],
                    y = cluster2["dummy"],
                    mode = "markers",
                    name = "Cluster 2",
                    marker = dict(color = 'rgba(0, 255, 200, 0.8)'),
                    text = None)

data = [trace1, trace2, trace3]

title = "Visualizing Clusters in One Dimension Using PCA"

layout = dict(title = title,
              xaxis= dict(title= 'PC1',ticklen= 5,zeroline= False),
              yaxis= dict(title= '',ticklen= 5,zeroline= False)
             )

fig = dict(data = data, layout = layout)

iplot(fig)


# <a id="PCA_2D"></a>
# ### 2-D visualization:

# The next plot displays the three clusters on the two *principal components* created for 2-D visualization:

# In[ ]:


#Instructions for building the 2-D plot

#trace1 is for 'Cluster 0'
trace1 = go.Scatter(
                    x = cluster0["PC1_2d"],
                    y = cluster0["PC2_2d"],
                    mode = "markers",
                    name = "Cluster 0",
                    marker = dict(color = 'rgba(255, 128, 255, 0.8)'),
                    text = None)

#trace2 is for 'Cluster 1'
trace2 = go.Scatter(
                    x = cluster1["PC1_2d"],
                    y = cluster1["PC2_2d"],
                    mode = "markers",
                    name = "Cluster 1",
                    marker = dict(color = 'rgba(255, 128, 2, 0.8)'),
                    text = None)

#trace3 is for 'Cluster 2'
trace3 = go.Scatter(
                    x = cluster2["PC1_2d"],
                    y = cluster2["PC2_2d"],
                    mode = "markers",
                    name = "Cluster 2",
                    marker = dict(color = 'rgba(0, 255, 200, 0.8)'),
                    text = None)

data = [trace1, trace2, trace3]

title = "Visualizing Clusters in Two Dimensions Using PCA"

layout = dict(title = title,
              xaxis= dict(title= 'PC1',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'PC2',ticklen= 5,zeroline= False)
             )

fig = dict(data = data, layout = layout)

iplot(fig)


# <a id="PCA_3D"></a>
# ### 3-D Visualization:

# This last plot below displays our clusters on the three *principal components* created for 3-D visualization:

# In[ ]:


#Instructions for building the 3-D plot

#trace1 is for 'Cluster 0'
trace1 = go.Scatter3d(
                    x = cluster0["PC1_3d"],
                    y = cluster0["PC2_3d"],
                    z = cluster0["PC3_3d"],
                    mode = "markers",
                    name = "Cluster 0",
                    marker = dict(color = 'rgba(255, 128, 255, 0.8)'),
                    text = None)

#trace2 is for 'Cluster 1'
trace2 = go.Scatter3d(
                    x = cluster1["PC1_3d"],
                    y = cluster1["PC2_3d"],
                    z = cluster1["PC3_3d"],
                    mode = "markers",
                    name = "Cluster 1",
                    marker = dict(color = 'rgba(255, 128, 2, 0.8)'),
                    text = None)

#trace3 is for 'Cluster 2'
trace3 = go.Scatter3d(
                    x = cluster2["PC1_3d"],
                    y = cluster2["PC2_3d"],
                    z = cluster2["PC3_3d"],
                    mode = "markers",
                    name = "Cluster 2",
                    marker = dict(color = 'rgba(0, 255, 200, 0.8)'),
                    text = None)

data = [trace1, trace2, trace3]

title = "Visualizing Clusters in Three Dimensions Using PCA"

layout = dict(title = title,
              xaxis= dict(title= 'PC1',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'PC2',ticklen= 5,zeroline= False)
             )

fig = dict(data = data, layout = layout)

iplot(fig)


# ## PCA Remarks:
# 
# As we can see from the plots above: if you have data that is highly *clusterable*, then PCA is a pretty good way to view the clusters formed on the original data. Also, it would seem that visualizing the clusters is more effective when the clusters are visualized using more principle components, rather than less. For example, the 2-D plot did a better job of providing a clear visual representation of the clusters than the 1-D plot; and the 3-D plot did a better job than the 2-D plot!

# <a id="7"></a>
# # **Method #2:** *T-Distributed Stochastic Neighbor Embedding* (T-SNE):

# Our next method for visualizing our clusters is [T-Distributed Stochastic Neighbor Embedding](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding) (T-SNE).
# 
# Here is a good [video](https://www.youtube.com/watch?v=wvsE8jm1GzE) by Google that gives a quick overview of what the algorithm does. And here is a [video](https://www.youtube.com/watch?v=NEaUSP4YerM) that gives a helpful and simplified explanation of how the algorithm does what it does, if you're interested.
# 
# In short, T-SNE is an interesting and complicated machine learning algorithm that can help us visualize high-dimensional data. It is a method for performing dimensionality reduction, and it is for this reason that we can use it to help us visualize our three clusters that were built on high-dimensional data.

# Note: And just like before, we will use this algorithm to visualize our data in [**1-D**](#T-SNE_1D), [**2-D**](#T-SNE_2D), and [**3-D**](#T-SNE_3D) space!

# Once again, we create a sub-DataFrame called `plotX` that will hold a sample of the data from `X` for the purpose of visualization.

# In[ ]:


#plotX will hold the values we wish to plot
plotX = pd.DataFrame(np.array(X.sample(5000)))
plotX.columns = X.columns


# Next up, we have to decide what level of `perplexity` we would like to use for our T-SNE algorithm. The `perplexity` is a hyperparameter used in the T-SNE algorithm that greatly determines how the data returned from the algorithm is distributed.
# 
# To see the role that `perplexity` plays in shaping the distibution of the data through T-SNE, check out this clearly written, and interactive [article](https://distill.pub/2016/misread-tsne/) by some of the Engineers/Scientists at [Google Brain](https://ai.google/research/teams/brain).
# 
# I have found, through a few trials, that `perplexity = 50` works fairly well for this data, but am convinced that there probably exists a more ideal value for `perplexity` between the values of `30` and `50`. If you're up for the challenge, feel free to fork this Kernel and try to find the value for `perplexity` that best displays the clusters formed on the original data.

# In[ ]:


#Set our perplexity
perplexity = 50


# We initialize our T-SNE models:

# In[ ]:


#T-SNE with one dimension
tsne_1d = TSNE(n_components=1, perplexity=perplexity)

#T-SNE with two dimensions
tsne_2d = TSNE(n_components=2, perplexity=perplexity)

#T-SNE with three dimensions
tsne_3d = TSNE(n_components=3, perplexity=perplexity)


# We build our new DataFrames to help us visualize our data in 1-D, 2-D, and 3-D space:

# In[ ]:


#This DataFrame holds a single dimension,built by T-SNE
TCs_1d = pd.DataFrame(tsne_1d.fit_transform(plotX.drop(["Cluster"], axis=1)))

#This DataFrame contains two dimensions, built by T-SNE
TCs_2d = pd.DataFrame(tsne_2d.fit_transform(plotX.drop(["Cluster"], axis=1)))

#And this DataFrame contains three dimensions, built by T-SNE
TCs_3d = pd.DataFrame(tsne_3d.fit_transform(plotX.drop(["Cluster"], axis=1)))


# (Note that, above, we performed our T-SNE algorithms on data that *exluded* the `Cluster` variable.)

# Rename the columns of these newly created DataFrames:

# In[ ]:


TCs_1d.columns = ["TC1_1d"]

PCs_1d.columns = ["PC1_1d"]

#"TC1_2d" means: 'The first component of the components created for 2-D visualization, by T-SNE.'
#And "TC2_2d" means: 'The second component of the components created for 2-D visualization, by T-SNE.'
TCs_2d.columns = ["TC1_2d","TC2_2d"]

TCs_3d.columns = ["TC1_3d","TC2_3d","TC3_3d"]


# We concatenate these newly created DataFrames to `plotX` so that they can be used by `plotX` as columns.

# In[ ]:


plotX = pd.concat([plotX,TCs_1d,TCs_2d,TCs_3d], axis=1, join='inner')


# And we create one new column for `plotX` so that we can use it for 1-D visualization.

# In[ ]:


plotX["dummy"] = 0


# Now we divide our DataFrame, `plotX`, into three new DataFrames.
# 
# Each of these new DataFrames will hold all of the values contained in exacltly one of the clusters. For example, all of the values contained within the DataFrame, `cluster0` will belong to 'cluster 0', and all the values contained in DataFrame, `cluster1` will belong to 'cluster 1', etc.

# In[ ]:


cluster0 = plotX[plotX["Cluster"] == 0]
cluster1 = plotX[plotX["Cluster"] == 1]
cluster2 = plotX[plotX["Cluster"] == 2]


# ## T-SNE Visualizations:

# <a id="T-SNE_1D"></a>
# ### 1-D Visualization:

# The plot below displays our three original clusters on the single dimension created by T-SNE for 1-D visualization:

# In[ ]:


#Instructions for building the 1-D plot

#trace1 is for 'Cluster 0'
trace1 = go.Scatter(
                    x = cluster0["TC1_1d"],
                    y = cluster0["dummy"],
                    mode = "markers",
                    name = "Cluster 0",
                    marker = dict(color = 'rgba(255, 128, 255, 0.8)'),
                    text = None)

#trace2 is for 'Cluster 1'
trace2 = go.Scatter(
                    x = cluster1["TC1_1d"],
                    y = cluster1["dummy"],
                    mode = "markers",
                    name = "Cluster 1",
                    marker = dict(color = 'rgba(255, 128, 2, 0.8)'),
                    text = None)

#trace3 is for 'Cluster 2'
trace3 = go.Scatter(
                    x = cluster2["TC1_1d"],
                    y = cluster2["dummy"],
                    mode = "markers",
                    name = "Cluster 2",
                    marker = dict(color = 'rgba(0, 255, 200, 0.8)'),
                    text = None)

data = [trace1, trace2, trace3]

title = "Visualizing Clusters in One Dimension Using T-SNE (perplexity=" + str(perplexity) + ")"

layout = dict(title = title,
              xaxis= dict(title= 'TC1',ticklen= 5,zeroline= False),
              yaxis= dict(title= '',ticklen= 5,zeroline= False)
             )

fig = dict(data = data, layout = layout)

iplot(fig)


# <a id="T-SNE_2D"></a>
# ### 2-D Visualization:

# The next plot displays the three clusters on the two dimensions created by T-SNE for 2-D visualization:

# In[ ]:


#Instructions for building the 2-D plot

#trace1 is for 'Cluster 0'
trace1 = go.Scatter(
                    x = cluster0["TC1_2d"],
                    y = cluster0["TC2_2d"],
                    mode = "markers",
                    name = "Cluster 0",
                    marker = dict(color = 'rgba(255, 128, 255, 0.8)'),
                    text = None)

#trace2 is for 'Cluster 1'
trace2 = go.Scatter(
                    x = cluster1["TC1_2d"],
                    y = cluster1["TC2_2d"],
                    mode = "markers",
                    name = "Cluster 1",
                    marker = dict(color = 'rgba(255, 128, 2, 0.8)'),
                    text = None)

#trace3 is for 'Cluster 2'
trace3 = go.Scatter(
                    x = cluster2["TC1_2d"],
                    y = cluster2["TC2_2d"],
                    mode = "markers",
                    name = "Cluster 2",
                    marker = dict(color = 'rgba(0, 255, 200, 0.8)'),
                    text = None)

data = [trace1, trace2, trace3]

title = "Visualizing Clusters in Two Dimensions Using T-SNE (perplexity=" + str(perplexity) + ")"

layout = dict(title = title,
              xaxis= dict(title= 'TC1',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'TC2',ticklen= 5,zeroline= False)
             )

fig = dict(data = data, layout = layout)

iplot(fig)


# <a id="T-SNE_3D"></a>
# ### 3-D Visualization:

# This last plot below displays our clusters on the three dimensions created by T-SNE for 3-D visualization:

# In[ ]:


#Instructions for building the 3-D plot

#trace1 is for 'Cluster 0'
trace1 = go.Scatter3d(
                    x = cluster0["TC1_3d"],
                    y = cluster0["TC2_3d"],
                    z = cluster0["TC3_3d"],
                    mode = "markers",
                    name = "Cluster 0",
                    marker = dict(color = 'rgba(255, 128, 255, 0.8)'),
                    text = None)

#trace2 is for 'Cluster 1'
trace2 = go.Scatter3d(
                    x = cluster1["TC1_3d"],
                    y = cluster1["TC2_3d"],
                    z = cluster1["TC3_3d"],
                    mode = "markers",
                    name = "Cluster 1",
                    marker = dict(color = 'rgba(255, 128, 2, 0.8)'),
                    text = None)

#trace3 is for 'Cluster 2'
trace3 = go.Scatter3d(
                    x = cluster2["TC1_3d"],
                    y = cluster2["TC2_3d"],
                    z = cluster2["TC3_3d"],
                    mode = "markers",
                    name = "Cluster 2",
                    marker = dict(color = 'rgba(0, 255, 200, 0.8)'),
                    text = None)

data = [trace1, trace2, trace3]

title = "Visualizing Clusters in Three Dimensions Using T-SNE (perplexity=" + str(perplexity) + ")"

layout = dict(title = title,
              xaxis= dict(title= 'TC1',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'TC2',ticklen= 5,zeroline= False)
             )

fig = dict(data = data, layout = layout)

iplot(fig)


# ## T-SNE Remarks:
# 
# 
# The T-SNE algorithm did a fairly decent job in visualizing the clusters, too. But, there were a few noticable differences when comparing it's resulting plots to PCA's resulting plots. 
# 
# One major difference between the plots produced by PCA and T-SNE is that T-SNE's plots seemed to have it's clusters overlapping with eachother more so than in PCA's plots. For example, if you look at the [**2-D plot**](#PCA_2D) fomed from PCA, you see three distinct sections of the data-points with strict, visible borders separating each colour into groups. Whereas, if you look at the [**2-D**](#T-SNE_2D) plot formed from T-SNE, you, again, see three sections formed within the data-points, but this time, datapoints between each cluster seem to 'intermingle' and overlap more.
# 
# The other major difference between the plots created by PCA and the plots created by T-SNE, is the shape. Because both PCA and T-SNE perform dimensionality reduction in very different ways (and with different objectives), the resulting shape or distibution of the points produced by the algorithms will almost always be very different.
# 
# Bear in mind that the plots resulting from the T-SNE algorithm are quite variable, in that they depend very heavily on the value chosen for `perplexity`.

# <a id="8"></a>
# # Conclusion:
# 
# So there you have it: two interesting methods to view clusters formed on high-dimensional data.
# One method was the standard and reliable PCA algorithm, and the other method was the somewhat more interesting and exotic T-SNE algorithm.
# 
# Both algorithms definitely have their own strengths and weaknesses when it comes to performing this task, and I'd imagine that the effectiveness of each algorithm depends largely on the type of data being given. So, in the end, it's largely up to the user which algorithm he or she prefers to use when visualizing clusterings on high-dimensional data.

# <a id="9"></a>
# # Closing Remarks:
# 
# I learned about quite alot in the making of this kernel -- about clusterability, perplexity, how to use plotly, the importance of feature-engineering, and much more. In all honesty, this was a ton of fun to make and has only further deepened my interest in [unsupervised learning](https://en.wikipedia.org/wiki/Unsupervised_learning) and data visualization. I hope to make more kernels like this in the future and to continue to sharpen my skills in this area.
# 
# If you've got any feedback for me: please leave a comment below, as I'd love to hear what you've got to say. And if you found this kernel to be interesting or useful to you, please consider giving it an upvote - I'd appreciate it very much :)
# 
# Till next time!
# *-Josh*
