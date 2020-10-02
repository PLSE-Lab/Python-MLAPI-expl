#!/usr/bin/env python
# coding: utf-8

# # ML visualization with yellowbrick (3)
# 
# This notebook is the final one in a short series of notebooks I've written exploring the `yellowbricks` API. See the first one [here](https://www.kaggle.com/residentmario/ml-visualization-with-yellowbrick-1) and the second one [here](https://www.kaggle.com/residentmario/ml-visualization-with-yellowbrick-2). This last section looks at two remaining plot types useful for working with unsupervised clustering algorithms.
# 
# ## Data
# 
# Here is the data we will be using in one of the examples:

# In[1]:


import pandas as pd
pd.set_option("max_columns", None)
df = pd.read_csv("../input/recipeData.csv", encoding='latin-1')

from sklearn.preprocessing import StandardScaler
import yellowbrick as yb

df = df.dropna(subset=['Size(L)', 'OG', 'FG', 'ABV', 'IBU', 'Color', 'BoilSize', 'BoilTime'])
X = df.iloc[:, 6:6+7]
trans = StandardScaler()
trans.fit(X)
X = pd.DataFrame(trans.transform(X), columns=X.columns)
y = (df['Style'] == 'American IPA').astype(int)

df.head()


# ## KElbowVisualizer
# 
# The elbow plot is a standard way of choosing how many clusters to use when generating clusters in a dataset with unsupervised learning techniques. The basic idea is that we want to have as many clusters on the data as there are clusters in the data. If we use too few clusters, some information intrinsic to the dataset will be lost; if we use too many we will overfit and start collecting small-scale local differences. When selecting the number of clusters it pays dividends to have "just enough"; the elbow plot can be used to determine when you hit that point.

# In[ ]:


from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans

import numpy as np
np.random.seed(42)
X_sample = X.sample(5000)
y_sample = y.iloc[X_sample.index.values]

vzr = KElbowVisualizer(KMeans(), k=(4, 10))
vzr.fit(X_sample)
vzr.poof()


# On the left is a measurement of the average level of fit for the clusters. By default this is the distortion score, which is the sum of the squared distances between each of the points and the cluster centroid. A couple of other metrics are provided (and described in the `yb` documentation). A doubled axis on the right, corresponding with the dashed green plot, shows the computational cost of adding each additional cluster to the dataset&mdash;a really nice value-added to the plot. In this case, I would probably cut the number of clusters at 5 or 7; adding any more clusters than that seems to come with little reward. The lack of improvement between the fifth and sixth cluster is unusual, and bears further investigation.
# 
# Overall this is a quite nice plot type, I'm sure this is something I will be using!

# In[ ]:


from sklearn.datasets import make_blobs
X, y = make_blobs(centers=8)


# ### SilhouetteVisualizer
# 
# This function is a convenient wrapper for a plotting method originally provided in an `sklearn` code sample. That sample ([here](http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html)) still has a very good and succient description of how the silhoutte visualization works:
# 
# > Silhouette analysis can be used to study the separation distance between the resulting clusters. The silhouette plot displays a measure of how close each point in one cluster is to points in the neighboring clusters and thus provides a way to assess parameters like number of clusters visually. This measure has a range of [-1, 1].
# >
# > Silhouette coefficients (as these values are referred to as) near +1 indicate that the sample is far away from the neighboring clusters. A value of 0 indicates that the sample is on or very close to the decision boundary between two neighboring clusters and negative values indicate that those samples might have been assigned to the wrong cluster.
# 
# Silhoutte plots may be used to analyze the appropriateness of a chosen number of clusters for the given data, by bumping up the cluster count every time that you see a poorly fitted cluster. They are a heavier approach to cluster number selection than the elbow plot, but also tell you much more about the composition of the clusters themselves.
# 
# Here we apply the silhouette plot to a selection of clusters from a synthetic dataset:

# In[ ]:


from yellowbrick.cluster import SilhouetteVisualizer

clf = KMeans(n_clusters=8)
vzr = SilhouetteVisualizer(clf)
vzr.fit(X)
vzr.poof()


# We see that each of the blobs has an approximately similar number of points. The 0 blob has the most weakly fitted data; otherwise the classes are similar.
# 
# ## Conclusions
# 
# `yellowbrick` also has a couple of functions for dealing with text data, which I appreciate but which I won't spend time looking at. The highlight is the [tSNE corpus visualization](http://www.scikit-yb.org/en/latest/api/text/tsne.html). Besides that, and some minor styling stuff, that's all there is to `yellowbricks` right now!
# 
# Overall I find this library to be great overall, but definitely still unrefined. The execution of the visual ideas here feels like it could use some more refinement, e.g. the library is still in an early stage of its life and not very mature. I have some ideas of some plot types and adaptations I would like to add or see added, and there are some plot types which feel kind of lazy, like the copy-paste bar plots. However, the built-in plots that *are* good are super great and useful to have on hand. Things like `SilhouetteVisualizer`, `KElbowVisualizer`, and `ROCAUC` definitely feel like tools I will be returning to in the future.
