#!/usr/bin/env python
# coding: utf-8

# ### How to view data sets with more than 3 dimensions
# 
# Visualization is one of the most important tasks during the exploratory analysis of any set of data. In general, we should not limit our interpretations to descriptive statistics.
# 
# FJ Anscombe has proven this by showing 4 sets of data that, despite having virtually identical averages, variances and correlations, are totally different:
# 
# ![](https://media.licdn.com/dms/image/C4E12AQFS1wg6hiH14Q/article-inline_image-shrink_1500_2232/0?e=1543449600&v=beta&t=K9KQZLY4MUBEbzugqEsisOLQh5A9ZsBAOJNIN0z5CI0)
# 
# These sets, shown in the figure above, became known as the [Anscombe Quartet](https://en.wikipedia.org/wiki/Anscombe%27s_quartet). Other similar examples (a little more curious) can be seen [here](https://www.autodeskresearch.com/publications/samestats).
# 
# If on the one hand the visualization facilitates interpretation, on the other hand it is limited in relation to the number of dimensions. It is a fact that we, humans, can not see more than three dimensions.
# 

# The problem is that virtually no real problem can be explained well in up to 3 variables. In this case, a common procedure is generation of paired charts.
# 
# Considering the famous Iris data set, we would have:

# In[ ]:


# Load libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
get_ipython().run_line_magic('matplotlib', 'inline')

# Load dataset
dataset = pd.read_csv('../input/Iris.csv')

# Show pairplot
sns.pairplot(dataset, hue='Species')


# Although it produces an interesting result, this approach still fails to illustrate the joint relationship of all variables (sepal_length, sepal_width, petal_length, and petal_width).
# 
# Viewing multivariate data is a challenge, but fortunately we can rely on some statistical methods that can "compact" the data into fewer dimensions. One such method is [Principal Component Analysis (PCA).](https://en.wikipedia.org/wiki/Principal_component_analysis)
# 
# The objective of the PCA is to summarize the data through components that best explain its variance. It is important to note that in performing such a procedure, some of the information contained will certainly be lost. For data visualization, PCA is useful since most of the information can be explained by 2 or 3 dimensions.

# In[ ]:


# Do PCA
pca = PCA(2)
datasetPCA = pca.fit_transform(dataset[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']])
datasetPCA = pd.DataFrame({'pc1': datasetPCA[:, 0], 'pc2': datasetPCA[:, 1], 'class': dataset['Species']})

datasetPCA.head()


# In[ ]:


# 0.92 + 0.05 = 0.97
pca.explained_variance_ratio_


# Considering the example of Iris flowers, the Principal Component Analysis can explain, in only 2 dimensions, more than 97% of the information in the original data set, which has 4 variables.
# 
# After performing the analysis, we have as a result the chart below. It shows the dispersion of the data in 2 dimensions and can be understood as an aggregation of the previous set of graphs, which greatly facilitates the interpretation.

# In[ ]:


# Split dataset by class
setosa = datasetPCA[datasetPCA['class'] == 'Iris-setosa']
versicolor = datasetPCA[datasetPCA['class'] == 'Iris-versicolor']
virginica = datasetPCA[datasetPCA['class'] == 'Iris-virginica']


# In[ ]:


# Plot in 2D
plt.scatter(x=setosa['pc1'], y=setosa['pc2'])
plt.scatter(x=versicolor['pc1'], y=versicolor['pc2'])
plt.scatter(x=virginica['pc1'], y=virginica['pc2'])
plt.legend(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])


# In this way we can, even more clearly, verify the existence of different patterns for each type of flower.
# 
# In addition to being a powerful resource for exploratory visualizations, dimensionality reduction methods are also often used in the phase known as Feature Engineering, but this subject deserves a separate article.
# 
# Thank you!
