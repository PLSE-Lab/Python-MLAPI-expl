#!/usr/bin/env python
# coding: utf-8

# # Clustering consumers by a Machine Learning algorithm: K-means

# ## Problem Statement

# We have a dataset containing information about Satisfaction and loyalty of customers of a company. We intend to figure out which customers are loyal to our company and who are not! 
# 
# Note: Satisfaction is self-reported (on a scale of 1 to 15, 15 is the highest satisfaction). Loyalty is measured based on the number of purchases per year and some other factors. It is continuous data type. Range is from -2.6 to +2.6, as the data is already standardized.
# 

# ## Load the libraries and dataset

# In[ ]:


# Load libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import random
random.seed(420)
# Load the data
data = pd.read_csv("../input/consumer-classification/cunsumer classification.csv")
data.head()


# ## Draw a Scatterplot to find any relationship between satisfaction and loyalty

# In[ ]:


plt.scatter(data['Satisfaction'], data['Loyalty'])
plt.xlabel('Satisfaction')
plt.ylabel('Loyalty')
plt.show()


# Clearly there is no correlation between these two variables since this Scatterplot is scattered randomly, ultimately depicting no pattern.

# ## Build the Machine Learning model

# Let us develop a k-means algorithm - a popular clustering method - to cluster/ group the customers into several categories based on the given features: satisfaction and loyalty.

# ### K-means clustering algorithm

# In[ ]:


# Clustering
# Please note that I already performed k-means over the original dataset, but the performance was poor due to the non-scaling attribute. In other words, SATISFACTION ranges from 1 to 15, so the algorithm considered this feature more crucial compared to LOYALTY, which ranges from -2.6 to +2.6 
from sklearn import preprocessing
x_scaled = preprocessing.scale(data.copy())
x_scaled
random.seed(420)
kmeans_new = KMeans(4) # The value of optimal k=4 is determined based on Elbow method.
kmeans_new.fit(x_scaled) # The K-means algorithm is fitted on the scales dataset, not the original one.
clusters_new =  data.copy()
# Lets predict
clusters_new['cluster_pred'] = kmeans_new.fit_predict(x_scaled)
clusters_new.head()
# kmeans_new.cluster_centers_


# * Consumer 1: Belongs to a cluster where both satisfaction and loyalty are low.
# * Consumer 2: Belongs to a cluster where satisfaction is high but customers are not loyal.
# * Consumer 5: Belongs to a cluster where they are not that satisfied by loyal.
# 
# > The explanation seems fuzzy? Don't worry. We will leverage **Data Visualization** techniques to interpret the clustering more clearly.

# ## Visualizing Consumer Clustering/Segmentation

# In[ ]:


scatter = plt.scatter(clusters_new['Satisfaction'], clusters_new['Loyalty'], c=clusters_new['cluster_pred'], cmap='rainbow')
plt.title('Clustering consumers into four categories: Unfriends, Roamers, Fans, & Promoters')
plt.xlabel('Satisfaction')
plt.ylabel('Loyalty')
# classes = ['Unfriendly', 'Fans', 'Roamers', 'Promoter']; plt.legend(handles=scatter.legend_elements()[0], labels=classes, loc='upper left')
# Since the centroid of the initial cluster is determined randomly, the legends are lining randomly. I need to think about it to solve the issue. Lets skip the legend issue for now.
plt.show()


# 1. Cluster 1 (Bottom Left) **Unfriends**: Both customers' satisfaction and loyalty are poor.
# 2. Cluster 2 (Bottom Right) **Roamers**: Customer are NOT loyal even though satisfied. Such a crap! 
# 3. Cluster 3 (Upper Left) **Fans**: They are not that satisfied, still stick around our brand. We are lucky.
# 4. Cluster 4 (Upper Right) **Supported**: Customers are satisfied as well as loyal to our company. Thanks God.

# **Insight obtained from the figure above:**
# 
# > The customers fall into ***Roamers*** are risky for our company on the ground that although they are highly satisfied with our service, they are NOT loyal. 

# In[ ]:


# Similar steps can be performed for SVM
# Lets practice that at home
# Further resource: https://scikit-learn.org/stable/modules/svm.html

