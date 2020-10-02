#!/usr/bin/env python
# coding: utf-8

# # Clustering: review with python

# # 1 - Knowing the problem
# According to the documentation, we have here:
# > "... some basic data about your customers like Customer ID, age, gender, annual income and spending score. Spending Score is something you assign to the customer based on your defined parameters like customer behavior and purchasing data."
# 
# Here I intend to find groups of consumers based on their similarities.

# # 2 - Knowing the dataset

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from yellowbrick.cluster import KElbowVisualizer, silhouette_visualizer
from scipy.cluster.hierarchy import fcluster, linkage, dendrogram
from sklearn.cluster import DBSCAN
from sklearn import metrics


# In[ ]:


df = pd.read_csv("../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv")


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


df=df.drop(['CustomerID'], axis=1)


# In[ ]:


sns.pairplot(df, hue="Gender");


# # 2- Clustering

# ## 2.0 - Preprocessing
# Here, after take care of the categorical column, I'm going to bring variables into the same scale using standardization.

# In[ ]:


df=pd.get_dummies(df, drop_first=True)
dfs=StandardScaler().fit_transform(df)


# ## 2.1 - K-Means
# 
# How it works? Basically:
# * 1- inicial clusters centers (centroids) are randomly chosen
# * 2- the observations are assigned to the closest centroid, based in some distance measure
# * 3- the centroids are recalculated with the means of the observations that makes part of each cluster.
# * 2 and 3 happen repeatedly, with the goal of minimize the total within cluster variation, until have no changes or reach some tolerance.   
# 
# With K-means we have to pre-specify the number of clusters. Here, we gonna use the elbow method, that is, we gonna try some numbers of clusters (n) and observe the within clusters variation related to each n. When the sum of a cluster does not mean a considerable reduction of the within clusters variation we have a possible good choice of n (elbow on the graph).  

# In[ ]:


distortions=[]
for i in range (1,15):
    km=KMeans(n_clusters= i,
              n_init=5,  # run 5 times with different random inicial centroids
              max_iter=500,  # max iteration by run
              random_state=1)
    km.fit(dfs)
    distortions.append(km.inertia_)  # inertia = within-cluster sum-of-squares 


# In[ ]:


plt.plot(range(1,15), distortions,marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()


# In[ ]:


# using Yellowbricks - distortion metric = sum of squared distances from each point to its assigned center
vis=KElbowVisualizer(km, k=(1, 15))
vis.fit(dfs)
vis.show();


# In[ ]:


km= KMeans(n_clusters= 5,
          n_init=5, 
          max_iter= 500,
          random_state=1)


# In[ ]:


df['cluster kmeans']= km.fit_predict(dfs)
df.groupby('cluster kmeans').agg(Age_mean=('Age', 'mean'),
                                 AnIncome_mean=('Annual Income (k$)', 'mean'),
                                 SpenScore_mean=('Spending Score (1-100)', 'mean'),
                                 Gender=('Gender_Male', 'mean'),
                                 Count=('cluster kmeans', 'count'))


# So, we have 5 groups here:
# 
# A group of men/women mean age of 38 years with high income and low score.  
# A group of women mean age of 28 years with medium income and relative high score.  
# A group of women mean age of 48 years with relative low income and low  score.  
# A group of men mean age of 56 years with relative low income and low score.  
# A group of men mean age of 28 years with medium income and relative high score.  
# 
# How to know if this is a good clustering? Let's try sillhouette analysis. The silhouette plot shows, basically, how similar observations from a cluster are to observations from a neighbor cluster. The coefficient goes from -1 to 1 (well clustered).

# In[ ]:


metrics.silhouette_score(dfs, df['cluster kmeans'])


# In[ ]:


vis = silhouette_visualizer(km, dfs)
vis


# In[ ]:


df=df.drop(['cluster kmeans'], axis=1)


# Side Note: PAM is similar to K-means, but makes the centroids using the median (medoids), so it seems to be more robust to outliers. CLARA is also similar to K-means and seems like a good option to larger datasets. The dataset is splitted, PAM is aplied to subsets to choose medoids and the complete dataset is assigned to some cluster.

# ## 2.2 - Hierarquical
# 
# With this method, we don't have to specify the number of clusters and we can plot a dendrogram. 
# Here I'm going to use the agglomerative aproach where closest single observations will be combined until form a single one cluster. Divisive aproach is kind like the oposit, a single cluster will be splitted.
# But how measure the similarity between clusters? Some of the options are the complete linkage and single linkage. While complete linkage considers the distance between the most dissimilar observations, the single linkage considers the distance between the most similar observations.
# 
# How it works? Basically in case of agglomerative with complete linkage:
# 
# * 1- a distance (similarity) matrix is calculated.
# * 2- single samples (inicially considered as clusters) will be merged based on distance between the most dissimilar samples. 
# * 3- the matrix is updated
# * 4- the steps are repeated until remains one cluster with all observations.
# 

# In[ ]:


dist_matrix = linkage(dfs, method='complete', metric='euclidean')
dn=dendrogram(dist_matrix)
plt.show()


# In[ ]:


df['cluster hier'] = fcluster(dist_matrix,4, criterion='maxclust')
df.groupby('cluster hier').agg(Age_mean=('Age', 'mean'),
                                 AnIncome_mean=('Annual Income (k$)', 'mean'),
                                 SpenScore_mean=('Spending Score (1-100)', 'mean'),
                                 Gender=('Gender_Male', 'mean'),
                                 Count=('cluster hier', 'count'))


# So, we have 4 groups here:
# 
# A group of men/women mean age of 33 years with relative high income and high score.  
# A group of men/women mean age of 40 years with relative high income and very low score.  
# A group of men/women mean age of 55 years with relative low income and low score.  
# A group of men/women mean age of 27 years with relative low income and better score.  

# In[ ]:


metrics.silhouette_score(dfs, df['cluster hier'])


# In[ ]:


df=df.drop(['cluster hier'], axis=1)


# ## 2.3 - DBSCAN
# This method builds clusters based on density. MinPts and eps are parameters to be estimated. MinPoints is the minimum neighbors within eps radius of neighborhood.
# 
# How it works? Basically:
# 
# * after distance of each point to others being calculated, neighborhood is defined and points are classified in core (neighbors>=MinPoints), border (neighbors < MinPoints but is neighbor of some core point) or outlier.
# * clusters based on core points continue to be building.
# * definition of noise/outliers points.
# 
# While Kmeans and Hierarquical are good for finding spherical/convex clusters in datasets with low noise and outliers,
# DBSCAN has some advantages:  
# 
# * doesn't need the number of clusters to be pre-specified.
# * it can find differents shapes of clusters
# * it can find outliers (not all points have to be assign to a cluster)

# In[ ]:


db=DBSCAN(eps=0.8,
         min_samples=3,
         metric='euclidean')
df['cluster dbscan'] = db.fit_predict(dfs)

df.groupby('cluster dbscan').agg(Age_mean=('Age', 'mean'),
                                 AnIncome_mean=('Annual Income (k$)', 'mean'),
                                 SpenScore_mean=('Spending Score (1-100)', 'mean'),
                                 Gender=('Gender_Male', 'mean'),
                                 Count=('cluster dbscan', 'count'))


# So, we have (besides 11 outliers) 4 groups here:
# 
# A group of men mean age of 39 years with medium income and medium score.  
# A group of women mean age of 38 years with medium income and medium score.   
# A group of men mean age of 61 years with low income and very low score.  
# A group of women mean age of 44 years with high income and low score. 

# In[ ]:


metrics.silhouette_score(dfs, df['cluster dbscan'])


# Until now, kmeans seems to be the best method. Remember he gave us two groups mean age of 28 (one male and another female) with medium income and relative high score, a group of women mean age of 48 with relative low income and low score, group of men mean age of 56 years with relative low income and low score, but also a group with males and females  mean age of 38 with relative high income and low score (which seems like an oportunity).

# References:
# 
# * Practical Guide to Cluster Analysis in R: Unsupervised Machine Learning by Alboukadel Kassambara  
# * Hands-On Machine Learning with Scikit-Learn and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems from Aurelien Geron
# * Python Machine Learning: Machine Learning and Deep Learning with Python, scikit-learn, and TensorFlow from Sebastian Raschka e Vahid Mirjalili
# * Machine Learning with R by Brett Lantz
# * Hands-On Machine Learning with R by Bradley Boehmke & Brandon Greenwell
