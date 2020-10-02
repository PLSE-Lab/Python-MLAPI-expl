#!/usr/bin/env python
# coding: utf-8

# ## Wine data Segmentation
# ### A explainable Clustering Analysis
# 
# ### Content of this Kernel:
# 
# * Exploratory Data Analysis with some Data Visualization
# * Data Preprocessing
# * K-Means Clustering with K selection techniques
# * Visualization of Clusters using PCA

# ## Importings
# #### Loading all necessary libraries to start the analysis

# In[ ]:


import pandas as pd
pd.set_option('display.max_columns', None)
pd.options.display.float_format = "{:.2f}".format

import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm


# #### Reading and transforming the dataset in a pandas DataFrame from a csv file

# In[ ]:


data=pd.read_csv('/kaggle/input/wine-dataset-for-clustering/wine-clustering.csv')


# ## Exploratory Data Analysis (EDA)

# #### Checking the data head, as we can see all the data is numerical: there are no categorical values

# In[ ]:


data.head()


# In[ ]:


data.info()


# #### Looking for some statistical information about each feature, we can see that the features have very diferrent scales

# In[ ]:


data.describe()


# #### Checking the skewness of our dataset. 
# * A normally distribuited data has a skewness close to zero. 
# * Skewness greather than zero means that there is more weight in the left side of the data.
# * In another hand, skewness smaller than 0 means that there is more weight in the right side of the data
# 
# <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/f/f8/Negative_and_positive_skew_diagrams_%28English%29.svg/446px-Negative_and_positive_skew_diagrams_%28English%29.svg.png">

# In[ ]:


data.skew()


# #### Plotting the histogram of each _numerical_ variable (in this case, all features), the main idea here is to visualize the data distribution for each feature. This method can bring fast insights as:
# * Check the kind of each feature distribution
# * Check data symmetry
# * Verify features frequency
# * Identify outliers

# In[ ]:


sns.set(style='white',font_scale=1.3, rc={'figure.figsize':(20,20)})
ax=data.hist(bins=20,color='red' )


# #### To reinforce our insights about the data symmetry and their outliers, we can da plot some boxplots:
# #### "A box plot is a method for graphically depicting groups of numerical data through their quartiles. The box extends from the Q1 to Q3 quartile values of the data, with a line at the median (Q2). The whiskers extend from the edges of box to show the range of the data. The position of the whiskers is set by default to 1.5*IQR (IQR = Q3 - Q1) from the edges of the box. Outlier points are those past the end of the whiskers."

# In[ ]:


data.plot( kind = 'box', subplots = True, layout = (4,4), sharex = False, sharey = False,color='black')
plt.show()


# #### Checking if there are null values, as we can see our dataset hasn`t null values

# In[ ]:


data.isnull().sum().sort_values(ascending=False).head()


# ## Data Preprocessing

# #### We are going to use a K-means algorithm, as it uses the distance as the principal metric to alocate the data in your respective cluster we need to be careful with scale, because we can give more "relevance" to large scale features and despite the low scale ones
# #### To prevent that, we can use lot of Scaling methods, in this case i`m going to Satandardize the data: to have a 0 mean and unit variance

# In[ ]:


from sklearn.preprocessing import StandardScaler

std_scaler = StandardScaler()
data_cluster=data.copy()
data_cluster[data_cluster.columns]=std_scaler.fit_transform(data_cluster)


# #### Checking if the Standardization was made correctly, looking for mean=0 and std=1

# In[ ]:


data_cluster.describe()


# #### As we are dealing with a multi-dimensional dataset, I`ll be using a Principal Component Analysis to reduce it`s dimension and make it "plottable" in a cartesian plane. Remebering that we will probably lose some information/variance in this process but it is just for viualization purposes
# 

# In[ ]:


from sklearn.decomposition import PCA
pca_2 = PCA(2)
pca_2_result = pca_2.fit_transform(data_cluster)

print ('Cumulative variance explained by 2 principal components: {:.2%}'.format(np.sum(pca_2.explained_variance_ratio_)))


# In[ ]:


sns.set(style='white', rc={'figure.figsize':(9,6)},font_scale=1.1)

plt.scatter(x=pca_2_result[:, 0], y=pca_2_result[:, 1], color='red',lw=0.1)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Data represented by the 2 strongest principal components',fontweight='bold')
plt.show()


# ## Model Implementation

# #### We are using the K-means algorithm, to choose K (number of clusters) were combined two techniques: the Silhouette Score and K-means Inertia (with Elbow analysis)

# #### First, we will compute all inertias. Basically, the lower the Inertia the better the clustering 

# In[ ]:


import sklearn.cluster as cluster

inertia = []
for i in tqdm(range(2,10)):
    kmeans = cluster.KMeans(n_clusters=i,
               init='k-means++',
               n_init=15,
               max_iter=500,
               random_state=17)
    kmeans.fit(data_cluster)
    inertia.append(kmeans.inertia_)


# #### Next, we will compute the silhouette score. Here, the bigger score the better the clustering 

# In[ ]:


from sklearn.metrics import silhouette_score

silhouette = {}
for i in tqdm(range(2,10)):
    kmeans = cluster.KMeans(n_clusters=i,
               init='k-means++',
               n_init=15,
               max_iter=500,
               random_state=17)
    kmeans.fit(data_cluster)
    silhouette[i] = silhouette_score(data_cluster, kmeans.labels_, metric='euclidean')


# In[ ]:


sns.set(style='white',font_scale=1.1, rc={'figure.figsize':(12,5)})

plt.subplot(1, 2, 1)

plt.plot(range(2,len(inertia)+2), inertia, marker='o',lw=2,ms=8,color='red')
plt.xlabel('Number of clusters')
plt.title('K-means Inertia',fontweight='bold')
plt.grid(True)

plt.subplot(1, 2, 2)

plt.bar(range(len(silhouette)), list(silhouette.values()), align='center',color= 'red',width=0.5)
plt.xticks(range(len(silhouette)), list(silhouette.keys()))
plt.grid()
plt.title('Silhouette Score',fontweight='bold')
plt.xlabel('Number of Clusters')


plt.show()


# #### As we can see, in K=3 all the metrics indicates that it is the best clusters number. So, we'll be using it

# In[ ]:


kmeans = cluster.KMeans(n_clusters=3,random_state=17,init='k-means++')
kmeans_labels = kmeans.fit_predict(data_cluster)

centroids = kmeans.cluster_centers_
centroids_pca = pca_2.transform(centroids)

pd.Series(kmeans_labels).value_counts()


# #### Here, we can visualize each feature distribution according to each cluster, in this step we can define some characteristics for each group

# In[ ]:


data2=data.copy()
data2['Cluster']=kmeans_labels

aux=data2.columns.tolist()
aux[0:len(aux)-1]

for cluster in aux[0:len(aux)-1]:
    grid= sns.FacetGrid(data2, col='Cluster')
    grid.map(plt.hist, cluster,color='red')


# #### Another approach, is looking each cluster centroid to define the cluster characteristics

# In[ ]:


centroids_data=pd.DataFrame(data=std_scaler.inverse_transform(centroids), columns=data.columns)
centroids_data.head()


# ## PCA Clusters Visualization

# In[ ]:


sns.set(style='white', rc={'figure.figsize':(9,6)},font_scale=1.1)

plt.scatter(x=pca_2_result[:, 0], y=pca_2_result[:, 1], c=kmeans_labels, cmap='autumn')
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
            marker='x', s=169, linewidths=3,
            color='black', zorder=10,lw=3)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Clustered Data (PCA visualization)',fontweight='bold')
plt.show()


# In[ ]:




