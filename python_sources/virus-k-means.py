#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


pd.read_csv('/kaggle/input/sars-coronavirus-accession/SARS_CORONAVIRUS_287BP_MN975263.1_accession_nucleotide.csv')


# In[ ]:


data = pd.read_csv('/kaggle/input/sars-coronavirus-accession/MN997409.1-4NY0T82X016-Alignment-HitTable.csv')


# In[ ]:


data.head()


# In[ ]:


data = data.rename(columns={"MN997409.1": "query acc.ver", "MN997409.1.1": "subject acc.ver",
                            "100.000":"% identity","29882":"alignment length","0":"mismatches",
                            "0.1":"gap opens","1":"q. start","29882.1":"q. end","1.1":"s. start",
                           "29882.2	":"s. end","0.0":"evalue","55182":"bit score"})


# In[ ]:


data


# In[ ]:


data = data.append(pd.Series(['MN997409.1',	'MN997409.1.1',	
                    100.000,	29882,	
                    0,	0.1,	1,	29882.1,	
                    1.1,	29882.2,	0.0,	55182], index=data.columns), ignore_index=True)


# In[ ]:


data.tail()


# In[ ]:


data


# In[ ]:


data.shape


# In[ ]:


data.describe()


# In[ ]:


data.corr()


# In[ ]:


plt.figure(figsize=(14,8))
sns.heatmap(data.corr(), vmax=1, square=True,annot=True,cmap='viridis')

plt.title('Correlation between different fearures')


# # K-Means Clustering
# 
# We will see the genome information clusters

# In[ ]:


from sklearn.cluster import KMeans


# In[ ]:


data['subject acc.ver'] = data['subject acc.ver'].astype("category").cat.codes


# In[ ]:


data.head()


# In[ ]:


X = data.iloc[:,1:].values


# In[ ]:


from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm


# In[ ]:


from sklearn.preprocessing import StandardScaler
standardized_data = StandardScaler().fit_transform(X)
print(standardized_data.shape)


# In[ ]:


range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
silohuette_scores = []
for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)
    silohuette_scores.append(silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values =             sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

plt.show()


# In[ ]:


silohuette_scores = [float(i) for i in silohuette_scores]


# In[ ]:


silohuette_scores


# In[ ]:


plt.figure(figsize=(6,6))
sns.barplot([2,3,4,5,6,7,8,9,10],
            [0.6051456304424345,
 0.7737574709178652,
 0.9304299438199628,
 0.9626586534510194,
 0.9625756535242548,
 0.9585239650538647,
 0.9614207899227608,
 0.9626169262248865,
 0.9644636830559539],palette='viridis')
plt.xlabel('Clusters')
plt.ylabel('Silhouette Score')


# In[ ]:


clusterer = KMeans(n_clusters=5, random_state=10)
cluster = clusterer.fit(X).labels_


# In[ ]:


cluster


# In[ ]:


from sklearn.preprocessing import StandardScaler
plt.figure(figsize=(15,4))
scaler = StandardScaler()
sns.heatmap(scaler.fit(cluster.cluster_centers_).transform(cluster.cluster_centers_)
            ,annot=True,xticklabels = data.columns.drop('query acc.ver'),yticklabels=['Cluster 1',
                                                            'Cluster 2',
                                                            'Cluster 3',
                                                            'Cluster 4',
                                                            'Cluster 5',])


# In[ ]:


covar_matrix = np.matmul(standardized_data.T , standardized_data)


# In[ ]:


covar_matrix.shape


# In[ ]:


from scipy.linalg import eigh 


# In[ ]:


#top2 eigenvalue
values, vectors = eigh(covar_matrix, eigvals=(9,10))


# In[ ]:


vectors.shape


# In[ ]:


#transpose
vectors = vectors.T
vectors.shape


# In[ ]:


new_coordinates = np.matmul(vectors, standardized_data.T)
print ("Resultant at new data shape: ", vectors.shape, "*", standardized_data.T.shape," = ", new_coordinates.shape)


# In[ ]:


new_coordinates = np.vstack((new_coordinates)).T

df = pd.DataFrame(data=new_coordinates, columns=("1st_principal", "2nd_principal"))
print(df.head())


# In[ ]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit_transform(X)


# In[ ]:


plt.figure(figsize=(15,2.5))
sns.heatmap(pca.components_,annot=True,xticklabels = data.columns.drop('query acc.ver'),yticklabels=['Component 1','Component 2'])
plt.title('Principal Component Linear Coefficients of Columns')


# In[ ]:


sns.set(style='whitegrid')
plt.figure(figsize=(6,6))
sns.scatterplot(X[:,0],X[:,1],hue=cluster,palette='coolwarm')
#sns.FacetGrid(df, size=6).map(plt.scatter, '1st_principal', '2nd_principal').add_legend()
plt.title('PCA Visualization of Sequences')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()


# In[ ]:


pca.explained_variance_ratio_.sum()


# In[ ]:




