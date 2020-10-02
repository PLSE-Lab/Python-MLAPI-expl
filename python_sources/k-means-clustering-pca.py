#!/usr/bin/env python
# coding: utf-8

# ## K-Means Clustering and PCA of Human Activity Recognition
# 
# ***Ruslan Klymentiev***
# 
# **Date created: **July 21st, 2018

# ### Intro
# 
# Clustering was always a subject I tried to avoid (for no reason). In this project I will finally use my knowledge of clustering and PCA algorithms to explore the Human Activity Recognition dataset. 
# 
# I would love to point on resourses I have learned from:
# 
# 1. DataCamp Tutorial: [Python Machine Learning: Scikit-Learn Tutorial](https://www.datacamp.com/community/tutorials/machine-learning-python);
# 
# 2. DataCamp course: [Unsupervised Learning in Python](https://www.datacamp.com/courses/unsupervised-learning-in-python/);
# 
# 3. Cognitive Class course: [Machine Learning with Python](https://courses.cognitiveclass.ai/courses/course-v1:CognitiveClass+ML0101ENv3+2018/info);
# 
# 4. And of course [Prof. Google](http://google.com)!
# 
# ### Dataset info
# 
# Human Activity Recognition database built from the recordings of 30 subjects performing activities of daily living (ADL) while carrying a waist-mounted smartphone with embedded inertial sensors. The experiments have been carried out with a group of 30 volunteers within an age bracket of 19-48 years. Each person performed six activities (*WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING*) wearing a smartphone (Samsung Galaxy S II) on the waist. Using its embedded accelerometer and gyroscope, we captured 3-axial linear acceleration and 3-axial angular velocity at a constant rate of 50Hz. The experiments have been video-recorded to label the data manually. 

# In[ ]:


import random 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from IPython.display import display
from sklearn.cluster import KMeans 
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score, adjusted_mutual_info_score, silhouette_score
get_ipython().run_line_magic('matplotlib', 'inline')

np.random.seed(123)


# In[ ]:


Data = pd.read_csv('../input/train.csv') 


# In[ ]:


Data.sample(5)


# In[ ]:


print('Shape of the data set: ' + str(Data.shape))


# In[ ]:


#save labels as string
Labels = Data['activity']
Data = Data.drop(['rn', 'activity'], axis = 1)
Labels_keys = Labels.unique().tolist()
Labels = np.array(Labels)
print('Activity labels: ' + str(Labels_keys))


# In[ ]:


#check for missing values
Temp = pd.DataFrame(Data.isnull().sum())
Temp.columns = ['Sum']
print('Amount of rows with missing values: ' + str(len(Temp.index[Temp['Sum'] > 0])) )


# In[ ]:


#normalize the dataset
scaler = StandardScaler()
Data = scaler.fit_transform(Data)


# In[ ]:


#check the optimal k value
ks = range(1, 10)
inertias = []

for k in ks:
    model = KMeans(n_clusters=k)
    model.fit(Data)
    inertias.append(model.inertia_)

plt.figure(figsize=(8,5))
plt.style.use('bmh')
plt.plot(ks, inertias, '-o')
plt.xlabel('Number of clusters, k')
plt.ylabel('Inertia')
plt.xticks(ks)
plt.show()


# **Looks like the best value ("elbow" of the line) for k is 2 (two clusters).**

# In[ ]:


def k_means(n_clust, data_frame, true_labels):
    """
    Function k_means applies k-means clustering alrorithm on dataset and prints the crosstab of cluster and actual labels 
    and clustering performance parameters.
    
    Input:
    n_clust - number of clusters (k value)
    data_frame - dataset we want to cluster
    true_labels - original labels
    
    Output:
    1 - crosstab of cluster and actual labels
    2 - performance table
    """
    k_means = KMeans(n_clusters = n_clust, random_state=123, n_init=30)
    k_means.fit(data_frame)
    c_labels = k_means.labels_
    df = pd.DataFrame({'clust_label': c_labels, 'orig_label': true_labels.tolist()})
    ct = pd.crosstab(df['clust_label'], df['orig_label'])
    y_clust = k_means.predict(data_frame)
    display(ct)
    print('% 9s' % 'inertia  homo    compl   v-meas   ARI     AMI     silhouette')
    print('%i   %.3f   %.3f   %.3f   %.3f   %.3f    %.3f'
      %(k_means.inertia_,
      homogeneity_score(true_labels, y_clust),
      completeness_score(true_labels, y_clust),
      v_measure_score(true_labels, y_clust),
      adjusted_rand_score(true_labels, y_clust),
      adjusted_mutual_info_score(true_labels, y_clust),
      silhouette_score(data_frame, y_clust, metric='euclidean')))


# *More on clustering metrics can be found in [DataCamp Tutorial](https://www.datacamp.com/community/tutorials/machine-learning-python).*

# In[ ]:


k_means(n_clust=2, data_frame=Data, true_labels=Labels)


# **It looks like algorithm found patterns for Moving and Not-Moving activity with high level of accuracy.**
# 
# **Check how it will cluster by 6 clusters (original number of classes).**

# In[ ]:


k_means(n_clust=6, data_frame=Data, true_labels=Labels)


# **Doesn't look like good connection between clusters and original labels so I will stick with 2 clusters.**

# In[ ]:


#change labels into binary: 0 - not moving, 1 - moving
Labels_binary = Labels.copy()
for i in range(len(Labels_binary)):
    if (Labels_binary[i] == 'STANDING' or Labels_binary[i] == 'SITTING' or Labels_binary[i] == 'LAYING'):
        Labels_binary[i] = 0
    else:
        Labels_binary[i] = 1
Labels_binary = np.array(Labels_binary.astype(int))


# In[ ]:


k_means(n_clust=2, data_frame=Data, true_labels=Labels_binary)


# ### Principal component analysis (PCA)
# 
# > Principal Component Analysis is a dimension-reduction tool that can be used to reduce a large set of variables to a small set that still contains most of the information in the large set.
# 
# **2-cluster algorithm seems to fbe able to find patterns for moving/not-moving labels perfectly so far, but let's see if it can still be improved by dimension reduction. **

# In[ ]:


#check for optimal number of features
pca = PCA(random_state=123)
pca.fit(Data)
features = range(pca.n_components_)

plt.figure(figsize=(8,4))
plt.bar(features[:15], pca.explained_variance_[:15], color='lightskyblue')
plt.xlabel('PCA feature')
plt.ylabel('Variance')
plt.xticks(features[:15])
plt.show()


# **1 feature seems to be best fit for our algorithm.**

# In[ ]:


def pca_transform(n_comp):
    pca = PCA(n_components=n_comp, random_state=123)
    global Data_reduced
    Data_reduced = pca.fit_transform(Data)
    print('Shape of the new Data df: ' + str(Data_reduced.shape))


# In[ ]:


# pca_transform(n_comp=3)
# k_means(n_clust=2, data_frame=Data_reduced, true_labels=Labels)


# In[ ]:


# colors = ['green', 'blue', 'orange', 'gray', 'pink', 'red']
# fig = plt.figure(figsize=(12,8))
# ax = fig.add_subplot(111, projection='3d')
# for i in range(len(colors)):
#     x = Data_reduced[:, 0][Labels == Labels_keys[i]]
#     y = Data_reduced[:, 1][Labels == Labels_keys[i]]
#     z = Data_reduced[:, 2][Labels == Labels_keys[i]]
#     ax.scatter(xs=x, ys=y, zs=z, zdir='y', s=20, c=colors[i], alpha=0.2)

# ax.set_xlabel('First Principal Component')
# ax.set_ylabel('Second Principal Component')
# ax.set_zlabel('Third Principal Component')
# ax.set_title("PCA Scatter Plot")
# plt.show()


# In[ ]:


pca_transform(n_comp=1)
k_means(n_clust=2, data_frame=Data_reduced, true_labels=Labels_binary)


# **Inertia and Silhouette seems to be much better now after reduction. **
# 
# **Just check clustering model for 2 components.**
# 

# In[ ]:


pca_transform(n_comp=2)
k_means(n_clust=2, data_frame=Data_reduced, true_labels=Labels_binary)


# **No improvements here.**
# 
# **So far it seems like this was best I could do. Still learning clustering algorithms and I might come back to this project later.**
# 
# **If you know any interesting dataset to practice clustering on (not Iris dataset, haha), please suggest!**
