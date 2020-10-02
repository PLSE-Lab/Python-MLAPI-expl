#!/usr/bin/env python
# coding: utf-8

# **Clustering and Multiclass Classification (Predictive Modeling / Machine Learning)**

# **INTRODUCTION:**
# 
# The data contains real information about the student's knowledge status about the subject of Electrical DC Machines.It has been obtained from UCI ML Repo. It was the Ph.D. Thesis of Dr. Hamdi Tolga Kahraman back in 2009. It is an unlabelled dataset containing 5 features explained below:
# 
# 
# STG (The degree of study time for goal object materials)
# <br>
# SCG (The degree of repetition number of user for goal object materials)
# <br>
# STR (The degree of study time of user for related objects with goal object)
# <br>
# LPR (The exam performance of user for related objects with goal object)
# <br>
# PEG (The exam performance of user for goal objects)

# In[ ]:


import os
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# In[ ]:


# Load the data from the excel file and look at column names
os.chdir("/kaggle/input")
orig = pd.read_csv('user-knowledge/User Knowledge.csv')
orig.columns


# In[ ]:


# Keep only the columns containing the data about student's knowledge
knowledge = orig.iloc[:,:5]
knowledge.head()


# In[ ]:


# Plot histograms of the featuers to visualize the data
knowledge.hist(bins=50, figsize = (8,8))
plt.show()


# **THEORY**
# 
# The most common methods used for identifying clusters or classes in unlabelled data are: 1) K-Means Clustering and 2) Hierarchical Clustering. While both are used for the same purpose, their underlying techniques are different.
# 
# ***Comparison***: It is natural to wonder which medhod to choose when performing a clustering task. There are several points of cmparison between the two: While Hierarchical Clustering is highly interpretable by looking at the dendograms, it has a higher time complexiy O(n^2) as compated to K-Means Clustering which has a linear time complexity O(n). Even by iterating K-Means for different initial clusters, it would be more efficient for clustering large amounts of data. In contrast, K-Means clustering requires the data to be continuous while Hierarchical Clustering can be run on categorical data by defining a similarity metric  rather than distance.
# 
# Note: If one of the features has a range of values much larger than the others, clustering will be completely dominated by that one feature. Hence, it is important to ensure that the range of the variables is similar by normalizing the data before clustering.
# 
# ***Number of clusters***: Sometimes we might know exactly what is the number of clusters required for further analysis. For example, while clustering the data for physical features of people for clustering them into small, medium and large sized, we know that k is 3. However in some cases we might not be pre-decided about the number of clusters. In those cases, if using K-Means Clustering, we may use the 'elbow method' to choose the optimal number of clusters or use our judgement to choose where to draw the line in the dendograms obtained from Hierarchical Clustering.

# In this analysis, we will explore K-Means clustering and look closely at the elbow method.

# In[ ]:


# Perform k-Means Clustering with values of k from 1 to 10 and plot k v/s Within Cluster Sum of Squares
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=400, n_init=20, random_state=0)
    kmeans.fit(knowledge)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In the elbow method, the optimal number of clusters is chosen as the point beyond which the rate of decrease of the within clusters sum of squares starts to fall significantly. In some cases, we need not use the elbow method if we are certain about the number of clusters required. For example, in this case, suppose that we wanted to form 3 clusters of student's knowledge to be able to classify them in three different groups and potentially use different strategies to help them better their knowledge.

# In[ ]:


# K-Means Clustering with 3 clusters
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=400, n_init=20, random_state=0)
kmeans.fit(knowledge)
k_class = kmeans.predict(knowledge)


# In[ ]:


# Using PCA and filtering 3 principal components for data visualization
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(knowledge)
PDF = pd.DataFrame(data = principalComponents, columns = ['PC1', 'PC2', 'PC3'])


# In[ ]:


# Add a column 'Class' to the data sets
PDF.loc[:, 'Cluster'] = pd.Series(k_class)
knowledge_class = knowledge.copy()
knowledge_class['Class'] = k_class


# In[ ]:


# Count of points in each cluster
PDF['Cluster'].value_counts()


# In[ ]:


# Assign a color to each cluster
PDF['Color'] = PDF['Cluster'].map({0 : 'red', 1 : 'blue', 2 : 'green'})


# In[ ]:


# Plot the first 2 principal components and color by cluster
a1 = PDF['PC1']
a2 = PDF['PC2']
a3 = PDF['PC3']
c1 = PDF['Color']
plt.scatter(a1, a2, c = c1, alpha=0.3, cmap='viridis')


# In[ ]:


# 3-D plot of the data using 3 principal components
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(a1, a2, a3, alpha = 0.4, c = c1)


# Let us look at how to the 3 classes differ by calculating their averages on each column.

# In[ ]:


knowledge_class.groupby(['Class']).mean()


# Next, we want to perform classification on unseen data and the new categorical target values of class. We can use multiclass classification methods in Machine Learning on this data. The data appears to be well separated in space as seen from the plots. First we will split the data into training and test sets. Then, we will train the Machine Learning models on the trainnig data and evaluate their performance on the test data. There are numerous ways to evaluate performance of the model. Here, we will use the most simple metric, accuracy to evaluate our models. 
# 
# The algorithms to be used for this multi-class classification task and the reason why they were selected from the list of all algorithms are stated below:
# * KNN (K-Nearest Neighbors) - KNN uses distance as the metric and the labels for the dataset were also obtained using distance as the metric when we applied K-Means Clustering. Thus, KNN may perform well on this dataset.
# * Decision Tree Classifier - We almost always want to apply a few Machine Learning methods to any dataset and compare them based on a suitable evaluation metric rather than selecting one final model based only on intusion. Although decision tess may not perform best on a small data such as this one, they are highly interpretable.
# * Naive Bayes - Based on assumption that variables are independent and making a probabilistic estimation using  amaximum likelihood hypothesis, this algorithm is highly efficient as compared to other Machine Lerning models.

# In[ ]:


# Slipt the data into train and test data sets
X = knowledge_class.iloc[:, :-1]
Y = knowledge_class.iloc[:, -1]
xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size = 0.25, random_state = 0)


# In[ ]:


# KNN for various values of k and plot of k v/s accuracy
from sklearn.neighbors import KNeighborsClassifier
accuracy = []
for i in range(1,12):
    knn = KNeighborsClassifier(n_neighbors = i).fit(xTrain, yTrain)
    accuracy.append(knn.score(xTest, yTest))

plt.plot(range(1,12), accuracy)
plt.xlabel('k')
plt.ylabel('Accuracy') 
plt.title('k v/s Accuracy for KNN')


# In[ ]:


# KNN model and evaluation for optimal value of k (8 in this case)
knn = KNeighborsClassifier(n_neighbors = accuracy.index(max(accuracy))+1).fit(xTrain, yTrain)
knn_predictions = knn.predict(xTest)
knn_accuracy = knn.score(xTest, yTest)
knn_accuracy


# In[ ]:


knn_CM = confusion_matrix(yTest, knn_predictions) # KNN Confusion Matrix
knn_CM


# In[ ]:


# Decision Tree Classifier and evaluation for optimal value of k
from sklearn.tree import DecisionTreeClassifier
dtree_model = DecisionTreeClassifier(max_depth = 2).fit(xTrain, yTrain) 
dtree_predictions = dtree_model.predict(xTest)
dt_accuracy = dtree_model.score(xTest, yTest)
dt_accuracy


# In[ ]:


DT_CM = confusion_matrix(yTest, dtree_predictions) # Decision Tree confusion Matrix
DT_CM


# In[ ]:


# Gaussian Naive Bayes model and evaluation for optimal value of k
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB().fit(xTrain, yTrain)
gnb_predictions = gnb.predict(xTest)
gnb_accuracy = gnb.score(xTest, yTest)
gnb_accuracy


# In[ ]:


NB_CM = confusion_matrix(yTest, gnb_predictions) # Naive Bayes confusion Matrix
NB_CM


# We conclude  that the Naive Bayes classifier performed better than KNN and Decision Tree classifier based on the results of accuracy as can be verified by comparing the confusion matrices.
