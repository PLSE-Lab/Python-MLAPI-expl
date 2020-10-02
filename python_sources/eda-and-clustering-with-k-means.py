#!/usr/bin/env python
# coding: utf-8

# ## EDA and Clustering with K-Means
# 
# We will start by importing the required libraries

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import missingno as msno
from sklearn.preprocessing import LabelEncoder
get_ipython().run_line_magic('matplotlib', 'inline')

import os
print(os.listdir("../input"))


# ### Loading the Dataset
# Using pandas we load the dataset and print out its info to get somewhat of an overview of it.
# 

# In[ ]:


dataset = pd.read_csv('../input/Mall_Customers.csv')
dataset.head(15)


# In[ ]:


dataset.info()


# ### Checking for Missing Data
# Missing data can damnage our model's performance, thats why it is necessary to check it. The following visualization is done with the `missingno` lubrary which puts white lines on the grey bars if there is any missing data. As we can see, there is none, so we can move on with our analysis.

# In[ ]:


msno.matrix(dataset)


# ### Encoding Categorical Data
# As you may have noticed, the column gender is categorical with two categories. Obviously a text label like 'Male' or 'Female' is not going to work with our algorithms. We ue `LabelEncoder` from sklearn to achieve this. It encode Male to 1 and Female to 0.

# In[ ]:


label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(dataset.iloc[:,1].values)
dataset['Gender'] = integer_encoded
dataset.head(20)


# ### Visualizing the Dataset
# Now, lets draw the pair wise scatter plots for all the features.

# In[ ]:


sns.pairplot(dataset.iloc[:,1:5])


# There can be some conslusions from the data:-
# * In our data, we can observe that middle-aged people earn the most and young to middle-aged people spend the most. Thus they are a good target to focus new products and advertisement.
# * Spending and earning are more or less uniform for both the genders, so gender neutral products will probably have a good market.
# * Spending Score and Annual income has clear clusters, thus they are good features which can be considered for clustering the customers

# #### Heatmap of the Dataset
# It can help us understand the correlations of the columns better

# In[ ]:


hm=sns.heatmap(dataset.iloc[:,1:5].corr(), annot = True, linewidths=.5, cmap='Blues')
hm.set_title(label='Heatmap of dataset', fontsize=20)
hm


# ### Segmentation 1: Using all the Features

# In[ ]:


dataset_1 = dataset.iloc[:,1:5]
dataset_1.head(10)


# ### The Model: K-Means
# To get started with K-Means we will first have to figure out a good value for K and to do that we use the Elbow curve. The curve has been drawn considering the value of K from 1 to 20

# In[ ]:


results = []
for i in range(1,10):
    kmeans = KMeans(n_clusters=i, init='k-means++')
    res = kmeans.fit(dataset_1)
    results.append(res.score(dataset_1))
plt.plot(range(1,10),results)
plt.xlabel('Num Clusters')
plt.ylabel('score')
plt.title('Elbow Curve')


# Seeing the Elbow Curve there is not a point with a clear elbow, so our clustering may not be mostly accurate using these features. Still, K=3 may give us the best possible result, considering all the features would help us to find the target audiences for gender specific or age specific products.

# ### Fitting the Model

# In[ ]:


model = KMeans(n_clusters= 3, init='k-means++')
clusters = model.fit_predict(dataset_1)
clusters


# In[ ]:


dataset_1['Cluster'] = clusters
sns.pairplot(dataset_1)


# From the above visualization we have the following clusters
# * People with low income, but they spend across the range and they also have no specific age. As they can afford less, they are a good target for low cost products and discounts.
# * Middle aged people with high annual income and high spending score. This is the group that we want to target with most advertisement and product offerings. It also suggests that products for middle aged people will be a good idea.
# * People who earn a lot but spend little. They are not the audience we want to be targeting the most, as they don't buy products even if they can afford them
# 
# Only middle aged people earn a lot and spend a lot

# ### Segmentation 2: Using only Annual Income and Spending Score
# This gives us a good idea what we want to do with products and offers which are gender and age neutral. We use the same analysis and model that we used in the first case.

# In[ ]:


dataset_2 = dataset.iloc[:,3:5]
dataset_2.head(10)


# In[ ]:


results = []
for i in range(1,10):
    kmeans = KMeans(n_clusters=i, init='k-means++')
    res = kmeans.fit(dataset_2)
    results.append(res.score(dataset_2))
plt.plot(range(1,10),results)
plt.xlabel('Num Clusters')
plt.ylabel('score')
plt.title('Elbow Curve')


# We notice that 5 will give us the best clustering

# In[ ]:


model = KMeans(n_clusters=5, init='k-means++')
clusters = model.fit_predict(dataset_2)
clusters


# In[ ]:


segments = pd.DataFrame()
segments['Annual Income'] = dataset['Annual Income (k$)']
segments['Spending Score'] = dataset['Spending Score (1-100)']
segments['Cluster'] = clusters
sns.pairplot(segments)


# In[ ]:


cluster_1 = segments[clusters==0].iloc[:,0:2]
cluster_2 = segments[clusters==1].iloc[:,0:2]
cluster_3 = segments[clusters==2].iloc[:,0:2]
cluster_4 = segments[clusters==3].iloc[:,0:2]
cluster_5 = segments[clusters==4].iloc[:,0:2]

plt.scatter(cluster_1.iloc[:,0],cluster_1.iloc[:,1],c='red',label='Cluster 1')
plt.scatter(cluster_2.iloc[:,0],cluster_2.iloc[:,1],c='blue',label='Cluster 2')
plt.scatter(cluster_3.iloc[:,0],cluster_3.iloc[:,1],c='magenta',label='Cluster 3')
plt.scatter(cluster_4.iloc[:,0],cluster_4.iloc[:,1],c='black',label='Cluster 4')
plt.scatter(cluster_5.iloc[:,0],cluster_5.iloc[:,1],c='green',label='Cluster 5')

plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('Customer Segments')
plt.legend()


# From the above visualizations we have the following clusters:
# * People with low earning and low spending score, they are not a big target audience for us. But we can increasse their spending score by offering low price products and attractive discounts.
# * People with high annual icnome and high spending score, they are the target that we want to focus most on, as they can afford costly products and they are interested in buying from us.
# * People we should least focus on as they don't buy from us even if they can afford it.
# * People with average income and average spending score. They need to be retained, as they dont spend a huge ammount they would probably be attracted a lot by discounts and offers catered to them. Also this is the largest segment of the customers, so we should concentrate to increase their retentivity and spending score.
# * People with low income but with high spending score, in our dataset, they are a minority, but they would like low price products as they can afford them and it is clear that if they spend so much in our mall even with low income, they are likely to come back to us. The advertisement focus for them should be less but should alert them of products which are low priced and provide good value.

# In[ ]:




