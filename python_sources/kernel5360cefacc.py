#!/usr/bin/env python
# coding: utf-8

# <H2>Customer segmentation with K-MEANS</H2>
# <p>in this notebook we are working on customer segmentation on the database of a mall to classify their customers into groups getting insights that can help the marketing team to come up with plan wisely.</p>

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans


# In[ ]:


df = pd.read_csv('../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')


# In[ ]:


df.head()


# In[ ]:


df.describe()


# In[ ]:


df.drop(columns='CustomerID', axis=1, inplace=True)


# In[ ]:


df.isna().sum()


# In[ ]:


df.rename(columns={"Gender":"gender","Age":"age","Annual Income (k$)":"income","Spending Score (1-100)":"score"},inplace=True)


# In[ ]:


plt.figure(figsize=(7,4))
sns.set(style="whitegrid")
sns.boxplot(x='gender',y='score', data=df)
plt.title("GENDER X SCORE")


# In[ ]:


plt.figure(figsize=(7,4))
sns.regplot(x='age',y='score', data=df)
plt.title("Age X SCORE")


# In[ ]:


plt.figure(figsize=(7,4))
sns.regplot(x='income',y='score', data=df)
plt.title("Income X SCORE")


# In[ ]:


df['gender'].value_counts()


# In[ ]:


df.corr()


# <h2>Label Encoder and Data Normalization</h2>
# <p>As gender is a categorical feature and may be of high value for our algorithm we need to use label encodering to transform the column into numerical categories.</br>
# Lets also apply data normalization to our dataframe as the distribuition is large in our features.
# </p>
# 

# In[ ]:


le = LabelEncoder()
le.fit(df['gender'])
df['gender'] = le.transform(df['gender'])


# In[ ]:


x = df[['age','gender','income','score']]
scaler = StandardScaler().fit(x)
x = scaler.transform(x)


# <H2>How Many clusters should we have?</H2>
# <p>Here we use the  elbow techinique to find out how many clusters we should have before creating our machine learning object and passing in the parameter number of clusters. It's clear that 4 is the right number.</p>

# In[ ]:


errors = []
for i in range(1,10):
    k = KMeans(n_clusters=i, init='k-means++')
    k.fit(x)
    errors.append(k.inertia_)


# In[ ]:


plt.plot(range(1,10),errors)
plt.annotate('N',xy=(3.8,350))


# In[ ]:


kmeans = KMeans(n_clusters=4, init='k-means++')
kmeans.fit(x)


# In[ ]:


#get the labels and centroids from machine learning algorithm K-MEANS
labels = kmeans.labels_
centroids = kmeans.cluster_centers_


# In[ ]:


#Apply the labels to the dataframe
df['cluster'] = labels


# <h2>Score Distribuition  by clusters</h2>
# <p>It's noticeable through this boxplot that the clusters 1 and 2 have a distribuition superior than the others, we can also see that cluster 2 has a minimum higher, both have a medium around the same point and 75% quartile of cluster 2 is from 70 to 90 points while cluster 1 is 70 to 80. They are quite similiar in some aspects but its obvius at this point that customers belonging to clusters 0 and 3 are not our target here.</p>

# In[ ]:


sns.boxplot(x='cluster',y='score',data=df)


# <h2>Data Visualization</h2>
# <p>Now that we know that both clusters 1 and 3 differ significantly from the the other clusters on customer spending score, lets visualize the age and income of the customers belonging to those clusters.</p>

# In[ ]:


colors = ['slateblue','firebrick','lightsteelblue','tomato']
clusters = df['cluster'].unique()
plt.figure(figsize=(10,8))
for i, col in zip(range(len(centroids)),colors):
    members = df['cluster'] == i
    centroid = centroids[i]
    plt.scatter(df[members]['age'],df[members]['score'],c = col, label='Cluster{}'.format(i))
plt.ylabel('Score', fontsize='large')
plt.xlabel('Age', fontsize='large')
plt.title('Cluster Age X Score', fontsize='x-large', y=1.020)
plt.legend(loc='upper right', fontsize='large')    


# In[ ]:


colors = ['slateblue','firebrick','lightsteelblue','tomato']
clusters = df['cluster'].unique()
plt.figure(figsize=(10,8))
for i, col in zip(range(len(centroids)),colors):
    members = df['cluster'] == i
    centroid = centroids[i]
    plt.scatter(df[members]['income'],df[members]['score'],c = col, label='Cluster{}'.format(i))
plt.ylabel('Score', fontsize='large')
plt.xlabel('Anual income', fontsize='large')
plt.title('Cluster Income X Score', fontsize='x-large', y=1.020)
plt.legend(loc='upper right', fontsize='large')    


# <h2>Analysis on the cluster 1</h2>
# <p>After the visual and mathematical analysis on the cluster 3 we have defined its members to have:</p>
# <ul>
#   <li>Age Between 18 to 40</li>
#   <li>Income Between 40k to 120k</li>
#   <li>Gender Male</li>
# </ul>
# <p>We have labeled them as <b>Male Young High Income customers</b></p>

# In[ ]:


df[df['cluster'] == 1].describe()


# <h2>Analysis on the cluster 2</h2>
# <p>After the visual and mathematical analysis on the cluster 1 we have defined its members to have:</p>
# <ul>
#   <li>Age Between 20 to 33</li>
#   <li>Income Between 41k to 137k</li>
#   <li>Gender Female</li>
# </ul>
# <p>We have labeled them as <b>Female Young High Income customers</b></p>

# In[ ]:


df[df['cluster'] == 2].describe()


# <h1>Conclusions</h1>
# <p>Through this segmentation the algorithm we could classify our customers into different groups and target the right type of customer who is more likely to spend money on our mall. Now the marketing team can go ahead and create a strategy to draw in customers aged between 20-40 with income around 40k-130k both males or females.</p>
# 
# <b>Notebook created by Christian Fuin - 31/05/2020</b>
