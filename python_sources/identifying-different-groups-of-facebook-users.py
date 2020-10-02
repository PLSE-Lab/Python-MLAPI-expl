#!/usr/bin/env python
# coding: utf-8

# # Introduction

# Let's say we want to identify certain groups of facebook users based on their behavior. Some facebook users might not react on the posts. Some people may 'like' react more and use other reactions like 'love' or 'wow' to a very minute extent. Some people may share posts a lot while some may not. Some only react and not comment on posts. Some may comment but react much, etc.
# 
# We are going to identify such groups with the help of <b>machine learning</b>.
# 
# **Note**: This dataset is Thailand Facebook Live dataset. I have removed the labels to make it an unsupervised problem to demonstrate KMeans Clustering.

# # Choosing the Machine Learning Algorithm

# We are going to use K-Means Clustering for this purpose. K-Means Clustering is the most popular algorithm which is used for unsupervised learning. This algorithm is used when we have un-labelled data.

# # Exploratory Data Analysis (EDA)

# First, let's import the relevant python libraries.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# I gathered some facebook data from the internet. Let's get this data into a pandas dataframe.

# In[ ]:


import os

os.chdir('/kaggle/')
os.listdir('/kaggle/input')


# In[ ]:


data = pd.read_csv('/kaggle/input/fb_users.csv')
df = data.copy()  # To keep the data as backup

df.head()


# In[ ]:


df.shape


# Let's check to make sure we don't have any null values in the data.

# In[ ]:


df.isnull().sum()


# We have no null values. So, we need not worry about it!

# Let's first look at the basic statistics of the data. Let's see what we can find from it.

# In[ ]:


df.describe()


# Our basic statistics shows that almost all the labels are right skewed. Also, by looking at the min and max values, we can say that the difference of these values is quite great. Hence, we need to scale our data.
# 
# Now, to scale our data, we need to use a great python library 'sklearn' and it's MinMaxScaler() function from sklearn.preprocessing module.

# In[ ]:


from sklearn.preprocessing import MinMaxScaler

# Let's instantiate MinMaxScaler object
scaler = MinMaxScaler()

# Scale
scaled_array = scaler.fit_transform(X=df)

type(scaled_array)


# The output we got after scaling the dataframe is a numpy array. We need to convert it into a dataframe.

# In[ ]:


df_scaled = pd.DataFrame(data=scaled_array, columns=df.columns)
df_scaled.head()


# # Clustering

# As of now, we have our scaled data ready to be trained. But we have one problem: how many clusters (groups) to put into the data to train? Now, to find out the right number of clusters, we need to decide based on two metrics: <b>Elbow Analysis</b> and <b>Silhouette Score</b>.

# ## Elbow Analysis

# In[ ]:


# First, we need to import the sklearn's KMeans library
from sklearn.cluster import KMeans

cost = []

for k in range(1, 16):
    kmean = KMeans(n_clusters=k, random_state=0)
    kmean.fit(df_scaled)
    cost.append([k, kmean.inertia_])

# Plotting the K's against cost using matplotlib library which we imported at the start
plt.figure(figsize=(15, 8))
plt.plot(pd.DataFrame(cost)[0], pd.DataFrame(cost)[1])
plt.xlabel('K')
plt.ylabel('Cost')
plt.title('Elbow Analysis')
plt.grid(True)

plt.show()


# Okay, we can see that on 2 clusters, elbow is being formed. But let's not be hasty in selecting the number 2. Let's also find out the silhouette score to be sure.

# ## Silhoutte Score

# In[ ]:


# Let's first import the relevant library
from sklearn.metrics import silhouette_score

score = []

for k in range(2, 16):
    kmean = KMeans(n_clusters=k, random_state=0)
    kmean.fit(df_scaled)
    score.append([k, silhouette_score(df_scaled, kmean.labels_)])

# Let's plot the Number of K's against Silhouette Score
plt.figure(figsize=(15, 8))
plt.plot(pd.DataFrame(score)[0], pd.DataFrame(score)[1])
plt.xlabel('K')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.title("Silhouette Score against each number of clusters")

plt.show()


# Silhouette score of 3 is pretty high which is a good thing. And by looking at both the elbow analysis and silhouette score. We can decide to choose 3 because the cost of 3 on the Elbow Analysis is quite low and its silhouette score is quite high.
# 
# So, let's choose 3.

# ## Training on the chosen number of K

# In[ ]:


# Let's make a new dataframe named 'predict' and copy the contents of our original df into it
pred = df.copy()

# Let's train the model based on 3 clusters
kmean_3k = KMeans(n_clusters=3, random_state=0)
labels = kmean_3k.fit_predict(df_scaled)

pred['clusters'] = labels
pred.head()


# # Profiling

# Having decided how many clusters to use, we would like to get a better understanding of what values are in those clusters are and interpret them. That's where profiling comes in. We can use profiling for data analysis to get insights. Let's do this to find out the user behavior.

# In[ ]:


pivoted = pred.groupby(['clusters']).median().reset_index()
pivoted


# From the pivoted dataframe shown above, we can see that there are three groups of Facebook users:
# 1. **Group 0**: Which indicates that the user of this group might not use Facebook a lot or use it only for surfing. Their number of reactions are around 48 and comments only 3. They don't share a lot of posts. And mostly they use 'like' react on posts.
# 
# 
# 2. **Group 1**: This is the group of people who, according to the provided dataset, happen to use Facebook quite a lot. But they are the kind of people who usually give people the 'like' react mostly.
# 
# 
# 3. **Group 2**: This group also shows that people use Facebook a lot. These people tend to comment and share the posts a lot. They also tend to use other reacts on posts besides the 'like' react.
# 
# Now, to check the number of people in each group:

# In[ ]:


pred.clusters.value_counts()


# Let's graph this by using python's popular <b>seaborn</b> library.

# In[ ]:


import seaborn as sns
sns.set()

plt.figure(figsize=(10, 6))
sns.countplot(x='clusters', data=pred)


# By looking at the above graph. We can conclude that in the given limited dataset, most people belong to group 0.
