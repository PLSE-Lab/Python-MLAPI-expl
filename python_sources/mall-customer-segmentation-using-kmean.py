#!/usr/bin/env python
# coding: utf-8

# # Who spends most at the Mall?
# 
# ## Mall Customer analysis using K-Mean clustering

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans 


# In[ ]:


# Load the dataset 
data=pd.read_csv('../input/Mall_Customers.csv')
data.head()


# ## Data pre-processing

# In[ ]:


# check for NULLs
data.isnull().sum()


# In[ ]:


# Drop customer ID (Categorical variable)
df = data.drop('CustomerID', axis=1)
df.head()


# In[ ]:


df = df.drop_duplicates()
df.count()


# ## Exploratory Data Analysis

# In[ ]:


# Overall summary
df.describe()


# In[ ]:


df.describe().transpose()


# In[ ]:


fig = plt.figure(figsize=(15,10))

plt.subplot(2, 2, 1)
sns.set(style = 'whitegrid')
sns.distplot(df['Annual Income (k$)'], color = 'green',kde=False)

plt.title('Distribution of Annual Income', fontsize = 20)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Count')

plt.subplot(2, 2, 2)
sns.set(style = 'whitegrid')
sns.distplot(df['Age'], color = 'red',kde=False) # Turned off KDE 
plt.title('Distribution of Age', fontsize = 20)
plt.xlabel('Age (years)')
plt.ylabel('Count')


plt.subplot(2, 2, 3)
labels = ['Men', 'Women']
percentages = df['Gender'].value_counts(normalize=True) * 100

explode=(0.1,0)
plt.pie(percentages, explode=explode, labels=labels,  
       autopct='%1.0f%%', 
       shadow=False, startangle=0,   
       pctdistance=1.2,labeldistance=1.4)
plt.axis('equal')
#plt.title("Gender Ratios")
plt.legend(frameon=False, bbox_to_anchor=(1.5,0.8))

plt.tight_layout()


# Above plots shows that in this data sample:
#  
# 1. Most shoppers have an annual income < ~ $90K
# 2. Most shoppers are under the age ~ 50
# 3. Most of the shoppers are (~60%) are male.

# ### Pairwise relationships
# 
# Provides an overview of which variables will play a role in clustering

# In[ ]:


fig = plt.figure(figsize=(15,15))
sns.set(style = 'whitegrid')
#sns.set(style="ticks")

sns.pairplot(df, hue="Gender", palette="Set2")


# Above set of pariwise plots indicate that in this perticular sample of mall customers there are no significant correlations between the Age, Income and the Spending score. Also the gender doesn't seem to play a notable role in the individual features.
# 
# However, there seems to be a noticable clustering associated with the annual income and the spending score. Perhaps this will get more clear in the clustering analysis.

# ## Modeling
# ## Use K-Mean Clustering to Indentify Target Shoppers

# ### Normalize data over the standard deviation
# Normalization helps to interpret features with different magnitudes and distributions equally

# In[ ]:


from sklearn.preprocessing import StandardScaler
X = df.values[:,1:]
X = np.nan_to_num(X)

Clus_dataSet = StandardScaler().fit_transform(X)
Clus_dataSet 


# ### 1. Segmentation using Annual Income, Age and Spending Score****

# #### optimum number of clusters for K-means
# 
# For each k value, initialise k-means and use the inertia attribute to identify the sum of squared distances of samples to the nearest cluster centre.

# In[ ]:


sum_of_squared_distances = []
K = range(1,9)
for k in K:
    km = KMeans(n_clusters=k, init = 'k-means++', 
                max_iter = 300, n_init = 10, random_state = 0)
    km = km.fit(Clus_dataSet)
    sum_of_squared_distances.append(km.inertia_)
    
plt.plot(K, sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('sum of squared distances')
plt.title('Elbow Method')
plt.show()


# In the above plot, the curve is too smooth to identify an elbow. This could either be because there are too many features in the data set interfering with the KMean calculations or we need to find a better method to determine the optimal K or use a different clustering method such as DBSCAN. 

# Lets try segmentation using just two features.
# 
# ### Segmentation using Age and Spending Score

# In[ ]:


Clus_dataSet1 = pd.DataFrame(Clus_dataSet[:])
Clus_dataSet1 = Clus_dataSet1[[0,2]]

sum_of_squared_distances = []
K = range(1,9)
for k in K:
    km = KMeans(n_clusters=k, init = 'k-means++', 
                max_iter = 300, n_init = 5, random_state = 0)
    km = km.fit(Clus_dataSet1)
    sum_of_squared_distances.append(km.inertia_)
    
plt.plot(K, sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('sum of squared distances')
plt.title('Elbow Method for Optimum K')
plt.show()


# Unable to find optimum K here too because the curve is too smooth.

# ### Segmentation using Annual Income and Spending Score

# In[ ]:


Clus_dataSet2 = pd.DataFrame(Clus_dataSet[:])
Clus_dataSet2 = Clus_dataSet2[[1,2]]

sum_of_squared_distances = []
K = range(1,9)
for k in K:
    km = KMeans(n_clusters=k, init = 'k-means++', 
                max_iter = 300, n_init = 5, random_state = 0)
    km = km.fit(Clus_dataSet2)
    sum_of_squared_distances.append(km.inertia_)
    
plt.plot(K, sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('sum of squared distances')
plt.title('Elbow Method for Optimum K')
plt.show()


# Here we see a clear elbow at K = 5. So that is our optimum K.
# Use K = 5 to find customer clusters.

# In[ ]:


# apply k-means on the dataset
from sklearn.cluster import KMeans 
clusterNum = 5
k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
k_means.fit(X)
labels = k_means.labels_
print(labels)


# ## Insights

# ### Asign cluster labels to the data

# In[ ]:


# assign the labels to each row in dataframe.
df["cluster"] = labels
df.head(5)


# ### Check the cluster centeroids

# In[ ]:


# We can easily check the centroid values by averaging the features in each cluster.
df.groupby('cluster').mean()


# ### Customer segmentation using Annual Income and Spending scores

# In[ ]:


# look at the distribution of customers based on their age and income:

# Create plot
fig = plt.figure()

sns.lmplot( x="Annual Income (k$)", y="Spending Score (1-100)", 
           data=df, fit_reg=False, hue='cluster', legend=False)
plt.legend(loc='upper right')
labels = ['Average shoppers', 'Budget shoppers', 
          'Under spending shoppers','Over spending shoppers',
          'High spending shoppers']
#plt.legend(labels)
plt.legend(labels, loc='center right', 
           bbox_to_anchor=(1.75, 0.5), ncol=1)

#ax.scatter(X[:, 1], X[:, 2], c=labels, alpha=0.5)
plt.xlabel('Annual Income', fontsize=18)
plt.ylabel('Spending Score (1-100)', fontsize=16)
plt.title('K-Mean Clusters')
plt.show()


# Above scatter plot shows a clear segmentation of mall customers based on their annual income and the spending scores.
# 

# In[ ]:


sns.pairplot(df[df['cluster'] == 2], hue="Gender", palette="Set2")


# Based on the above pair plots, to increase the spending score of the shoppers in the mall, the mall marketing should cater their efforts towards converting the high income 'under shoppers' who seems to be mostly middle aged (35-50) males with a median income in the range ~60 K
# 
