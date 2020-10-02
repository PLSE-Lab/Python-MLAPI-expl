#!/usr/bin/env python
# coding: utf-8

# In[43]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
warnings.filterwarnings("ignore")
import os
print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.


# **Problem statement:** Analyse the data and draw inferences from the shopping behavior of a group of customers. Also, to write an prediction module to predict what kind of shopping habits would a given customer have.

# **Loading and Describing the data**
# We should first load the data, and describe the data statistically. This will help us understand the data and the data types.

# In[44]:


data = pd.read_csv("../input/Mall_Customers.csv")

pd.set_option('display.max_columns', 10)

print("Data Sample\n{}\n".format(data.head()))
print("Data description\n{}\n".format(data.describe()))
print("Data types\n{}\n". format(data.dtypes))


# In[45]:


# getting the list of columns

columns = data.columns

print(columns)


# **Data Visualisation**
# 
# Now that we have looked at the data, it's time we visualize the data, to view the correlation of the attributes with other attributes. This way we can analyse the data.

# In[46]:


import matplotlib.pyplot as plt
import seaborn as sns

n=0

plt.figure(1, figsize = (25,6))

for i in ['Age', 'Annual Income (k$)','Spending Score (1-100)']:
    n = n+1
    plt.subplot(1,3,n)
    plt.subplots_adjust(hspace = 0.5, wspace = 0.5)
    sns.distplot(data[i], bins=20)
    plt.title("Distplot of {}".format(i))
plt.show()


# In[47]:


# plot of Male count Vs Female count
plt.figure(1, figsize = (10,5))
sns.countplot(y = 'Gender', data = data)
plt.show()


# In[48]:


# plotting [Age, income, spending score] with one another

plt.figure(1, figsize = (25,25))
n = 0

for x in ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']:
    for y in ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']:
        if( x != y ):
            n = n + 1
            plt.subplot(3, 3, n)
            plt.subplots_adjust(hspace = 0.5, wspace = 0.5)
            sns.regplot(x = x, y = y, data = data)
            plt.title("Plot of {} vs {}".format(x,y))

plt.show()


# In[49]:


# plotting Age vs Annual income wrt Gender

plt.figure(1, figsize = (25,25) )
n = 0

for x in ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']:
    for y in ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']:
        if( x != y ):
            n = n + 1
            plt.subplot(3, 3, n)
            plt.subplots_adjust(hspace = 0.5, wspace = 0.5)
            for gender in ['Male', 'Female']:
                sns.regplot(x = x, y = y, data = data[ data['Gender'] == gender ], label = gender)
                plt.title("{} vs {} wrt Gender".format(x,y))
plt.legend()
plt.show()


# From the above visualisations, the following inferences can be drawn.
# 
# 1. The cusotmers can be clustered on the basis of Annual income and spending score, wrt gender.
# 2. Gender probably doesn't have much effect on clustering customers based on Annual Income and spending score
# 3. Spending score drops significantly more with increasing Age in FEMALES than in MALES
# 4. Trends in Spending score vs Annual income vs Age wrt gender can be studied.

# In[50]:


# applying Elbow method to calculate the optimum K
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

attr = ['Annual Income (k$)', 'Spending Score (1-100)']

K = range(1,10)
distorts = []
for k in K:
    kmeans = KMeans(k)
    kmeans.fit(data[attr])
    distorts.append(sum(np.min(cdist(data[attr], kmeans.cluster_centers_, 'euclidean'), axis=1)) / data.shape[0])

plt.plot(K, distorts)
plt.xlabel('K')
plt.ylabel('Distortions')
plt.title("Plot for K vs Distortions")


# In[51]:


# Applying K means clustering on Annual Income and Spending score

attr = ['Annual Income (k$)', 'Spending Score (1-100)']

# From the result of the elbow algorithm, the elbow occurs at K = 5

N = 5

kmeans = KMeans(n_clusters = 5 ,init='k-means++', n_init = 10 ,max_iter=300,
                tol=0.0001,  random_state= 111  , algorithm='elkan')

kmeans.fit(data[attr])

centroids = kmeans.cluster_centers_

print("The centroids are:{}".format(centroids))

cluster = kmeans.fit_predict(data[attr])

data['cluster'] = cluster


# In[52]:


# visualising the clusters and the centroids.


for c in range(0,5):
    plt.scatter(x = data[data['cluster'] == c]['Annual Income (k$)'], 
               y = data[data['cluster'] == c]['Spending Score (1-100)'], label = 'Cluster{}'.format(c))

for c in range(0,5):
    plt.scatter(x = centroids[c][0], 
               y = centroids[c][1],s = 300, label = 'Centroid{}'.format(c))
    
plt.title("Cluster plot of Annual income vs Spending Score")
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()


# We can hence cluster a set of customers into 5 groups based on their income and shopping behavior.
# 
# The above plot visualisation produces the following information
# 
# 1. The data is clustered into 5 types of customer:
#     1. High spending score & low income
#     2. High spending score & high income
#     3. Medium spending score & medium income
#     4. low spending score & low income
#     5. low spending score & high income
# 2. The customers within the "High income" group should be targetted for serious advertising.
# 3. The customers within the "high spending score and low income" should be analysed wrt Gender and Age, to understand the expenditure trends of the "low income" group of customers.

# In[53]:


# count plot of all the clusters

plt.figure(1, figsize = (10,6))
sns.countplot(y = 'cluster', data = data)
plt.title('Count plot of cusomters wrt clusters')
plt.show()


# In[54]:


# Analysis of the customers belonging to "High spending score and low income" cluster, aka cluster1

data_c0 = data[data['cluster'] == 0]
data_c0.head(10)


# In[55]:



data_c1 = data[data['cluster'] == 1]
data_c1.head(10)


# In[56]:


# analysing the age distribution of cluster 0 data wrt gender

plt.figure(1, figsize = (10,6))
for gender in ['Male', 'Female']:
    sns.distplot(data_c0[data_c0['Gender'] == gender]['Age'], hist=False, rug = True, label=gender)
plt.title('Age distribution for Cluster 0 cusomters wrt Gender')
plt.legend()
plt.show()


# In[57]:


# Visualizing Spending score distribution of cluster 0 customers wrt gender

plt.figure(1, figsize = (10,6))
for gender in ['Male', 'Female']:
    sns.distplot(data_c0[data_c0['Gender'] == gender]['Spending Score (1-100)'], hist=False, rug = True, label=gender)
plt.title('Spending Score distribution for Cluster 0 cusomters wrt Gender')
plt.legend()
plt.show()


# In[58]:


# analysing the age distribution of cluster 1 data wrt gender

plt.figure(1, figsize = (10,6))
for gender in ['Male', 'Female']:
    sns.distplot(data_c1[data_c1['Gender'] == gender]['Age'], hist=False, rug = True, label=gender)
plt.title('Age distribution for Cluster 0 cusomters wrt Gender')
plt.legend()
plt.show()


# **Conclusion**
# 
# We draw the following set of conclusions from the notebook:
# 1. The customers within cluster 0 has more FEMALE spenders than MALE spenders within the range of 18-25.
# 2. However the spending score of both the gender groups are more or less similar for cluster 0 and cluster 1.
# 3. The most valuable customers are those with "high income and high spending score"
# 4. Also, the customers belonging to "medium income medium spending score" are significantly high in number, meaning that the revenue generated from them would be significatly high
# 
# Hence, we can conclude, that the most valuable customers for the mall are those with "medium income medium spending" and "high income high spending".
