#!/usr/bin/env python
# coding: utf-8

# ### Clustering 

# Grouping similar entities together help profile the attributes of different groups. In other words, this will give us insight into underlying patterns of different groups.

# In[ ]:


# Importing the libraries
import pandas as pd
from pandas import DataFrame
import pylab
import calendar
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import warnings
pd.options.mode.chained_assignment = None
from matplotlib.pylab import rcParams
rcParams['figure.figsize']=9,6
warnings.filterwarnings("ignore", category=DeprecationWarning)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


'''import os
print(os.getcwd())
os.chdir('D:\\DS_Notes\\Datasets_new\\')
# Importing the dataset
dataset= pd.read_csv('Mall_Customers.csv')
# displaying the data universities.csv
dataset.head()'''


# In[ ]:


dataset = pd.read_csv("../input/Mall_Customers.csv")
dataset.head()


# From the above data set, observe that the first 3 columns are not necessary for clustering analysis hence we don't conside them. By taking the rest of the values, create a new dataset in a numpy array.

# In[ ]:


X = dataset.iloc[:, [3, 4]].values
# printing first 6 rows of the dataset
X[:6]


# Plot the values using scatter plot and check the graph and have a basic idea of how many clusters can be formed out of it.

# In[ ]:


plt.scatter(dataset['Annual Income (k$)'],dataset['Spending Score (1-100)'],color='blue',s=20)
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(loc='best')
rcParams['figure.figsize']=8,6
plt.show()


# From the above graph we can predict that, we can make a rough idea that we can make 5 clusters out of it. So we shall consider the number of clusters be 5

# In[ ]:


# Fitting K-Means to the dataset 
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 40)
# Compute cluster centers and predict cluster index for each sample.
y_kmeans = kmeans.fit(X)
y_kmeans


# In[ ]:


y_kmeans = kmeans.fit_predict(X)
y_kmeans


# In[ ]:


kmeans.cluster_centers_


# ### let us undersand the parameters used.
# 
# n_clusters : The number of clusters to form as well as the number of centroids to generate (An idea from above plot )
# 
# 
# KMeans : The number of clusters to form as well as the number of centroids to generate.
# 
# random_state = random state '1' means reproducability, if any one uses the same value, they will get the the same output
# 
# n_init = Number of time the k-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia.
# 

# In[ ]:


# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 50, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 50, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 50, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 50, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 50, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 100, c = 'yellow', label =
'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(loc='best')
plt.show()


# In[ ]:


# naming each cluster based on the income and the amount they spend.
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 50, c = 'red', label = 'sensible')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 50, c = 'blue', label = 'standard')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 50, c = 'green', label = 'Target')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 50, c = 'cyan', label = 'careful')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 50, c = 'magenta', label = 'careless')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 100, c = 'yellow', label ='Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()


# In[ ]:


# add a new column and assign the cluster values to it
dataset["clusters"] = kmeans.labels_
dataset.head()


# " kmeans labels "  is a single column so there is no issue in adding the column to the existing dataset

# In[ ]:


centers = pd.DataFrame(kmeans.cluster_centers_)
centers 


# Cluster centers is a dataframe as we have 2 cloumns and 5 rows, also, the existing data set has more than 5 rows. In this case, in order to merge both the datasets, there should be a common column.

# In[ ]:


centers["clusters"] = [0,1,2,3,4]
#centers["clusters"] = range(5) 
centers


# In[ ]:


# performing merge operation as there are two similar columns named " clusters"
dataset = dataset.merge(centers)
dataset.head(10)


# In[ ]:


s1_grps = pd.DataFrame(kmeans.labels_)
s1_grps.cloumns = ('clusters')
#s1_grps.rename(columns={'0': 'cluster'}, inplace=True)
#s1_grps = s1_grps.rename(columns={'0':'Cluster'},inplace=False)
s1_grps.head()


# In[ ]:


s2_univs = dataset.iloc[:,0]
#s2_univs = dataset['CustomerID']
s2_univs.head()


# In[ ]:


rslt = pd.concat([s1_grps,s2_univs],axis=1)
rslt.rename(columns={'0':'Cluster'}, inplace=True)
#rslt['s2_univs'] = rslt.s2_univs.astype(int)
rslt.head(10)
#list(rslt)


# In[ ]:


dataset = pd.read_csv("../input/Mall_Customers.csv")
dataset.head()


# In[ ]:


# Now consider three input values
d = dataset.iloc[:, [2, 3, 4]]
d.head()


# In[ ]:


X = dataset.iloc[:, [2, 3, 4]].values
X


# In[ ]:


# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)
dataset["clusters"] = kmeans.labels_
dataset.head()


# In[ ]:


centers = pd.DataFrame(kmeans.cluster_centers_)
centers


# In[ ]:


centers = pd.DataFrame(kmeans.cluster_centers_)
centers["clusters"] = range(5) #n_clusters
dataset = dataset.merge(centers)
dataset


# In[ ]:




