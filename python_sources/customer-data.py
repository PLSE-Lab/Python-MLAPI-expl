#!/usr/bin/env python
# coding: utf-8

# **Clustering Customer Data**
# 
# This kernel is built off of Sowmya Vivek's article "Clustering algorithms for customer segmentation". The same two-variable dataset was used that compared customer income with customer spend.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import matplotlib.pyplot as plt

import seaborn as sns; sns.set() #for plot styling
get_ipython().run_line_magic('matplotlib', 'inline')

plt.rcParams['figure.figsize'] = (16,9)
plt.style.use('ggplot')

dataset = pd.read_csv('../input/clustering-customer-data/CLV.csv')
dataset.head()
##len(dataset)
dataset.describe().transpose()


# **Data Visualization**
# 
# First we take a look at the distribution of values within the INCOME and SPEND variables.

# In[ ]:


plot_income = sns.distplot(dataset["INCOME"])
plot_spend = sns.distplot(dataset["SPEND"])
plt.xlabel('Income/Spend')


# In the original article, a violin plot was used next. However, I felt that it would be more beneficial to visualize a scatterplot. When we look at the scatterplot, we see a wide range of SPEND and INCOME values. At first glance, it would be difficult to come to any conclusions from this graph.

# In[ ]:


scatter_income = sns.scatterplot(x = "INCOME", y ="SPEND", data=dataset)


# **K-Means**
# 
# K-Means clustering can be used to create customer segments. From Vivek:
# 
# *"The objective of any clustering algorithm is to ensure that the distance between datapoints in a cluster is very low compared to the distance between 2 clusters. In other words, members of a group are very similar, and members of different groups are extremely dissimilar."*
# 
# In order to find a good estimate for the number of clusters to use,the concept of minimizing within cluster sum of square was used. This means the find the minimal distance between a central point and surrounding data points.

# In[ ]:


from sklearn.cluster import KMeans
wcss = []
for i in range (1,11): # this refers to 1 - 11 possible clusters
    km = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    km.fit(dataset)
    wcss.append(km.inertia_)
plt.plot(range(1,11), wcss)
plt.xlabel('Number of Clusters')
plt.ylabel('wcss')
plt.show()


# **Clusters**
# 
# As per the the elbow method, the number of clusters would be 4. However, for the sake of practice, we changed it to 6 to view the differences.
# Referencing Vivek's code, I ran into an issue plotting the different clusters. Rather than plotting four separate scatters, we could do the same work using just one line of code by using X.iloc[:, 0], X.iloc[:, 1] instead.

# In[ ]:


km6 = KMeans(n_clusters = 6, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
X = dataset
y_means = km6.fit_predict(X)
X.head()

plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y_means, s=50, cmap='viridis')
plt.scatter(km6.cluster_centers_[:,0], km6.cluster_centers_[:,1],s=200,marker='s', c='red', alpha=0.7, label='Customer Group')

plt.title('Customer segments')
plt.xlabel('Annual income of customer')
plt.ylabel('Annual spend from customer on site')
plt.legend()
plt.show()


# **Discussion**
# 
# Examining the scatterplot, we see 6 customer segments. They could be described as:
# 
#     i) Low Income/Low Spend
#     
#     ii) Low - Medium Income/High Spend
#     
#     iii) Medium Income/High Spend
#     
#     iv) Medium Income/Low Spend
#     
#     v) Medium - High Income/High Spend
#     
#     vi) High Income/High Spend
#     
# These segements could be described better, but they will do for now.
# 
# In terms of a sales aspect, a question that could arise from this graph is how do we increase the sales of Group 4 (Medium Income/Low Spend)? 
