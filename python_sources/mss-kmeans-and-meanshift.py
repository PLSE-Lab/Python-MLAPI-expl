#!/usr/bin/env python
# coding: utf-8

# ### Mall Segmentation Study
# This purpose of this study is to explore the given data and use clustering techniques for further analysis. 
# 
# 

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans,MeanShift,estimate_bandwidth
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('../input/Mall_Customers.csv')
list(df)

pd.options.mode.chained_assignment = None 


# In[ ]:


df.describe()


# In[ ]:


plt.rcParams['figure.figsize'] = (10,8)
sb.jointplot(x='Annual Income (k$)',y='Age',data=df,kind="kde")


# In[ ]:


sb.scatterplot(x='Age',y='Spending Score (1-100)',data=df)


# It appears mall-goers below the average age have the highest spending scores and older folks' scores are ~40 points lower. 

# In[ ]:


F = df[df.Gender == 'Female']
np.mean(F)


# In[ ]:


M = df[df.Gender == 'Male']
np.mean(M)


# In[ ]:


sb.scatterplot(x='Annual Income (k$)',y='Spending Score (1-100)',data=df)


# The spending score groups are distinct when plotting against annual income. We will use KMeans to assign each cluster to a group based on annual income.  

# In[ ]:


features = ['Annual Income (k$)','Spending Score (1-100)']
X = df[features]

kmeans = KMeans(n_clusters=5,random_state=13).fit(X)
centers = kmeans.cluster_centers_
labels = kmeans.labels_

X['labels'] = labels


# In[ ]:


sb.scatterplot(x='Annual Income (k$)',y='Spending Score (1-100)',data = X,hue='labels',palette="Set1",legend="full")


# There's a couple observations that appear to be mislabeled between groups 0 and 2. All other observations are labeled accordingly. 

# In[ ]:


features = ['Age','Spending Score (1-100)']
X = df[features]

bandwidth = estimate_bandwidth(X,quantile=0.20)
meanshift = MeanShift(bandwidth=bandwidth,bin_seeding=True)
meanshift.fit(X)

labels = meanshift.labels_

X['labels'] = labels


# By playing with the quantile value you can can increase or decrease the number of labels. Setting the quantile value to 0.20 yielded an interpretable grouping. 

# In[ ]:


sb.scatterplot(x='Age',y='Spending Score (1-100)',data = X,hue='labels',palette="Set1")


# We end up with an interesting grouping using MeanShift clustering that shows a clear seperation between spending scores. It makes sense that we see a dense cluster around 50 since it is the mean spending score. 

# 
