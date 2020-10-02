#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("../input/winequality.csv")


# Lets analyze wine data..

# In[ ]:


df['color'].replace({'white': 0, 'red': 1},inplace = True)
df.head()


# In[ ]:


df.describe()


# Checking distrubution of data..

# In[ ]:


sns.set()
df.hist(figsize=(10,10), color='green')
plt.show()


# Lets check the feature correlations with heatmap..

# In[ ]:


plt.figure(figsize=(12,12))
plt.title('Feature correlation', size=15)
sns.heatmap(df.astype(float).corr(),vmax=1.0, square=True, annot=True)


# Create a scatter for analysing the data..

# In[ ]:


col = np.where(df.quality>7,'green',np.where(df.quality<5,'red','blue'))
plt.scatter(df['alcohol'] ,df['quality'],color=col,alpha = .5)
plt.show()


# Clustering our wine data..

# In[ ]:


from sklearn.cluster import KMeans
inertia=np.empty(7)
for i in range(1,7):
    kmeans = KMeans(n_clusters=i, random_state=0).fit(df['quality'].values.reshape(-1,1))
    inertia[i] = kmeans.inertia_

plt.plot(range(0,7),inertia,'-o')
plt.xlabel('Number of cluster')
plt.ylabel('Inertia')
plt.show()


# 3 is our best clustering data depends on elbow method...Lets visualize it..

# In[ ]:


kmeans1 = KMeans(n_clusters=3, random_state=0)
clusters = kmeans1.fit_predict(df['quality'].values.reshape(-1,1))
df["cluster"] = clusters
df.head()

plt.scatter(df.quality[df.cluster == 0 ],df.pH[df.cluster == 0 ],color = "red")
plt.scatter(df.quality[df.cluster == 1 ],df.pH[df.cluster == 1 ],color = "green")
plt.scatter(df.quality[df.cluster == 2 ],df.pH[df.cluster == 2 ],color = "blue")
plt.show()


# Lets create a dendrogram for given data..We can check our cluster counts with dendrogram too..

# In[ ]:


from scipy.cluster.hierarchy import linkage,dendrogram

merg = linkage(df,method = 'ward')
dendrogram(merg, leaf_rotation = 90, leaf_font_size = 6)
plt.show()

