#!/usr/bin/env python
# coding: utf-8

# 
# 
# 
# ![](https://www.cs.us.es/~fran/curso_unia/_images/clustering_techniques.png)
# 
# **A question always make managers painful : How to deliver the right message to right audience?**
# 
# * First, you need to know your product / service.
# * Second, and even more crucial, you need to understand your customers.
# 
# **It's time to explore !**
# 
# Target:
# 1. Find the behaviour patterns
# 2. Find the best cluster numbers
# 3. Cluster
# 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import seaborn as sns 
import plotly as py
import plotly.graph_objs as go
from sklearn.cluster import KMeans
import warnings


# In[ ]:


df = pd.read_csv(r'../input/Mall_Customers.csv')
df.head()


# In[ ]:


plt.figure(1 , figsize = (16 , 16))
n = 0
for x in ['Age' , 'Annual Income (k$)' , 'Gender']:
    n += 1
    plt.subplot(3 , 1 , n)
    sns.boxplot(x=x, y='Spending Score (1-100)', data=df, whis='range')
    plt.title('Distplot of {}'.format(x))


# **It shows signicifant differences, I can see 6 differnt patterns at least.**
# 
# 1. Customers under 40 years old have higher spending score and the higher range.
# 2. Customers between 40 and 50 years old have lower spending score and even the median is 43 - 44, the general top score is 60,still much lower than youngs.
# 3. Customers are above 50 years old are quite versatile, not easy to describe the habits.
# 4. Lower income customers (< 40k) have strong purchasing demand and higher spending score
# 5. Middle Income Customers (40k - 50k) are quite conservative in purchasing.
# 6. High Income Customers (>50k) have strong purchasing ability and high spending score.
# 
# This is very interesting and reminds me '**the poor middle class**'. 
# *  You are a poor, you don't give a shit and try your best to buy best things you can afford for yourself.
# *  You are a middle class, tons of bills need to pay to maintain a middle class life and you have to calculate every penny on entertaining yourself.
# * You are a rich, you can buy whatever goods you want to buy.

# ### The optimal number of cluster
# 

# In[ ]:


df_p = df.drop(['CustomerID'], axis=1)
df_p['Gender'] = df_p['Gender'].factorize()[0]


# In[ ]:


from sklearn.manifold import TSNE
tsn = TSNE()
res_tsne = tsn.fit_transform(df_p)


# In[ ]:


from sklearn.cluster import AgglomerativeClustering as AggClus
clus_mod = AggClus(n_clusters = 5) # no reason, just a try
assign = clus_mod.fit_predict(df_p)
plt.figure(figsize=(8,8))
sns.set(style='whitegrid',palette='pastel')
cmap = sns.cubehelix_palette(dark=.5, light=.5, as_cmap=True)
sns.scatterplot(x=res_tsne[:,0],y=res_tsne[:,1],s=100, hue=assign, palette='Paired');


# In[ ]:


from scipy.cluster.hierarchy import dendrogram, ward
sns.set(style='whitegrid')
plt.figure(figsize=(8,8))
link = ward(res_tsne)
dendrogram(link)
ax = plt.gca()
bounds = ax.get_xbound()
ax.plot(bounds, [30,30],'--', c='m')
ax.plot(bounds,'--', c='c')


# #### The 6 is the lucky number !

# ## K-Mean Clustering

# #### 1. Segment Age and Spending Score

# In[ ]:


X1 = df[['Age' , 'Spending Score (1-100)']].iloc[: , :].values
inertia = []
for n in range(1 , 7):
    algorithm = (KMeans(n_clusters = n ,init='k-means++', n_init = 6 ,max_iter=300, 
                        tol=0.0001,  random_state= 101  , algorithm='elkan') )
    algorithm.fit(X1)
    inertia.append(algorithm.inertia_)


# In[ ]:


plt.figure(1 , figsize = (16 ,6))
plt.plot(np.arange(1 , 7) , inertia , 'o')
plt.plot(np.arange(1 , 7) , inertia , '-' )
plt.xlabel('Number of Clusters') , plt.ylabel('Inertia')
plt.show()


# In[ ]:


algorithm = (KMeans(n_clusters = 6 ,init='k-means++', n_init = 6 ,max_iter=300, 
                        tol=0.0001,  random_state= 111  , algorithm='elkan') )
algorithm.fit(X1)
labels1 = algorithm.labels_
centroids1 = algorithm.cluster_centers_

h = 0.02
x_min, x_max = X1[:, 0].min() - 1, X1[:, 0].max() + 1
y_min, y_max = X1[:, 1].min() - 1, X1[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = algorithm.predict(np.c_[xx.ravel(), yy.ravel()]) 


# In[ ]:


plt.figure(1 , figsize = (16 , 6) )
plt.clf()
Z = Z.reshape(xx.shape)
plt.imshow(Z , interpolation='nearest', 
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap = plt.cm.hsv, aspect = 'auto', origin='lower')

plt.scatter( x = 'Age' ,y = 'Spending Score (1-100)' , data = df , c = labels1 , 
            s = 200 )
plt.scatter(x = centroids1[: , 0] , y =  centroids1[: , 1] , s = 200 , c = 'w' )
plt.ylabel('Spending Score (1-100)') , plt.xlabel('Age')


# #### 2. Segment Annual Income and Spending Score
# 

# In[ ]:


X2 = df[['Annual Income (k$)' , 'Spending Score (1-100)']].iloc[: , :].values
inertia = []
for n in range(1 , 7):
    algorithm = (KMeans(n_clusters = n ,init='k-means++', n_init = 6 ,max_iter=300, 
                        tol=0.0001,  random_state= 101  , algorithm='elkan') )
    algorithm.fit(X2)
    inertia.append(algorithm.inertia_)
                   
    
algorithm = (KMeans(n_clusters = 6 ,init='k-means++', n_init = 6 ,max_iter=300, 
                        tol=0.0001,  random_state= 101  , algorithm='elkan') )
algorithm.fit(X2)
labels2 = algorithm.labels_
centroids2 = algorithm.cluster_centers_


h = 0.02
x_min, x_max = X2[:, 0].min() - 1, X2[:, 0].max() + 1
y_min, y_max = X2[:, 1].min() - 1, X2[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z2 = algorithm.predict(np.c_[xx.ravel(), yy.ravel()]) 


# In[ ]:


plt.figure(1 , figsize = (16 , 6) )
plt.clf()
Z2 = Z2.reshape(xx.shape)
plt.imshow(Z2 , interpolation='nearest', 
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap = plt.cm.Set3, aspect = 'auto', origin='lower')

plt.scatter( x = 'Annual Income (k$)' ,y = 'Spending Score (1-100)' , data = df , c = labels2 , 
            s = 200 )
plt.scatter(x = centroids2[: , 0] , y =  centroids2[: , 1] , s = 200 , c = 'w' )
plt.ylabel('Spending Score (1-100)') , plt.xlabel('Annual Income (k$)')


# ![I adore the tidy and intuitive](https://frankdatascience.files.wordpress.com/2018/09/1_2bpc6k2c4ojhp00ijxbska.jpeg?w=1023)
