#!/usr/bin/env python
# coding: utf-8

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


# Load data by pandas

# In[ ]:


df = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
df.head()


# Let's check the neibourhood in data frame.

# In[ ]:


df.neighbourhood.unique()


# Because the last_review, host_name and id are not important for analysis. We can drop these three columns and set 0 by NaN for reviews_per_month.

# In[ ]:


drop_elements = ['last_review', 'host_name','id']
df.drop(drop_elements, axis = 1, inplace= True)
df.fillna({'reviews_per_month':0}, inplace=True)
df.reviews_per_month.isnull().sum()


# In[ ]:


df.rename(columns={'neighbourhood_group':'borough'}, inplace = True)
df.head()


# Let's check the quantity of each type of room in each region by count plot.

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

fig = plt.subplots(figsize = (12,5))
sns.countplot(x = 'room_type', hue = 'borough', data = df)


# We observe that the most of rooms are concentrated in Manhattan and Brooklyn.

# Let's find the 'neighborhood' of the most rooms in these five 'borough'

# In[ ]:


a = df.groupby('borough')['neighbourhood'].value_counts()
b = a.index.levels[0]
a14 = []
a16 = []

for i in range(len(b)):
    df_level = a.loc[[b[i], 'neighbourhood']]
    df_level_ch = pd.DataFrame(df_level)
    for j in range(1):
        a13 = df_level_ch.iloc[j].name
        b1 = df_level_ch.values[j][0]
        print(a13, b1)
        a14.append(a13)
        a16.append(b1)
a15 = pd.DataFrame(a14)
a16 = pd.DataFrame(a16)
a17 = pd.concat([a15, a16], axis = 1)
a17.columns = ['borough', 'neighbourhood', 'total']
a17.plot.bar(x = 'neighbourhood', y = 'total')


# Let's check the average price of shared room in NY.

# In[ ]:


df.price.mean()


# Let's check the price distribution in these five regions.

# In[ ]:


plt.figure(figsize = (16,12))
prix = df[df['price'] < 600]
sns.violinplot(x = 'borough', y = 'price', data = prix)


# In general, there are more shared room in Manhattan and Brooklyn. This is related to the local economy and we know where is the most neighbourhood in each region and the price. The most expensive area is Manhattan and second one is Brooklyn. Because Manhattan has a high cost of living. Bronx is the cheap. The quantity of shared room is less than private room and entire home/appartment.

# If someone wants to do a bussiness on shared room in New York, how should it be priced.
# 
# We can use the cluster model for this question.

# In[ ]:


df_onehot = pd.get_dummies(df[['price']], prefix = "", prefix_sep = "")
df_onehot['neighbourhood'] = df['neighbourhood']
fixed_columns = [df_onehot.columns[-1]] + list(df_onehot.columns[:-1])
df_grouped = df_onehot.groupby('neighbourhood').mean().reset_index()
df_grouped.head(20)


# Cluster Neighbourhood

# In[ ]:


# Import libraries
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist


# In[ ]:


df_clustering = df_grouped.drop('neighbourhood',1)
df_clustering.head()


# Determine cluster number by Elbow Method

# In[ ]:


K = range(1,10)
distortions = []
for k in K:
    kmeans = KMeans(init = 'k-means++', n_clusters = k, n_init = 12, random_state = 0)
    kmeans.fit(df_clustering.values.reshape(-1,1))
    distortions.append(sum(np.min(cdist(df_clustering.values.reshape(-1, 1),kmeans.cluster_centers_, 'euclidean'), axis = 1)) / df_clustering.shape [0])

import matplotlib.pyplot as plt
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal K')
plt.show()               


# In[ ]:


num_clusters = 6

kmeans = KMeans(n_clusters = num_clusters, random_state = 0).fit(df_clustering)
kmeans.labels_


# In[ ]:


df_grouped.insert(0, 'Cluster', kmeans.labels_)


# In[ ]:


df_grouped.head()


# In[ ]:


df_co = df.copy()
df_co.drop_duplicates(subset = ['neighbourhood'])
drop_element = 'price'
df_co.drop(drop_element, axis = 1)
df_co.head()


# In[ ]:


df_merge = pd.merge(df_co, df_grouped[['Cluster','neighbourhood','price']],on = 'neighbourhood')
df_merge.head()


# In[ ]:


df_merge.head()


# ### Cluster 1

# In[ ]:


cluster_1 = df_merge.loc[df_merge['Cluster'] == 0]
cluster_1.head(3)


# In[ ]:


cluster_1.shape


# ### Cluster 2

# In[ ]:


cluster_2 = df_merge.loc[df_merge['Cluster'] == 1]
cluster_2.head()


# In[ ]:


cluster_2.shape


# ### Cluster 3

# In[ ]:


cluster_3 = df_merge.loc[df_merge['Cluster'] == 2]
cluster_3.head()


# In[ ]:


cluster_3.shape


# ### Cluster 4

# In[ ]:


cluster_4 = df_merge.loc[df_merge['Cluster'] == 3]
cluster_4.head()


# In[ ]:


cluster_4.shape


# ### Cluster 5

# In[ ]:


cluster_5 = df_merge.loc[df_merge['Cluster'] == 4]
cluster_5.head()


# In[ ]:


cluster_5.shape


# ### Cluster 6

# In[ ]:


cluster_6 = df_merge.loc[df_merge['Cluster'] == 5]
cluster_6.head()


# In[ ]:


cluster_6.shape


# In[ ]:


# This function is used to help plot the maps in the notebook
def embed_map(m, file_name):
    from IPython.display import IFrame
    m.save(file_name)
    return IFrame(file_name, width='100%', height='500px')


# In[ ]:


# Import libraries
import matplotlib.cm as cm
import matplotlib.colors as colors
import folium

# Creat the folium map

map_cluster = folium.Map(location = [40.730610,-73.935242], zoom_start=10)

# Set color for clusters
x = np.arange(num_clusters)
ys = [i + x +(i*x)**2 for i in range(num_clusters)]
colors_array = cm.rainbow(np.linspace(0,1,len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# Set the marker for the map
markers_colors = []
for lat, lng, cluster in zip(df_merge['latitude'], df_merge['longitude'], df_merge['Cluster']):
  label = folium.Popup(' Cluster ' + str(cluster), parse_html=True)
  folium.CircleMarker([lat, lng],
                      radius=2,
                      popup=label,
                      color=rainbow[cluster-1],
                      fill=True,
                      fill_color=rainbow[cluster-1],
                      fill_opacity=0.7).add_to(map_cluster)
embed_map(map_cluster, 'map_cluster.html')


# We can check the total number of shared room for each cluster.

# In[ ]:


df_clu=pd.DataFrame(np.arange(6).reshape((1,6)),index=['0'],columns=['cluster_1','cluster_2','cluster_3','cluster_4','cluster_5','cluster_6'])
df_clu.cluster_1 = len(cluster_1)
df_clu.cluster_2 = len(cluster_2)
df_clu.cluster_3 = len(cluster_3)
df_clu.cluster_4 = len(cluster_4)
df_clu.cluster_5 = len(cluster_5)
df_clu.cluster_6 = len(cluster_6)

plt.figure(figsize = (16,12))
sns.barplot(data = df_clu)
plt.xlabel('Cluster Number',fontsize =12)
plt.ylabel('Total Shared Room', fontsize = 12)
plt.title('Total Number shared Room for Each Cluster',fontsize = 12 )
plt.show()


# In domain of cluster 4 has the most quantity of shared room. In the domain of cluster 6 and cluster 1 have more room than the other. The cluster 3 has the least shared room. It means cluster 4 is the most acceptable.

# Let's check the average price for cluster 1, cluster 4 and cluster 6.

# In[ ]:


print('The mean price of cluster 1 is: ',cluster_1.price_x.mean())
print('The total review in cluster 1 is:',len(cluster_1.number_of_reviews))
print('The mean price of cluster 4 is: ',cluster_4.price_x.mean())
print('The total review in cluster 4 is:',len(cluster_4.number_of_reviews))
print('The mean price of cluster 6 is: ',cluster_6.price_x.mean())
print('The total review in cluster 6 is:',len(cluster_6.number_of_reviews))


# Refer to the average price of cluster 1, 4 and 6 (and average price of NY is 152.72), we can define our shared room price. Then, we can check which type of room is more polular in each cluster.

# In[ ]:


plt.figure(figsize = (16,12))
sns.countplot(x = 'room_type', hue = 'Cluster', data = df_merge)


# ### Conclusion
# 
# 
# 
# We observed if we have a private room for sharing, we can refer to cluster 1 and cluster 4 of pricing. If we have entire apt, we can refer to cluster 4 of pricing. For the area, we can choose Manhattan and Brooklyn.
