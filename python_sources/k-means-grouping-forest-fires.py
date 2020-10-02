#!/usr/bin/env python
# coding: utf-8

# # Analysing the  Brazil Forest burnings

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("../input/forest-fires-in-brazil/amazon.csv",encoding='latin1') 


# In[ ]:


data.head() 


# In[ ]:


data.isna().sum()


# No missing values.

# In[ ]:


data.duplicated().sum()


# 32 duplicates.

# In[ ]:


data.drop_duplicates(inplace=True) 


# In[ ]:


data.drop('date',axis=1,inplace =True) 


# The date features doesnt tells us much.

# In[ ]:


data = data.reset_index(drop=True)


# Reset the index because of the drop_duplicates.

# In[ ]:


data = data.replace({'Piau':'Piaui',
                     'Rio':'Rio de Janeiro'})


# # Graphs

# In[ ]:


plt.figure(figsize=(15,10))
ax=sns.distplot(data.number)
ax = ax.set(yticklabels=[],title='Histogram of number of fires reported')


# Very skewed right distribution.

# In[ ]:


ax = plt.figure(figsize=(15,10))

plt.subplot(2,1,1)
ax = sns.boxplot(data.year,data.number)

plt.subplot(2,1,2)
ax = sns.boxplot(data.state,data.number)
ax = plt.xticks(rotation=65)


# In[ ]:


pd.DataFrame(data.groupby(data.year).number.std())


# In[ ]:


pd.DataFrame(data.groupby(data.state).number.std())


# We can see that we have sparse data, regarding the number of fires. The standard deviation is very high, so for the next analysis I will be using only the median, beacause using average values could lead to false conclusions.

# In[ ]:


x = data.groupby(data.year).number.median()

ax = plt.figure(figsize=(15,10))
ax = plt.plot(x.index.values,x.values) 
ax = plt.title('Median of number of fires reported')
ax = plt.xlabel('Year')
ax = plt.ylabel('Median')

z = np.polyfit(x.index.values, x.values, 1)
p = np.poly1d(z)
ax = plt.plot(x.index.values,p(x.index.values),"r--")

ax = plt.legend(['Real Data','Trend Line'])


# We can see the median number of fires reported increased during this period, and it tends to continue that way.

# In[ ]:


x = data.groupby(data.month, sort=False).number.median()
plt.figure(figsize=(15,10))
ax = plt.plot(x.index.values, x.values) 
ax = plt.xticks(rotation=65)
ax = plt.title('Median of number of fires reported')
ax = plt.xlabel('Month')
ax = plt.ylabel('Median')


# In[ ]:


x = data.groupby(data.state).number.median()
plt.figure(figsize=(15,10))
ax = sns.barplot(y=x.index.values,x=x.values)
ax = ax.set(xlabel='Median of fires reported',ylabel='States',title='Fires Reported by State')


# In[ ]:


import geopandas as gpd


# In[ ]:


map_brazil = gpd.read_file('../input/brazil-geopandas-data/gadm36_BRA_1.shp')
map_brazil = map_brazil[['NAME_1','geometry']]
map_brazil = map_brazil.to_crs(epsg=4326)


# In[ ]:


map_brazil.head()


# In[ ]:


import unidecode
map_brazil['NAME_1'] = map_brazil['NAME_1'].apply(lambda x: unidecode.unidecode(x))
data['state'] = data['state'].apply(lambda x: unidecode.unidecode(x))


# In[ ]:


map_brazil['centroid'] = map_brazil.geometry.centroid


# In[ ]:


median = data.groupby('state').number.median()

map_brazil = map_brazil.join(median,on='NAME_1')


# In[ ]:


map_brazil.head()


# In[ ]:


data['month'] = data.month.replace(data.month.unique(),range(1,13))


# In[ ]:


fig,ax = plt.subplots(figsize=(20,10))
map_brazil.plot(column='number',ax=ax,alpha=0.4,edgecolor='black',cmap='YlOrRd',legend=True)
plt.title("Median Number of Fires")
plt.axis('off')


for x, y, label in zip(map_brazil.centroid.x, map_brazil.centroid.y, map_brazil.NAME_1):
    ax.annotate(label, xy=(x, y), xytext=(3,3), textcoords="offset points",color='blue')


# # K-means 

# In[ ]:


map_brazil['lat_long'] = map_brazil.centroid.apply(lambda x : x.coords[0])


# In[ ]:


x = map_brazil[['NAME_1','lat_long']]
x = x.set_index('NAME_1')


# In[ ]:


data = data.set_index('state')


# In[ ]:


data = data.join(x)


# In[ ]:


data[['x', 'y']] = pd.DataFrame(data['lat_long'].tolist(), index=data.index) 
data.drop('lat_long',axis=1,inplace=True)


# In[ ]:


from sklearn.cluster import KMeans


# In[ ]:


inertia =[] 
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
    inertia.append(kmeans.inertia_)


# In[ ]:


ax = plt.plot(range(1,10),inertia)


# By the elbow method we should use 3 clusters. 

# In[ ]:


kmeans = KMeans(n_clusters=3, random_state=0).fit(data)


# I'm going to use PCA for dimensionality reduction so I can plot the clusters. 

# In[ ]:


from sklearn.preprocessing import StandardScaler

x = StandardScaler().fit_transform(data)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['1', '2'])

plt.figure(figsize=(10,10))
ax = plt.scatter(principalDf['1'],principalDf['2'],c=kmeans.labels_)


# In[ ]:


pc = pd.DataFrame(pca.components_) 
pc.columns = data.columns 
pc


# The data is mainly grouped by number and state.
# 

# In[ ]:


data['labels'] = kmeans.labels_


# In[ ]:


data.labels.value_counts() 


# In[ ]:


plt.figure(figsize=(10,10))
ax = sns.boxplot(data.labels,data.number)


# In[ ]:


plt.figure(figsize=(10, 10))
ax = sns.countplot(data.month,hue=data.labels)


# In[ ]:


plt.figure(figsize=(10, 10))
ax = sns.countplot(data.year,hue=data.labels)


# In[ ]:


plt.figure(figsize=(10, 10))
ax = sns.countplot(data.index.values,hue=data.labels)
ax = plt.xticks(rotation=65)

