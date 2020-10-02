#!/usr/bin/env python
# coding: utf-8

# # Global Terrorism Data
# 
# In this notebook we will look at the Global Terrorism Database owned by the University of Mariland and try to answer a few limited questions regarding terrorism. We would also do a clustering of terrorist attacks based on latitude and longitude data using Density-based spatial clustering of applications with noise (DBSCAN) algorithm. 
# 
# 
# Note that all data used in this notebook belongs to:
# 
# National Consortium for the Study of Terrorism and Responses to Terrorism (START), University of Maryland. (2018). The Global Terrorism Database (GTD) [Data file]. Retrieved from https://www.start.umd.edu/gtd

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


raw_data = pd.read_csv('/kaggle/input/gtd/globalterrorismdb_0718dist.csv', engine='python')


# In[ ]:


pd.options.display.max_columns = 200
raw_data.head()


# After looking at the raw data and reading the explanation of its columns (https://start.umd.edu/gtd/downloads/Codebook.pdf), let us think of what information do we want to get out of the data, given the columns only (for the purpose of this notebook)
# 
# Some of the things that we want to know intuitively among other things are:
# 
# - What is the trend of terrorism attacks? Increasing, decreasing or relatively constant?
# - Which countries has the most cases of terrorist attacks?
# - Who / which group(s) are mostly responsible for the attacks?
# - From which countries do these groups originate from?
# - How many perpetrators usually is there for an attack if it is a group attack?
# - How many casualities are there for some of the deadliest attacks?
# - Can we see where are the attacks took place? 
# - What kind of attacks are there and what is the most common one?
# - Who are the targets of the attacks? 
# - What are the weapons mostly used in the attacks?
# 
# Let's try to answer some of them
# 

# In[ ]:


yearlyacts = raw_data.groupby('iyear',as_index = True, group_keys = False)['eventid'].count()
yearlyacts.plot(kind='line', figsize = (20,5),xticks = range(1970,2018),rot = 45, title='Yearly Cases of Terrorist Attacks',                linewidth = 3);


yearlyacts = raw_data.groupby(['iyear','region_txt'],as_index = False, group_keys = False)['eventid'].count()
#yearlyacts.region_txt.value_counts()
yearlyacts = yearlyacts.rename(columns={'region_txt': "Region", "eventid": "Cases"})

yearlyacts.pivot_table(index='iyear', columns='Region', aggfunc=np.sum, fill_value=0).plot(kind='line', figsize = (20,10),                                                                                              xticks = range(1970,2018),rot = 45,                                                                                              linewidth=5,title='Cases by Region');


# We can see above that since 2003 there had been an increase in the yearly number of the cases. If we divide by region, we can see that the increase was contributed mostly by the regions of Middle East & North Africa, South Asia, and Sub-saharan Africa.
# 
# Hence to narrow down our scope, we would use data from 2003 onwards for our clustering

# In[ ]:


atk_types = raw_data.groupby('attacktype1_txt', group_keys=False)['eventid'].count()
atk_types.nlargest(50).plot(kind = 'bar', figsize = (20,5),grid=True, rot=45, title='Types of Attack');


# In[ ]:


n_killsum = raw_data.groupby('country_txt',group_keys=False)['nkill'].sum()
n_cases = raw_data.groupby('country_txt', group_keys=False)['eventid'].count()
n_cases.nlargest(50).plot(kind = 'bar', figsize = (20,5),yticks = range(0,25001,2500),grid=True, title='Cases by Countries');


# In[ ]:


weapons = raw_data.groupby('weaptype1_txt', group_keys=False)['eventid'].count()
weapons.nlargest(50).plot(kind = 'bar', figsize = (20,5),grid=True, rot = 45, title= 'Weapon Used');


# In[ ]:


targets = raw_data.groupby('targtype1_txt', group_keys=False)['eventid'].count()
targets.nlargest(50).plot(kind = 'bar', figsize = (20,5),grid=True, rot=70, title='Targets');


# In[ ]:


print('Average number of perpetrators: ', round(raw_data[(raw_data.nperps > 0)]['nperps'].mean(),1))
print('Median of perpetrators: ', round(raw_data[(raw_data.nperps > 0)]['nperps'].median(),1))


# We can imagine that for most of the cases, the perpetrator is an individual / affiliated individual, hence the median of 2 person. While at the other end of the spectrum, there are organized terrorist groups which attacks with a large number of perpetrators, hence skewing the average way past the median

# In[ ]:


#print(lon[lon < -180])
print(raw_data.at[17658,'longitude']) #change this wrong data

raw_data.at[17658,'longitude'] = -86.185896

print(raw_data.at[17658,'longitude'])


# In[ ]:


lat = raw_data[raw_data['iyear']>=2003]['latitude']
lon = raw_data[raw_data['iyear']>=2003]['longitude']


plt.figure(figsize = (20,15))

#extent = [-20, 40, 30, 60]
#central_lon = np.mean(extent[:2])
#central_lat = np.mean(extent[2:])

#ax = plt.axes(projection=ccrs.AlbersEqualArea(central_lon, central_lat))
ax = plt.axes(projection=ccrs.Mercator())

#ax.set_extent(extent,crs=ccrs.PlateCarree())
ax.scatter(lon,lat,transform=ccrs.PlateCarree(),s=1, c = 'red')
ax.coastlines()
ax.add_feature(cartopy.feature.BORDERS, linestyle="-")
ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
plt.title('Worldwide Terrorists Attacks Since 2003', fontsize = 20);


# As we can see above, we can see the hotspots for terrorist attacks. Next we will try to examine the clusters of the attacks for the region of Europe & Middle East and South Asia.
# 
# We will use SKLearn's DBSCAN clustering library which clusters the attacks based on its density on the latitude longitude map. Here we define the minimum size of a cluster to be 50 attacks. 
# 
# We would also look at only domestic case of terrorist attacks meaning that the perpetrator is a citizen of the country where the attack took place. This is because we would like to find the origin of the attack groups by looking at their cluster, hence adding international attacks will have potentially unintended meaning for our result. 

# In[ ]:


df_filtered = raw_data[(raw_data['latitude']>25) & (raw_data['latitude']<60) &                 (raw_data['longitude']> -20) & (raw_data['longitude']< 60) & (raw_data['INT_LOG']==0) & (raw_data['iyear']>=2003)]

lat2 = df_filtered['latitude']
lon2 = df_filtered['longitude']

data = pd.DataFrame({'latitude':lat2, 'longitude': lon2})
data = data.dropna(how="any")
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

db = DBSCAN(eps=0.3, min_samples=50).fit(data_scaled)

labels = db.labels_

data['cluster'] = labels
df_filtered['cluster'] = data['cluster']
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)


# We can see that we ended up with 9 clusters, we can also see that we have 258 noise points which corresponds to the 258 un-clusterable attacks. We can think of the noise points as 'Others' cluster. 

# In[ ]:


plt.figure(figsize = (20,15))

#extent = [-20, 60, 25, 60]
#central_lon = np.mean(extent[:2])
#central_lat = np.mean(extent[2:])

legendname = []
for items in np.sort(df_filtered.cluster.unique()):
    if items == -1:
        legendname.append('Others / Cluster'+str(items))
    else:
        legendname.append('Cluster'+str(items))

#ax = plt.axes(projection=ccrs.AlbersEqualArea(central_lon, central_lat))
ax = plt.axes(projection=ccrs.Mercator())
#ax.set_extent(extent,crs=ccrs.PlateCarree())
scatter = ax.scatter(df_filtered.longitude,df_filtered.latitude,transform=ccrs.PlateCarree(), c = df_filtered.cluster,                     cmap = 'tab10', s = df_filtered['nkill']*10) #we define the marker size to corresponds to the number of casuality in the attack
ax.coastlines(linewidth = 2);
ax.add_feature(cartopy.feature.BORDERS, linestyle="-", linewidth = 2);
ax.add_feature(cartopy.feature.OCEAN);
#ax.add_feature(cartopy.feature.LAND, edgecolor='black')
ax.add_feature(cartopy.feature.LAKES, edgecolor='black');
ax.gridlines(crs=ccrs.PlateCarree());
plt.title('Europe and Middle East Domestic Terrorists Attacks Since 2003', fontsize = 20);
plt.legend(handles=scatter.legend_elements()[0], labels= legendname,fontsize=20, loc='upper right');


# Above we can see the clusters on the map for Europe and Middle East region. The circle size corresponds to the number of casualities in the attacks. Bigger circle means the number of casualities are higher and vice versa.
# 
# Now let's see how our clusters corresponds to the perpetrator group

# In[ ]:


x = df_filtered.groupby(['gname'],as_index = False,group_keys = False).filter(lambda x: len(x) >= 50)
x.groupby(['cluster','gname']).count()['eventid']


# Now lets look at the cases in South Asia

# In[ ]:


df_filtered_sa = raw_data[(raw_data['region_txt'] == 'South Asia')& (raw_data['INT_LOG']==0) & (raw_data['iyear']>=2003)]

lat_sa = df_filtered_sa['latitude']
lon_sa = df_filtered_sa['longitude']

data_sa = pd.DataFrame({'latitude':lat_sa, 'longitude': lon_sa}, index = df_filtered_sa.index)
data_sa = data_sa.dropna(how="any")

data_scaled_sa = scaler.fit_transform(data_sa)

db_sa = DBSCAN(eps=0.3, min_samples=100).fit(data_scaled_sa)

labels_sa = db_sa.labels_

data_sa['cluster'] = labels_sa

df_filtered_sa['cluster'] = data_sa['cluster']

# Number of clusters in labels, ignoring noise if present.
n_clusters_sa = len(set(labels_sa)) - (1 if -1 in labels_sa else 0)
n_noise_sa = list(labels_sa).count(-1)

print('Estimated number of clusters: %d' % n_clusters_sa)
print('Estimated number of noise points: %d' % n_noise_sa)


# In[ ]:


plt.figure(figsize = (20,15))

#extent = [-20, 60, 25, 60]
#central_lon = np.mean(extent[:2])
#central_lat = np.mean(extent[2:])

legendname_sa = []
for items in np.sort(data_sa.cluster.unique()):
    if items == -1:
        legendname_sa.append('Others / Cluster'+str(items))
    else:
        legendname_sa.append('Cluster'+str(items))

#ax = plt.axes(projection=ccrs.AlbersEqualArea(central_lon, central_lat))
ax = plt.axes(projection=ccrs.Mercator())
#ax.set_extent(extent,crs=ccrs.PlateCarree())
scatter = ax.scatter(df_filtered_sa.longitude,df_filtered_sa.latitude,transform=ccrs.PlateCarree(), c = df_filtered_sa.cluster,                    s = df_filtered_sa.nkill*10);
ax.coastlines(linewidth = 2);
ax.add_feature(cartopy.feature.BORDERS, linestyle="-", linewidth = 2);
ax.add_feature(cartopy.feature.OCEAN);
#ax.add_feature(cartopy.feature.LAND, edgecolor='black')
ax.add_feature(cartopy.feature.LAKES, edgecolor='black');
ax.gridlines(crs=ccrs.PlateCarree());
plt.title('South Asian Domestic Terrorists Attacks Since 2003', fontsize = 20);
plt.legend(handles=scatter.legend_elements()[0], labels= legendname_sa,fontsize=20);


# Above we can see the clusters on the map for South Asia region. The circle size corresponds to the number of casualities in the attacks. Bigger circle means the number of casualities are higher and vice versa.

# In[ ]:


x_sa = df_filtered_sa.groupby('gname',group_keys = False).filter(lambda x: len(x) >= 50)
x_sa.groupby(['cluster','gname']).count()['eventid']


# In[ ]:





# In[ ]:





# In[ ]:




