#!/usr/bin/env python
# coding: utf-8

# # 1. Import relevant libraries

# In[ ]:


import numpy as np # library to handle data in a vectorized manner

import pandas as pd # library for data analsysis
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

import json # library to handle JSON files




import requests # library to handle requests
from bs4 import BeautifulSoup # library to parse HTML and XML documents

from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

# import k-means from clustering stage
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

import folium # map rendering library

print("Libraries imported.")


# # 2. Webscrapping using BeautifulSoup

# In[ ]:


data=requests.get('https://en.wikipedia.org/wiki/List_of_neighbourhoods_in_Mumbai').text


# In[ ]:


soup=BeautifulSoup(data,'html.parser')


# In[ ]:


area=[]
loc=[]
lat=[]
lon=[]


# In[ ]:


for row in soup.find('table').find_all('tr'):
    cells=row.find_all('td')
    if (len(cells)>0):
        area.append(cells[0].text)
        loc.append(cells[1].text)
        lat.append(cells[2].text)
        lon.append(cells[3].text)


# Let us take a look at the various data from the table stored in the four lists.

# In[ ]:


area_mumbai=[]
for areas in area:
    area_mumbai.append(areas.replace('\n',''))


# In[ ]:


area_mumbai[0:5]


# In[ ]:


loc_mumbai=[]
for locations in loc:
    loc_mumbai.append(locations.replace('\n',''))


# In[ ]:


loc_mumbai[0:5]


# In[ ]:


lat_mumbai=[]
for lats in lat:
    lat_mumbai.append(lats.replace('\n',''))
lon_mumbai=[]
for lons in lon:
    lon_mumbai.append(lons.replace('\n',''))


# In[ ]:


lat_mumbai[0:5]


# In[ ]:


lon_mumbai[0:5]


# Let us transform all the webscrapped data into a more readable dataframe as shown below:

# In[ ]:


df_mumbai=pd.DataFrame(columns=['Area','Location','Latitude','Longitude'])
df_mumbai['Area']=area_mumbai
df_mumbai['Location']=loc_mumbai
df_mumbai['Latitude']=lat_mumbai
df_mumbai['Longitude']=lon_mumbai
df_mumbai


# We see that row number 82 has an incorrect longitude. We google it and correct the longitude value.

# In[ ]:


df_mumbai['Longitude'][82]=72.8479
df_mumbai


# In[ ]:


df_mumbai.to_csv('Mumbai neighborhood coordinates.csv')


# # 3. Using Folium to visualise the areas on map of Mumbai

# In[ ]:


latitude=19.07 
longitude=72.87

map_mumbai = folium.Map(location=[latitude, longitude], zoom_start=11)

for lat,lon,areas,location in zip(df_mumbai['Latitude'],df_mumbai['Longitude'],df_mumbai['Area'],df_mumbai['Location']):
                                        
                                        label='{} {}'.format(areas,location)
                                        label=folium.Popup(label)
                                        
                                        folium.CircleMarker(
                                            [lat,lon], radius=5,popup=label,color='orange',fill=True,fill_color='black',fill_opacity=0.6).add_to(map_mumbai)
                                   


# In[ ]:


map_mumbai


# # 4. Using Foursquare API to search for top 20 places within a radius of 500m

# In[ ]:


CLIENT_ID='Your client ID'
CLIENT_SECRET='Your client secret'
VERSION = '20180605'


# In[ ]:


venues = []

radius = 1000
LIMIT = 100


for lat, lon, loc,areas in zip(df_mumbai['Latitude'], df_mumbai['Longitude'], df_mumbai['Location'], df_mumbai['Area']):
    url = "https://api.foursquare.com/v2/venues/explore?client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}".format(
        CLIENT_ID,
        CLIENT_SECRET,
        VERSION,
        lat,
        lon,
        radius, 
        LIMIT)
    
    results = requests.get(url).json()["response"]['groups'][0]['items']
    
    for venue in results:
        venues.append((
            areas, 
            loc,
            lat, 
            lon, 
            venue['venue']['name'], 
            venue['venue']['location']['lat'], 
            venue['venue']['location']['lng'],  
            venue['venue']['categories'][0]['name']))


# In[ ]:


venues_df=pd.DataFrame(venues)
venues_df.rename(columns={0:'Area',1:'Location',2:'Area latitude',3:'Area longitude',4:'Venue name',5:'Venue latitude',6:'Venue longitude',7:'Venue category'},inplace=True)
venues_df.head()


# Let us check the number of unique venue categories.

# In[ ]:


venues_df['Venue category'].unique()


# # 5. Data analysis

# In[ ]:


len(venues_df['Venue category'].unique())


# In[ ]:


categories_onehot=pd.get_dummies(venues_df['Venue category'])


# In[ ]:


mumbai_category_df=pd.DataFrame(venues_df['Area'],columns=['Area'])


# In[ ]:


mumbai_category_df=mumbai_category_df.merge(categories_onehot,on=mumbai_category_df.index)


# In[ ]:


mumbai_category_df.head()


# In[ ]:


mumbai_category_grouped=mumbai_category_df.groupby(['Area']).mean().reset_index()


# In[ ]:





# In[ ]:


mumbai_category_grouped.drop('key_0',axis=1,inplace=True)


# In[ ]:


mumbai_category_grouped.head()


# # 6. Segregate the venue categories of interest

# Since we are primarily interested to set up a bar, we can segregate the above dataframe into venue categories such as bars, pubs, sports bars,cocktail bar,beer bar, beer garden.

# In[ ]:


bar_list=['Sports Bar','Gastropub','Bar','Beer Bar',
          'Beer Garden','Club House',
          'Lounge','Cocktail Bar','Hotel Bar',
          'Bistro','Brewery','Wine Bar','Nightclub']


# In[ ]:


bar_category_df=pd.DataFrame(columns=[mumbai_category_grouped.columns])
bar_category_df=mumbai_category_grouped[mumbai_category_grouped['Pub']>0]


# In[ ]:





# In[ ]:


for i in range(0,len(bar_list)):
    bar_category_df=bar_category_df.append(mumbai_category_grouped[mumbai_category_grouped['{}'.format(bar_list[i])]>0])


# In[ ]:


bar_category_df.reset_index(drop=True,inplace=True)


# In[ ]:


bar_category_df.head()


# # 5. Using KMeans clustering

# In[ ]:


wcss=[]
for i in range(1,15):
    kmeans=KMeans(n_clusters=i,max_iter=300)
    kmeans.fit(bar_category_df.iloc[:,1:])
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,15),wcss)
plt.ylabel('WCSS')
plt.xlabel('K clusters')
plt.xticks(np.arange(1,15))
plt.axvline(5,color='red')


# From the above elbow figure, it can't be completely determined as to what should be the optimum clusters. At k=5, the slope of WCSS reduces. Moreover, k=5 gives decent results in our analysis. Hence, we choose k=5 for further study.

# In[ ]:


k=5
kmeans=KMeans(n_clusters=k)
kmeans.fit(bar_category_df.iloc[:,1:])


# In[ ]:


labels=kmeans.labels_
cluster_df=pd.DataFrame(columns=['Area','Label'])
cluster_df['Area']=bar_category_df.iloc[:,0]


# In[ ]:


cluster_df['Label']=labels
cluster_df.head()


# In[ ]:


cluster_df=cluster_df.merge(df_mumbai,on='Area')


# In[ ]:


cluster_df


# # 6. Projecting the various clusters on a folium map

# In[ ]:


map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(k)
ys = [i+x+(i*x)**4 for i in range(k)]
colors_array = cm.rainbow(np.linspace(0,1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, area,loc,cluster in zip(cluster_df['Latitude'], cluster_df['Longitude'], cluster_df['Area'], cluster_df['Location'],cluster_df['Label']):
    label = folium.Popup('{} ({}) - Cluster {}'.format(area,loc,cluster+1), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=6,
        popup=label,
        color=rainbow[cluster-2],
        fill=True,
        fill_color=rainbow[cluster-2],
        fill_opacity=0.8).add_to(map_clusters)
       
map_clusters


# let us explore the various clustered areas as follows

# ## Cluster 1

# In[ ]:


cluster_df[cluster_df['Label']==0]


# ## Cluster 2

# In[ ]:


cluster_df[cluster_df['Label']==1]


# ## Cluster 3

# In[ ]:


cluster_df[cluster_df['Label']==2]


# # Cluster 4

# In[ ]:


cluster_df[cluster_df['Label']==3]


# # Cluster 5

# In[ ]:


cluster_df[cluster_df['Label']==4]


# In[ ]:


sizes=[]

for labels in np.arange(0,5):
    sizes.append(cluster_df[cluster_df['Label']==labels].shape[0])


# In[ ]:


sizes_df=pd.DataFrame(columns=['Label name','Label size'])


# In[ ]:


sizes_df['Label name']=np.arange(1,6)
sizes_df['Label size']=sizes


# In[ ]:


sizes_df.index=sizes_df['Label name']


# In[ ]:


sizes_df.drop('Label name', axis=1,inplace=True)
sizes_df


# From the above cluster sizes and markers on the folium map, it is clear that are to avoid cluster 5 which is at Vasai region.
# 
# Clusters 3 and 4 have high number of nightclub,bar,gastropubs in the vicinity. Hence, setting up shop here would require considerable capital.
# 
# Clusters 1 and 2 seem to be ideal for setting up a nightclub with low competition and much more room for improvement.
# 
# 
# For a detailed analysis and methodology that was used, kindly visit this article below:
# 
# [In depth analysis](https://www.linkedin.com/pulse/optimising-property-location-mumbai-arindam-baruah/)
# 
# 
# Please take a look at my GitHub profile if you find this interesting. Always love some great inputs from fellow Kagglers ! 
# 
# [My GitHub profile](https://github.com)

# In[ ]:




