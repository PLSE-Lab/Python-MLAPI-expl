#!/usr/bin/env python
# coding: utf-8

# ## BATTLE OF THE SOLAR POWER PLANTS

# #### Introduction:
# Pakistan' peak electricity demand is estimated at 25,000 MW while its total power generation capacity is 28,000 MWs. Pakistan meets two-thirds of its energy requirement from fuel oil and natural gas. The share of renewables in Pakistan's energy mix is around 5% only. 
# In November 2019 the Government of Pakistan unveiled its new Renewable Energy Development Policy. The policy aims to increase the share of renewables in Pakistan's energy mix by 30%. Around 8,000MW cheap renewable clean and green energy will be added to the system by 2025 while it will be increased to 20,000MW by 2030.
# In order to meet the goals stated in this policy, Pakistan will have to invest heavily in Solar and Wind energy plants. This will not be possible without Utility-Scale Soalr Projects. In this study we will try to determin the best places in Pakistan for setting up a solar Independent Power Plant (Solar IPP) which will provide better electricity output at optimized CAPEX.
# 

# #### Problem:
# Like with all major utility scale projects, transmission cost and transmission losses are a major bottleneck in a project's feasibility. Pakistan's regions which receive very high Direct Normal Irradiation (DNI) of above 6 kWh/m2/day are very distant from the national power grid. Which means developing a solar IPP in one of those regions will have huge transmission costs.
# In order to overcome this problem I have decided to chose sites which are nearest to existing power stations.
# Another major factor in developing a Solar IPP is the cost of Land. The nearer the IPP site is to a city or a major population center the higher will be the cost of land and there will also be difficulties in acquiring a single plot of land for the IPP.
# 
# 
# 

# #### Data: 
# The World Bank Group has worked extensively on Pakistan's electricity transmission grid data. A GEOJSON file listing all nodes in the national grid is available at their website https://energydata.info/dataset/pakistan-electricity-transmission-network-2017 I have used that data in this study. There are a lot of un-named entries in the data set which I cleaned to get the locations of the grid stations.
# Then I by using the Foursuare API I explored venues nearby the grid stations. The purpose was to explore how many venues are there within a range of 15 kms of the Grid Station and how what was the mean distance of those venues from the Grid locations. Greater the number of venues near a grid and smaller mean distance of those venues from the Grid Location would mean that the Grid station is in an urban area and land development costs would be much higher.
# Finally I use the Global Solar Atlas website https://globalsolaratlas.info/map?c=30.637912,68.994141,5&r=PAK to get the Specific photovoltaic power output (PVOUT) values of the grid locations. This results in a dataframe with coordinates of all grid locations, their nearby venues count, mean distances from the venues and the Specific photovoltaic power output (PVOUT) values for each location. Now we can run the K-means clustering algorithm to cluster the grid nodes and label them to find out the most appropriate grid nodes for the development of a Solar IPP.

# 1. Lets Start by importing the required Libraries

# In[ ]:


import numpy as np # library to handle data in a vectorized manner

import pandas as pd # library for data analsysis
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import json # library to handle JSON files

get_ipython().system("conda install -c conda-forge geopy --yes # uncomment this line if you haven't completed the Foursquare API lab")
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values

import requests # library to handle requests
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

# import k-means from clustering stage
from sklearn.cluster import KMeans

import json
import pandas as pd
import geojson
import geojsonio

#!conda install -c conda-forge folium=0.5.0 --yes # uncomment this line if you haven't completed the Foursquare API lab
import folium # map rendering library

print('Libraries imported.')


# 2. Lets import the GEOJSON file with the grid coordinates 

# In[ ]:


with open('PTN.geojson', 'r') as f:
    network_data = json.load(f)


# In[ ]:


power_network=network_data['features']


# In[ ]:


power_network[0]


# In[ ]:


# define the dataframe columns
column_names = ['Node name', 'Description','Node type', 'Latitude', 'Longitude'] 

# instantiate the dataframe
PTN = pd.DataFrame(columns=column_names)


# In[ ]:


PTN


# 3. Now lets read the data from GEOJSON file to our Dataframe

# In[ ]:


for data in power_network:
    name = data['properties']['name'] 
    description = data['properties']['other_tags']
        
    network_type = data['geometry']['type']
    network_latlon = data['geometry']['coordinates'][0]
    network_lat = network_latlon[1]
    network_lon = network_latlon[0]
    
    PTN = PTN.append({'Node name': name,
                                          'Description': description,
                                          'Node type':network_type,
                                          'Latitude': network_lat,
                                          'Longitude': network_lon}, ignore_index=True)


# In[ ]:


PTN.head()


# 4. Lets analyze the 'Node types'

# In[ ]:


PTN.groupby('Node type').count()


# In[ ]:


#As all datapoints are 'LineString' we can drop the entire column from our DF
PTN=PTN.drop('Node type', axis=1)


# In[ ]:


PTN


# In[ ]:


# Since we only required grid station locations which have a 'Node name', we are going to delete rows with no 'Node name' which 
#are mostly cables and powerlines

PTN=PTN.dropna(how='any',axis=0)
PTN=PTN.reset_index()


# In[ ]:


PTN=PTN.drop('index',axis=1)
PTN.shape


# In[ ]:


PTN=PTN.sort_values(by=['Node name']).reset_index().drop('index', axis=1)


# In[ ]:


PTN


# 5. Now lets view the Grid Locations on a map using Folium

# In[ ]:


#Lets get the coordinates of Pakistan using geolocator
address = 'Islamabad, PK'

geolocator = Nominatim(user_agent="pk_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Pakistan are {}, {}.'.format(latitude, longitude))


# In[ ]:


# Lets create map of Pakistan using latitude and longitude values and mark the locations of the Grid stations
map_PTN = folium.Map(location=[latitude,longitude], zoom_start=6)

# add markers to map
for lat, lng, label in zip(PTN['Latitude'], PTN['Longitude'], PTN['Node name']):
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_PTN)  
    
map_PTN


# 6. Now lets use the Foursquare API to get the nearby venues for a single Grid Node e.g IESCO Grid Station

# In[ ]:


CLIENT_ID = 'TQ0EQVSWEW1PZSVXLNC0CPJ1QEXZIYAAH1GGUVKU4IOWS4GP' # your Foursquare ID
CLIENT_SECRET = 'WOAR3RL51SFCKQZ0QZKQ2YVQIUHXXWLAABMAAOC5NZL4USGN' # your Foursquare Secret
VERSION = '20180605' # Foursquare API version

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)


# In[ ]:


index=PTN[PTN['Node name']=='IESCO Grid Station'].index
index


# In[ ]:


Grid_Lat=PTN.loc[37,'Latitude']
Grid_Lng=PTN.loc[37,'Longitude']
print('Coordinates of IESCO Grid Station:',Grid_Lat,Grid_Lng)


# In[ ]:


LIMIT=100
radius=5000
url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            Grid_Lat, 
            Grid_Lng, 
            radius, 
            LIMIT)


# In[ ]:


results = requests.get(url).json()
results


# In[ ]:


results['response']['groups'][0]['items']


# In[ ]:


venues = results['response']['groups'][0]['items']
nearby_venues = json_normalize(venues)


# In[ ]:


nearby_venues


# In[ ]:


def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']


# In[ ]:


filtered_columns = ['venue.name', 'venue.categories', 'venue.location.lat', 'venue.location.lng','venue.location.distance']
nearby_venues =nearby_venues.loc[:, filtered_columns]

# filter the category for each row
nearby_venues['venue.categories'] = nearby_venues.apply(get_category_type, axis=1)

# clean columns
nearby_venues.columns = [col.split(".")[-1] for col in nearby_venues.columns]

nearby_venues


# Excellent! The results give us the number of venues surrounding the grid location, and the distance of each venue from it
# 6. Now lets do this for all Grid Nodes

# In[ ]:


def getNearbyVenues(names, latitudes, longitudes, radius=15000):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        #print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]["groups"][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],
            v['venue']['location']['distance'],
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Node name', 
                  'Node Latitude', 
                  'Node Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude',
                  'Venue distance',
                  'Venue Category']
    
    return(nearby_venues)


# In[ ]:


# We will call this PTN_venues (PTN Stands for Pakistan Transmission Network)
PTN_venues = getNearbyVenues(names=PTN['Node name'],
                                   latitudes=PTN['Latitude'],
                                   longitudes=PTN['Longitude']                                
                                      )


# In[ ]:


PTN_venues


# In[ ]:


PTN_venues.to_csv (r'E:\Data of old laptop\M.Ali\USB Data\CVs\IBM DS\Capstone Project\PTN_venues.csv', index = False, header=True)


# In[ ]:


PTN_venues.shape


# In[ ]:


# Lets visualize the venue locations on the map 

map_PTN = folium.Map(location=[30.3753, 69.3451], zoom_start=5)

# add markers to map
for lat, lng, label in zip(PTN['Latitude'], PTN['Longitude'], PTN['Node name']):
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_PTN) 
    
for lat, lng, label in zip(PTN_venues['Venue Latitude'], PTN_venues['Venue Longitude'], PTN_venues['Venue']):
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='red',
        fill=True,
        fill_color='red',
        fill_opacity=0.7,
        parse_html=False).add_to(map_PTN) 
    
map_PTN


# We can see from the map that the Grid Stations located in the densely populated cities of Pakistan have the greates venue clusters around them. Look at the dense red clusters around Karachi, Lahore and Islamabad!

# In[ ]:





# In[ ]:





# 7. Great! Now that we have a Dataframe with all venue locations and distances from the Grid Station Locations, Lets calculate the number of venues surrounding all GS and the mean distance of venues from the central GS location

# In[ ]:


df_g=PTN_venues.groupby('Node name', as_index=False).count()
df_g


# Now lets calculate the mean distance from venues for a single Grid Station say Rohri

# In[ ]:


Rohri=PTN_venues[PTN_venues['Node name']=='Rohri New']
Rohri_mean=Rohri['Venue distance'].mean()
Rohri['mean distance']=Rohri_mean
Rohri


# Interesting! This means that the mean distance of all the venues from Rohri grid station is 11382 kms out of a total venue radius of 15 kms and the total number of venues in this radius is 4. Lets compare it with the IESCO grid station but bear in mind that this time the venue radius is 15 kms instead of 5 kms. 

# In[ ]:


Islamabad=PTN_venues[PTN_venues['Node name']=='IESCO Grid Station']
Islamabad_mean=Islamabad['Venue distance'].mean()
Islamabad['mean distance']=Islamabad_mean
Islamabad


# We can analyze that although the mean distance from venue in this case is not too different from Rohri, IESCO Grid dtation which distributes Power to Islamabad has 100 venues in the same radius. To give you a comparison of the two cities, please follow the below links
# https://en.wikipedia.org/wiki/Islamabad
# https://en.wikipedia.org/wiki/Rohri

# In[ ]:


# Now lets calculate the mean venue distance for all the Grid Station Nodes
mean_distance=[]

for n in df_g['Node name']:    
    x=PTN_venues[PTN_venues['Node name']==n]
    y=x['Venue distance'].mean()
    
    mean_distance.append(y)


# In[ ]:


mean_distance


# In[ ]:


distance=pd.DataFrame(mean_distance)


# In[ ]:


PTN.sort_values(by=['Node name'], inplace=True, ascending=True)
PTN=PTN.reset_index()


# In[ ]:


distance


# 8. Now lets add the no of venues and mean distance from venues to our main dataframe PTN.\
# As we can see that the df_g which shows number of venues per grid station has only 107 rows. Which means the other 13 Grid station do not have any venues in 15 km radius. We will assign 0 to the remaining GS. Also no venues mean no mean distance from venues. For the sake of analysis, we will assign those nodes the max mean distance from the distance dataframe

# In[ ]:


PTN['No of Venues'] = np.where(PTN['Node name'].isin(df_g['Node name']),'True', 'False')


# In[ ]:


PTN


# In[ ]:


df_T=PTN[PTN['No of Venues']=='True']
df_T=df_T.sort_values(by=['Node name'])


# In[ ]:


df_F=PTN[PTN['No of Venues']=='False']


# In[ ]:


df_T=df_T.reset_index().drop('index',axis=1)


# In[ ]:


df_T=df_T.drop([53,57,86])


# In[ ]:


df_T=df_T.reset_index().drop('index',axis=1)


# In[ ]:


df_T['No of Venues']=df_g['Venue']


# In[ ]:


df_T


# In[ ]:


df_f=df_T.append(df_F, ignore_index=True)


# In[ ]:


df_f=df_f.drop('level_0',axis=1)


# Now we are going to assign the number 0 to all entries that are False

# In[ ]:


df_f['No of Venues'][107:117]=0


# In[ ]:


df_f['Mean distance']=distance


# In[ ]:


df_f


# UhOh! we dont have mean distances for Grid stations from 111 to 116 because they do not have any venues around them. So we are going to assign those nodes the highest mean distance value

# In[ ]:


distance.max()


# In[ ]:


df_f['Mean distance'][107:117]=14593


# In[ ]:


df_f=df_f.sort_values(by=['Node name']).reset_index().drop('index', axis=1)


# In[ ]:


df_f


# Lets plot the mean distance Vs the number of nodes and observe their relationship

# In[ ]:


import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0.5,0.5,2,1])
ax.bar(df_f['No of Venues'],df_f['Mean distance'])
plt.show()


# We can see here that the nodes with lower number of venues surrounding them have higher mean distances from the venues

# 9. Now in the next step we would add the Specific Photovoltaic Power Output for these grid stations. Unfortunately there is no dataset available for these values so I had no choice but to make one by entering the coordinates of the grid stations one by one into Global Solar Atlas and getting their PVOUT values. You can also check out data related to solar radiation such as PVOUT, DNI and GHI values for your hometown by searching it. Foolowing is the link
# https://globalsolaratlas.info/map?r=PAK&c=30.626947,68.99585,5

# All we have to do is get PVOUT values from df_s and assign them to the PVOUT column in df_f

# In[ ]:


df_s=pd.read_csv(r"E:\Data of old laptop\M.Ali\USB Data\CVs\IBM DS\Capstone Project\DF_Solar1.csv")


# In[ ]:


df_f['PVOUT']=df_s['PVOUT']


# In[ ]:


df_f


# Now lets analyze the PVOUT values by plotting them against Latitude and Longitude to see how solar radiation varies accross the geography of the country

# In[ ]:


plt.scatter(df_f['Latitude'], df_f['PVOUT'], alpha=0.7)
plt.xlabel('Latitude', fontsize=18)
plt.ylabel('PVOUT', fontsize=16)

plt.show()


# We can observe that the PVOUT values decrease with increasing latitude. Hence location in the south of Pakistan recieve more sunlight than locations in the north. 

# In[ ]:


plt.scatter(df_f['Latitude'], df_f['No of Venues'], alpha=0.5)
plt.xlabel('Latitude', fontsize=18)
plt.ylabel('Number of Venues', fontsize=16)

plt.show()


# If we plot the Number of Venues against Latitude, we can observe that there is a greater concenration of nodes and their venues in the norther half of the country. This is because the Northern half of Pakistan is more populated than its souther half

# In[ ]:


plt.scatter(df_f['Longitude'], df_f['No of Venues'], alpha=0.5)
plt.xlabel('Number of Venues', fontsize=18)
plt.ylabel('Mean distances', fontsize=16)

plt.show()


# Also if we plot the Number of Venues against the Longitude, we can see that the number of locations and venues are more concentrated along the eastern half of Pakistan

# In[ ]:


plt.scatter(df_f['Longitude'], df_f['PVOUT'], alpha=0.5)
plt.xlabel('PVOUT', fontsize=18)
plt.ylabel('Mean distances', fontsize=16)

plt.show()


# But locations aong the western boundry of Pakistan have higher PVOUT values. Meaning they receive more sunlight!

# 10. Now lets use K-means clustering to group these sites into cluster and observe the resulting clusters 

# Lets drop the categorical variables from the dataframe

# In[ ]:


df_final=df_f.drop(['level_0','Node name','Description','Latitude','Longitude'], axis=1)


# In[ ]:


df_final


# Now let's normalize the dataset. We use StandardScaler() to normalize our dataset.

# In[ ]:


from sklearn.preprocessing import StandardScaler
X = df_final.values[:,1:]
X = np.nan_to_num(X)
Clus_dataSet = StandardScaler().fit_transform(X)
Clus_dataSet


# Lets find the optimum K Value for the model by the elbow method

# In[ ]:


distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(df_final)
    distortions.append(kmeanModel.inertia_)


# In[ ]:


plt.figure(figsize=(16,8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()


# As we can observe that the values of k after 3 has minimal impact on the distortion value, we will keep 3 numbers of clusters for the model.

# Lets apply k-means on our dataset to generate cluster labels, and take look at cluster labels.

# In[ ]:


clusterNum = 3
k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
k_means.fit(X)
labels = k_means.labels_
print(labels)


# In[ ]:


df_final["Clus_km"] = labels
df_final.head(5)


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:



plt.scatter(X[:,0],X[:, 1], c=labels.astype(np.float), alpha=0.5)
plt.xlabel('Distance from Venues', fontsize=18)
plt.ylabel('PVOUT', fontsize=16)

plt.show()


# Lets examine the clusters by getting their mean values
# 

# In[ ]:


df_final[df_final['Clus_km']==0].mean()


# In[ ]:


df_final[df_final['Clus_km']==1].mean()


# In[ ]:


df_final[df_final['Clus_km']==2].mean()


# We can observe that 
# 1. Cluster 0 has better mean PVOUT, low number of venues but lowest mean distance from venues.
# 2. Cluster 1 has lowest mean PVOUT highest number of venues but median mean distances.
# 3. Cluster 2 has highest mean PVOUT, lowest number of venues surrounding the grid stations and hgihest mean distances from the venues.

# In[ ]:


df_f["Clus_km"] = labels


# Now lets plot the custers on a map to see their locations

# In[ ]:


Cluster_0=df_f[df_f["Clus_km"]==0]


# In[ ]:


Cluster_1=df_f[df_f["Clus_km"]==1]


# In[ ]:


Cluster_2=df_f[df_f["Clus_km"]==2]


# In[ ]:


map_0 = folium.Map(location=[latitude,longitude], zoom_start=6)

# add markers to map
for lat, lng, label in zip(Cluster_0['Latitude'], Cluster_0['Longitude'], Cluster_0['Node name']):
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='purple',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_0)  
    
map_0


# In[ ]:


map_1 = folium.Map(location=[latitude,longitude], zoom_start=6)

# add markers to map
for lat, lng, label in zip(Cluster_1['Latitude'], Cluster_1['Longitude'], Cluster_1['Node name']):
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='green',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_1)  
    
map_1


# In[ ]:


map_2 = folium.Map(location=[latitude,longitude], zoom_start=6)

# add markers to map
for lat, lng, label in zip(Cluster_2['Latitude'], Cluster_2['Longitude'], Cluster_2['Node name']):
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='yellow',
        fill=True,
        fill_color='red',
        fill_opacity=0.7,
        parse_html=False).add_to(map_2)  
    
map_2


# In[ ]:




