#!/usr/bin/env python
# coding: utf-8

# 
# <center> <H1> Restaurant Location Recommendation Using Neighborhood Clustering </H1></center>

# ## Table of contents
# * [Introduction: Business Problem](#introduction)
# * [Data Acquisition and preperation](#data)
# * [Clustering and Analysis](#analysis)
# * [Recommendation](#recommendation)
# * [Results and Discussion](#results)

# <h3>Introduction: Business Problem</h3><a name='introduction'></a>
# <br>
# In ths project will try to find an optimal location for a restaurant. Specifically, this report will be targeted to stakeholders interested in opening an Indian Cusine restaurant in Delhi, India. Finding a suitable location for restaurants in major cities like delhi proves to be a daunting task. Various factors such as over-saturation or no demand ,for the type of restaurant that the customer wants to open, effect the success or failure of the restaurant. Hence, customers can bolster their decisions using the descriptive and predictive capabilites of data science.
# <br><br>
# We need to find locations(Neighborhood) that have a <b>potentially unfulfilled demand</b> for Indian Restaurant. Also, we need locations that have <b>low competition and are not already crowded</b>. We would also prefer location as close to popular city Neighborhood, assuming the first two conditions are met.
# <br><br>
# We will use our data science powers to generate a few most promissing neighborhoods based on this criteira. Advantages of each area will then be clearly Expressed so that best possible final location can be chosen by stakeholders.
# <br>
# <br>
# <br>

# <h2>Data Acquisition and preperation</h2><a name='data'></a>

# Based on definition of our problem, factors that will influence our decission are:
# * number of existing restaurants in the neighborhood (any type of restaurant)
# * number of and distance to Indian restaurants in the neighborhood, if any
# * distance of neighborhood from popular neighborhoods
# 
# In our project we will:
# * acquire the names and boroughs of the neighborhoods by scrapping a wikipedia page.
# * After we have got the names of all the neighborhoods, we will geocode them using the library geopy.geocoder (Nominatim).
# * Next, we use the foursquare API to find all types of restaurants within a 1000 meter radius for every neighborhood.
# 
# 

# In[ ]:


import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
from geopy.geocoders import Nominatim
import folium
import re
import json
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe
from sklearn.cluster import KMeans
# Matplotlib and associated plotting modules
import matplotlib.cm as cm

import matplotlib.colors as colors


# In[ ]:


# address = 'New Delhi'
# geolocator = Nominatim(user_agent="ny_explorer")
# location = geolocator.geocode(address)
# latitude = location.latitude
# longitude = location.longitude
# print('The geograpical coordinate of Toronto are {}, {}.'.format(latitude, longitude))


# In[ ]:


latitude = 28.6141793
longitude = 77.2022662


# ### Scrapping the Wikipedia Page
# 
# 
# This section is commented out due to geopy api not working. The code is just fine otherwise. The dataset used in this notebook was made by this code

# In[ ]:


#accessing the web page by Http request made by requests library
# req = requests.get("https://en.wikipedia.org/wiki/Neighbourhoods_of_Delhi").text
# soup = BeautifulSoup(req, 'lxml')
# div = soup.find('div', class_="mw-parser-output" )
# print("web Page Imported")


# In[ ]:


# #Code to extract the relevent data from the request object using beautiful soup
# data = pd.DataFrame(columns=['Borough','Neighborhood'])
# i=-1
# flag = False
# no=0
# prev_borough = None
# for child in div.children:
#     if child.name:
#         span = child.find('span')
#         if span!=-1 and span is not None:
#             try:
#                 if span['class'][0] == 'mw-headline' and child.a.text!='edit':
#                     prev_borough = child.a.text
#                     i+=1
#                     flag = True
#                     continue
#             except KeyError:
#                 continue
#         if child.name=='ul' and flag==True:
#             neighborhood = []
#             for ch in child.children:
                
#                 try:
#                     data.loc[no]=[prev_borough,ch.text]
#                     no+=1
#                 except AttributeError:
#                     pass
#         flag = False
# data[50:60]


# ### geocoding every neighborhood

# In[ ]:


# lat_lng = pd.DataFrame(columns=['latitude','longitude'])
# geolocator = Nominatim(user_agent="ny_explorer")
# for i in range(184):
#     address = data['Neighborhood'].loc[i]+',New Delhi'
#     try: 
#         location = geolocator.geocode(address)
#         lat_lng.loc[i]=[location.latitude,location.longitude]
#     except AttributeError:
#         continue


# In[ ]:


# df1 = data
# df2 = lat_lng
# delhi_neighbourhood_data = pd.concat([df1, df2], axis=1)
# delhi_neighbourhood_data.to_csv(r'E:\jupyter\Coursera Practice\delhi_dataSet.csv')


# In[ ]:


delhi_neighborhood_data = pd.read_csv(r'../input/delhi_dataSet.csv')
delhi_neighborhood_data.dropna(inplace=True)
delhi_neighborhood_data.reset_index(inplace=True)
delhi_neighborhood_data.drop(['index','Unnamed: 0'], axis=1, inplace=True)
delhi_neighborhood_data.head()


# ### Visualing the obtained data set

# In[ ]:


delhiData = delhi_neighborhood_data
map_delhi = folium.Map(location=[latitude, longitude], zoom_start=11)

# add markers to map
for lat, lng, label in zip(delhiData['latitude'], delhiData['longitude'], delhiData['Neighborhood']):
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_delhi)  
    
map_delhi


# Now that we have obtained the location of every neighborhood we can use the foursquare API to find the location of nearby restaurant

# <h3> Foursquare </h3>
# 
# 'The Foursquare Places API provides location based experiences with diverse information about venues, users, photos, and check-ins. The API supports real time access to places, Snap-to-Place that assigns users to specific locations, and Geo-tag.'(wikipedia)
# 
# Here we are using the explore api call and filtering the search only to find venues that are identified as restaurants.

# In[ ]:


# CLIENT_ID = 'PVEHZMCGQRW1UTUDAKHLC0RTRNC205YZ2NJDZDPPJOHQV5VH' # your Foursquare ID
# CLIENT_SECRET = 'XYAYEPCDCHKUT44EMD25OADY1UADBPQZEGVYH0IJRDEWKW1Q' # your Foursquare Secret
# VERSION = '20180605' # Foursquare API version
# radius = 1000
# LIMIT = 200

# print('Credentails Registered')


# In[ ]:


# def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
#     venues_list=[]
#     for name, lat, lng in zip(names, latitudes, longitudes):
#         print(name)
            
#         # create the API request URL
#         url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&categoryId={}&ll={},{}&radius={}&limit={}'.format(
#             CLIENT_ID, 
#             CLIENT_SECRET, 
#             VERSION,
#             '4d4b7105d754a06374d81259',
#             lat, 
#             lng, 
#             radius, 
#             LIMIT)
            
#         # make the GET request
#         try:
#             results = requests.get(url).json()["response"]['groups'][0]['items']
#         except KeyError:
#             continue
        
#         # return only relevant information for each nearby venue
#         venues_list.append([(
#             name, 
#             lat, 
#             lng, 
#             v['venue']['name'], 
#             v['venue']['location']['lat'], 
#             v['venue']['location']['lng'],  
#             v['venue']['categories'][0]['name']) for v in results])

#     nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
#     nearby_venues.columns = ['Neighborhood', 
#                   'Neighborhood Latitude', 
#                   'Neighborhood Longitude', 
#                   'Venue', 
#                   'Venue Latitude', 
#                   'Venue Longitude', 
#                   'Venue Category']
    
#     return(nearby_venues)


# In[ ]:


# delhi_venues = getNearbyVenues(names=delhiData['Neighborhood'],
#                                    latitudes=delhiData['latitude'],
#                                    longitudes=delhiData['longitude']
#                                   )


# In[ ]:


delhi_venues = pd.read_csv(r'../input/restaurant_dataSet.csv')


# In[ ]:


map_res = folium.Map(location=[latitude, longitude], zoom_start=11)

# add markers to map
for lat, lng, label in zip(delhi_venues['Venue Latitude'], delhi_venues['Venue Longitude'], delhi_venues['Venue']):
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=2,
        popup=label,
        color='red',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_res)  
    
map_res


# ### Summary
# * We have, as a result, generated to data Sets.
# * The first was the data set(delhi_data) that contained the borough, name, Latitude and Longitude of all the major Neighborhoods of Delhi
# * And, the second data set(delhi_venues) contained the geographical information pertinent to all the major restaurants in delhi
# <br><br>

# <h2>Clustering and Analysis</h2>
# <a name='analysis'></a>
# <br>
# Our goal here is to find the neighborhoods with low density of Indian restaurants. But, how will we decide which neighborhoods, currently operating on minimal number of Indian restaurants, have the potential for growth and which neighborhoods do not.
# <br><br>
# The most intuitive idea would be to find neighborhoods that have similar patterns of restaurant trends.
# <br><br>
# This can be achived by clustering the neighborhoods of the basis of the restaurant data we have acquired. Clustering is a predominant algorithm of unsupervised Machine Learning. It is used to segregate data entries in cluster depending of the similarity of their attributes, calculated by using the simple formula of euclidian distance.
# <br><br>
# We can then analyze these clusters separately and use those clusters that show high trends of Indian Restaurants

# ### Normalization of the data for clustering

# In[ ]:


# one hot encoding
delhi_onehot = pd.get_dummies(delhi_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
delhi_onehot['Neighborhood'] = delhi_venues['Neighborhood'] 

# move neighborhood column to the first column
fixed_columns = [delhi_onehot.columns[-1]] + list(delhi_onehot.columns[:-1])
delhi_onehot = delhi_onehot[fixed_columns]

delhi_onehot.head()


# In[ ]:


delhi_onehot.shape


# In[ ]:


#To be used while Generating Graphs
delhi_grouped = delhi_onehot.groupby('Neighborhood').mean().reset_index()
delhi_grouped.head()


# In[ ]:


for i in delhi_grouped.columns:
    print(i,end=", ")


# In[ ]:


delhi_grouped.shape


# In[ ]:


def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# In[ ]:


num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = delhi_grouped['Neighborhood']

for ind in np.arange(delhi_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(delhi_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()


# ### Applying the clustering algorithm

# In[ ]:


# set number of clusters
kclusters = 5

delhi_grouped_clustering = delhi_grouped.drop('Neighborhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(delhi_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10]


# In[ ]:


# add clustering labels
neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

delhi_merged = delhiData

# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
delhi_merged = delhi_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighborhood')

delhi_merged.dropna(inplace=True)
delhi_merged.head() # check the last columns!


# ### Cluster Visualization

# In[ ]:


# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(delhi_merged['latitude'], delhi_merged['longitude'], delhi_merged['Neighborhood'], delhi_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[int(cluster)-1],
        fill=True,
        fill_color=rainbow[int(cluster)-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# In[ ]:


clusterdata = pd.merge(delhi_onehot.groupby('Neighborhood').sum(),delhi_merged[['Neighborhood','Cluster Labels']],left_on='Neighborhood', right_on='Neighborhood',how='inner')
clusterdata = clusterdata.iloc[:,1:].groupby('Cluster Labels').sum().transpose()
clusterdata.head()


# ### Analyzing the Clusters

# In[ ]:


import seaborn as sns


# In[ ]:


def plot_bar(clusternumber):
    sns.set(style="whitegrid",rc={'figure.figsize':(20,10)})
    df = clusterdata[[clusternumber]].drop(clusterdata[[clusternumber]][clusterdata[clusternumber]==0].index)
    chart = sns.barplot(x=df.index, y=clusternumber, data=df)
    chart.set_xticklabels(chart.get_xticklabels(),rotation=90)


# In[ ]:


plot_bar(0)


# In[ ]:


plot_bar(1)


# In[ ]:


plot_bar(2)


# In[ ]:


plot_bar(3)


# In[ ]:


plot_bar(4)


# Analysing the bar graphs we can clearly see that <b>clusters 1 and 2</b> have a high demand for Indian Restaurants

# ## Recommendation
# <a name='recommendation'></a>

# ### In this section:
# * we will, first, analyze the density of the Indian Restaurants in generally for each neighborhood.
# * Then we will weed out the neighborhoods that in the highest 70 percentile of density
# * Find out the most popular neighborhoods
# * Will then try to find remaining neighborhoods that are close to them
# * Provide the a detailed recommendation of top 10 neighborhoods
# 
# 
# <br>
# Now, as clusters 1 and 2 have a maximum number of Indian Restaurants, we will focus our search on neighborhoods within these two clusters.
# 
# ### Why?
# We know that when we were clustering the neighborhoods the data used contained the mean of all types of restaurants present in the particular neighborhood. Therefore, we can say that the neighborhoods are clustered on their restaurant trends.<br>
# <br>
# Now, clusters 2 and 3 may collectively have the highest number of indian restaurant but there will be some neighborhoods in these clusters which would have a demand for Indian Restaurants, as these neighborhoods are in the same cluster, but would not have enough supply.
# 

# In[ ]:


delhi_venues.drop('Unnamed: 0',axis=1,inplace=True)


# In[ ]:


forheatmap=delhi_venues.copy()
forheatmap=pd.merge(forheatmap,delhi_merged[['Neighborhood','Cluster Labels']],left_on='Neighborhood', right_on='Neighborhood',how='inner')
forheatmap.drop(forheatmap[~forheatmap['Cluster Labels'].isin([1,2])].index, inplace=True)


# In[ ]:


forheatmap.head()


# In[ ]:


from folium.plugins import HeatMap


# In[ ]:


#heat map of all restaurants in selected Neighborhoods
res_heat = folium.Map(location=[latitude, longitude], zoom_start=11)
HeatMap(list(zip(forheatmap['Venue Latitude'],forheatmap['Venue Longitude'])),
        min_opacity=0.2,
        radius=10, blur=15,
        max_zoom=1
       ).add_to(res_heat)
for lat, lng, label in zip(forheatmap['Neighborhood Latitude'], forheatmap['Neighborhood Longitude'], forheatmap['Neighborhood']):
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=2,
        popup=label,
        color='red',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(res_heat)
res_heat


# In[ ]:


forindres = forheatmap[forheatmap['Venue Category']=='Indian Restaurant']

# heat map for Indian Restaurants in the selected Neighborhoods
res_heat_ind = folium.Map(location=[latitude, longitude], zoom_start=11)
HeatMap(list(zip(forindres['Venue Latitude'],forindres['Venue Longitude'])),
        min_opacity=0.2,
        radius=10, blur=15,
        max_zoom=1
       ).add_to(res_heat_ind)
for lat, lng, label in zip(forindres['Neighborhood Latitude'], forindres['Neighborhood Longitude'], forindres['Neighborhood']):
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=2,
        popup=label,
        color='red',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(res_heat_ind)
res_heat_ind


# now we will remove all neighborhoods with the following conditions:
# * Number of Indian restaurants >30%
# * Number of all restaurants >60% 
# <br>
# 
# '%' here refers to percentile

# In[ ]:


count_all = forheatmap[['Neighborhood','Venue']].groupby('Neighborhood').count().sort_values(by='Venue')
target_count = int(0.6*len(count_all))
print(count_all.iloc[target_count])
count_all.drop(count_all[count_all.Venue.values>7].index,inplace=True)
count_all.columns=['all count']
count_all.head()


# In[ ]:


count_ind = forheatmap[forheatmap['Venue Category']=="Indian Restaurant"][['Neighborhood','Venue']].groupby('Neighborhood').count().sort_values(by='Venue')
target_count = int(0.3*len(count_ind))
print(count_ind.iloc[target_count])
count_ind.drop(count_ind[count_ind.Venue.values>1].index,inplace=True)
count_ind.columns = ['ind count']
count_ind.head()


# In[ ]:


lowdensity = count_all.join(count_ind)
lowdensity.index.values


# In[ ]:


temp_recommend = delhiData.copy()
temp_recommend.drop(temp_recommend[~temp_recommend['Neighborhood'].isin(lowdensity.index.values)].index, inplace=True)
temp_recommend.head()


# Now, we will add the last constraint i.e the neighborhood should be close to popular neighborhoods

# In[ ]:


#most popular neighborhoods
top_nei = delhi_venues[['Neighborhood','Venue']].groupby('Neighborhood').count().sort_values(by='Venue', ascending=False).head(3).index.values
top_nei


# In[ ]:


toplatlng = delhiData[['Neighborhood','latitude','longitude']][delhiData['Neighborhood'].isin(top_nei)].reset_index()
toplatlng


# In[ ]:


from math import sin, cos, sqrt, atan2, radians

def distanceInKM(la1,lo1,la2,lo2):
    # approximate radius of earth in km
    R = 6373.0
    
    lat1 = radians(la1)
    lon1 = radians(lo1)
    lat2 = radians(la2)
    lon2 = radians(lo2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    dis = R * c
    return round(dis,4)

print("Result:", distanceInKM(toplatlng.iloc[2]['latitude'],toplatlng.iloc[2]['longitude'],toplatlng.iloc[0]['latitude'],toplatlng.iloc[0]['longitude']))


# In[ ]:


temp_recommend.reset_index(inplace=True)


# In[ ]:


temp_recommend.drop(columns=['index','Borough'], inplace=True)


# In[ ]:


temp_recommend.head()


# In[ ]:


for i in toplatlng.index:
    temp_recommend[toplatlng.iloc[i]['Neighborhood']] = temp_recommend.apply(lambda x : distanceInKM(toplatlng.iloc[i]['latitude'],toplatlng.iloc[i]['longitude'],x['latitude'],x['longitude']),axis=1)


# In[ ]:


temp_recommend.head()


# In[ ]:


# top 5 neighborhoods near Connaught Place
neiNearCP = temp_recommend.sort_values(by=['Connaught Place']).iloc[:,:3].head().set_index('Neighborhood')
neiNearCP


# In[ ]:


# top 5 neighborhoods near Hauz Khas Village
neiNearHK = temp_recommend.sort_values(by=['Hauz Khas Village']).iloc[:,:3].head().set_index('Neighborhood')
neiNearHK


# In[ ]:


# top 5 neighborhoods near Khirki Village
neiNearKV = temp_recommend.sort_values(by=['Khirki Village']).iloc[:,:3].head().set_index('Neighborhood')
neiNearKV


# In[ ]:


final_recommend=neiNearCP.append(neiNearHK).append(neiNearKV).reset_index()
final_recommend.drop_duplicates(inplace=True)
final_recommend.reset_index(inplace=True)
final_recommend.drop(columns=['index'],inplace=True)
final_recommend


# ## Final Recommendation

# In[ ]:


final = folium.Map(location=[latitude, longitude], zoom_start=11)

# add markers to map
for lat, lng, label in zip(final_recommend['latitude'], final_recommend['longitude'], final_recommend['Neighborhood']):
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(final)  
    
final


# ## Results and Discussions
# <a name='results'></a>

# Our Analysis was done on over 186 neighborhoods, containing over 848 restaurants within 2km radius of every neighborhood. We segragated these neighborhoods on the basis of types and amounts of restaurants. Five clusters were obtained, each having a unique collection of restaurants. Since, we were focused on finding optimal neighborhoods for opening Indian restaurants, we selected cluster 2 and 3 which had the highest number of Indian restaurants. The above actions left us with the only those neighborhoods that had a shared characteristics of and that had a high demand for Indian restaurants.
# <br><br>
# Next, we plotted a heat map for analysing the density of restaurants in the remaining neighborhoods. This allowed us to select neighborhoods that had few or no Indian restaurants and were not overcrowded by other kinds of restaurants. A total of 57 neighborhoods were left. After this, we found out the top three most popular neighborhoods(namely: Canaught Place, Hauz khas Village and Khirki Village), and the distance of every remaining neighborhoods from all three of them. Then, we extracted top 5 closest neighborhoods from each of three most popular neighborhoods mentioned above. Taking the union of the resulting three dataset we get 11 neighborhoods that satisfy all three conditons layed out in the business problem by the customer.
# <br><br>
# The neighborhoods recommendation obtained here are not completely accurate. This is due to the limitations in the dataset used in the project. Due to lack of cross referencing sources, we may have missed a few neighborhoods from our consideration. The foursquare API does not contain, or does not rely, a comprehensive dataset about the restaurants present in delhi. Surely, in a city like Delhi with a population of over 19 million, there are much more restaurants than 848.

# In[ ]:





# In[ ]:


import pandas as pd
delhi_dataSet = pd.read_csv("../input/delhi_dataSet.csv")
restaurant_dataSet = pd.read_csv("../input/restaurant_dataSet.csv")

