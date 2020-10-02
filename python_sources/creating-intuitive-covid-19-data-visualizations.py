#!/usr/bin/env python
# coding: utf-8

# # Creating intuitive COVID-19 data visualizations
# Given COVID-19 daily reports on case count and location, it may not be immediately obvious of any patterns present in the data. Which locations/regions are more affected by the virus, and in which places are new cases likely to be appear? What areas should healthcare professionals focus their resources and efforts on? 
# 
# Visualizing this data in an intuitive way provides clearer insights on transmission behavior and patterns. In this project, we used data from Allegheny County, PA. We wanted to focus our efforts on creating a local-level tool at a finer resolution to track coronavirus spread, and using official county data was an excellent way to gather the specific data we needed. Another reason we chose Allegheny County in particular was because we were collaborating with the local county health department and UPMC Emergency Medical Services to better coordinate COVID-19 response.
# 
# Traditional methods of viewing geographic data, such as a choropleth map, are often not specific or visually effective enough for county-level data. Additionally, simply showing one map for each day makes it hard to reason about patterns over time, but animating it also introduces the problem of sudden jumps in color or distribution between frames, since human vision is particularly wired to understand continuous motion. Altering the traditional choropleth, we performed a series of interpolations to more effectively bring out patterns and trends. The result is a cognitively-engaging visualization that makes it ergonomically efficient for humans to observe dynamic patterns in data.

# In[ ]:


from IPython.display import Image
Image(filename='/kaggle/input/images/cbar40.png', width=1000, height=1000)


# ![](https://storage.googleapis.com/kagglesdsdata/datasets/591876/1085258/alleghenyT.gif?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1587272665&Signature=dM2dbhSjcZ5r0jwu%2BywOJHodrMkQB1TzPFJ19CcaFK55%2BTcU3Ps8K3jtCwKn%2BEZ4fZ2xaEA6Z2ophmqlVy9eJHP%2B4KjbeIDLnBVsKHT%2FuDO7zPgZv4GmPNGkpeVHnn8zyKKhuLEJCdN%2BTgARwMLaOdzr1qQl07w9QsL7VpwXBCwy%2FCQPYP7%2FykCOln4rwXw%2FyLRxTPRCwJNlYDzAwPgU8aQoOcbocoHenH%2F40PGN%2BRB36Aarx9R6okSl6iF3Hs5c5GAi4HE1fPDHfZogXHh2nIbNbqVckt%2FV8KUybauak3BkC5kA%2Fwl8AwB%2BTC7WFtOmvk3PYq6%2B2B91UBwDbC6zcA%3D%3D)

# From this animation, compared to one made with a choropleth (shown later) it becomes much easier to see where exactly new cases are appearing and how the case distribution changes over time. We observe various clusters with rising case counts, which intuitively informs the viewer of where COVID-19 is most prevalent in Allegheny County. The temporal interpolation also lends to a more visually appealing and easier to understand animation.

# In[ ]:


Image(filename='/kaggle/input/images/transmission_graph.png', width=500, height=500)


# In addition to creating the map, we also wanted to potentially visualize the pattern of global transmissions through radial directed graphs. For this, we used the open-source Johns Hopkins COVID-19 dataset.
# 
# **Table of Contents**
# * Motivation
# * Loading/Preprocessing data
# * Temporal interpolation
# * Spacial interpolation
# * Visualizing data
# * Creating graphs to further visualize global transmission
# * Conclusion
# * References

# # Motivation
# In order to visualize the distribution of cases in Allegheny County, and to potentially find patterns in spread of the virus, a traditional choropleth is a good start:

# In[ ]:


Image(filename='/kaggle/input/images/Screen Shot 2020-04-15 at 10.04.55 PM.png', width=500, height=500)


# However, this doesn't provide much information. First, the high concentration of cases in Pittsburgh (the municipality in the center) draws attention away from potential information about cases in other regions. In order to fix this, we can try better scaling the data towards the lower values and animating the case counts per day:

# ![](https://storage.googleapis.com/kagglesdsdata/datasets/591876/1085258/movie%20%282%29.gif?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1587272787&Signature=XXJehlnShsdS3xfhbO32wmPV9BygRLmlICgbbePouDVZ%2BrVngK1dfNylOZkPl2reoH2P3fI4wyMhF5WA5rZkX9YEIs0jXz%2BaXV53zjvPPEQlwHjd16scTxvo9tXwshOEuTtUBEwXpeLTSwgworydmfTD4ND%2F3opUgxQYvRGB3RJBKSTsrDWT44x94VFCNjZet5I5vo3VBWUoiJjrHONOLTpGSkAGHMZZClujzNulZaDzhDEQrb19icfy0N6mAAuo41cvU%2B9ym1gd3lU1%2BEF4dMXzc7I1FWvwLELGiZ2645%2FoTTt4K3BPwjxtJ6TA%2FIYDtP%2Bq%2BOY3h0JtW3WZBY3WZA%3D%3D)

# However, this doesn't provide much insight either. The lack of transition between days makes it hard to follow the progression of the virus, and the abrupt color change between municipalities also makes it hard to discern locational information from the map, especially since the size of municipalities in Allegheny County is not uniform, so the color displayed on the map is not consistent with the actual patterns in the data. To solve this, we use interpolation to creater a smoother visualization in time and space.

# # Loading/Preprocessing Data
# We gathered daily data about cases in each municipality by referencing publicly posted data from Allegheny County's official website: https://www.alleghenycounty.us/Health-Department/Resources/COVID-19/COVID-19.aspx

# In[ ]:


# importing necessary packages

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

from math import radians, cos, sin, asin, sqrt, atan2
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pylab import rcParams
import seaborn as sns
import folium
#import spectra
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon
import warnings
import geoplot as gplt
import geoplot.crs as gcrs
from matplotlib import cm
import warnings
warnings.filterwarnings('ignore')


# Reading the shapefile of Allegheny County to map cases onto:

# In[ ]:


#using a shapefile of Allegheny County to map data onto
sf_path = "/kaggle/input/allegheny-county/Allegheny_County_Municipal_Boundaries.shp"
sf = gpd.read_file(sf_path, encoding='utf-8')

#reformatting the shapefile to our preferred coordinate system
coordsf = sf.to_crs({'init': 'epsg:4326'})

#reformatting the geometry of the shapefile for a better format
from shapely.geometry.polygon import Polygon
from shapely.geometry.multipolygon import MultiPolygon
coordsf["geometry"] = [MultiPolygon([feature]) if type(feature) == Polygon     else feature for feature in coordsf["geometry"]]
coordsf.plot()


# Now we load and scale the Allegheny County data. We chose to upscale the data by a constant factor of 5 so that we could work with integers during the interpolation phase.

# In[ ]:


#importing data collected from ACHN's website
map_data = pd.read_csv("/kaggle/input/allegheny-covid-timeseries/alleghenyapril10.csv")
map_data.head()


# In[ ]:


#scaling function
def scale_rows(df, countcol):
    for _, row in df.iterrows():
        for i in range(int(row[countcol])-1):
            # Append this row at the end of the DataFrame
            df = df.append(row)

    # Remove countcol (could do a drop too to do that...)
    notcountcols = [x for x in df.columns if x != countcol]
    df = df[notcountcols]
    # optional: sort it by index
    df.sort_index(inplace=True)
    return df
#set parameters for map
sns.set(context='paper', style='ticks', palette='inferno')
sns.mpl.rc("figure", figsize=(10, 6))
mpl.rcParams['figure.dpi']= 150
sf_path = "/kaggle/input/allegheny-county/Allegheny_County_Municipal_Boundaries.shp"
sf = gpd.read_file(sf_path, encoding='utf-8')

#upscaling data for interpolation - also capping cases in Pittsburgh to bring focus to other areas
for i in range (3, 15):
    for j in range (1, 130):
        map_data.iloc[:, i][j] = map_data.iloc[:, i][j]* 5
for i in range (3, 15):
    map_data.iloc[:, i][127] = 200
    
#reformatting the data for interpolation
county = map_data['county']
lat = map_data['lat']
long = map_data['long']
map_data = map_data.drop(columns = ['county', 'lat', 'long'], axis = 1)
map_data.rename(columns = {'3/27/2020':pd.to_datetime('3/27/2020')}, inplace = True)
map_data.rename(columns = {'3/28/2020':pd.to_datetime('3/28/2020')}, inplace = True)
map_data.rename(columns = {'3/29/2020':pd.to_datetime('3/29/2020')}, inplace = True)
map_data.rename(columns = {'3/30/2020':pd.to_datetime('3/30/2020')}, inplace = True)
for i in range (1, 11):
    map_data.rename(columns = {'4/' + str(i) + '/2020':pd.to_datetime('4/' + str(i) + '/2020')}, inplace = True)
map_data.head()


# # Temporal interpolation
# In order to get a smooth transition in the final animation, we need to create more frames between days so that the motion displayed is continuous. To accomplish this we performed a linear interpolation of 3 columns between each datapoint.

# In[ ]:


#interpolating
upsampled = map_data.resample(rule = '5H', axis =1).mean()
upsampled2 = upsampled.interpolate(method = 'linear', axis = 1)
upsampled2 = upsampled2.round(0)
upsampled2.insert(0, "county", county, True) 
upsampled2.insert(1, "lat", lat, True) 
upsampled2.insert(2, "long", long, True) 
upsampled2.head()


# # Spacial interpolation
# One way of visualizing our temporally interpolated data on a map is to use choropleths, but as mentioned before the stark color change between boundaries makes patterns hard to detect, especially since Allegheny County is composed of rather large municipalities. A dot map doesn't provide many benefits either since our data only contains information about each municipality, so all the dots would be clustered in predictable locations. A heat map is closer to what we want to achieve, but since it is only based on the raw distribution of the points, also has the problem of the color change being too abrupt and introduces potential bias from the clustering of the municipalities. We find that projecting a bivariate kernel density plot onto the map yields desriable results.
# 
# At a high level, our goal was to take the categorical information on the left, and merge it with the geographic information on the right:

# In[ ]:


Image(filename='/kaggle/input/images/Screen Shot 2020-04-16 at 12.52.26 AM.png', width=500, height=500)


# # Visualizing data
# In order to visualize our data on a map, we set a point in each municipality as a geographical reference using its latitude and longitude. For each municipality point, we gave it a weight representing its number of cases on a given day. Thus in the final image the color in each location originates from the relative weighted average, described by the kernel density estimation, of the points surrounding it.

# In[ ]:


#for i in range (3, 71): normally, we use this loop to procedurally generate frames for the animation.
    i = 70 #for demonstration purposes, we'll render the final frame
    
    #doing some geographical preprocessing
    column_name = upsampled2.columns[i]
    upsampled3 = upsampled2[['county', column_name, 'lat', 'long']]
    upsampled3[upsampled3.columns[1]] = upsampled3[upsampled3.columns[1]].astype('int64')
    #weighting the points
    mp = scale_rows(upsampled3, upsampled3.columns[1])
    map_data_new = mp
    #map_data_new = upsampled3
    
    #joining the quantitative data with the shapefile to plot it
    covid_data2 = sf.set_index('LABEL').join(map_data_new.set_index('county'))
    #more preprocessing ...
    covid_data2['long'] = -1 * covid_data2['long']
    geo = [Point(xy) for xy in zip(covid_data2['long'], covid_data2['lat'])]
    covid_data2['geometry'] = geo
    covid_data2 = covid_data2[['geometry', 'lat', 'long']]
    lat = covid_data2['lat']
    long = covid_data2['long']
    covid_data2 = covid_data2.to_crs({'init': 'epsg:4326'})
    covid_data2['lat'] = lat
    covid_data2['long'] = long
    covid_data2.geometry = [Point(xy) for xy in zip(covid_data2['long'], covid_data2['lat'])]
    covid_data2 = covid_data2[['geometry']]

    #setting the projection for the map, using Allegheny County's coordinates
    proj = gcrs.AlbersEqualArea(central_latitude=40.4451, central_longitude=-80.0088)

    fig = plt.figure(figsize=(10,5))
    ax1 = plt.subplot(121, projection=proj)

    gplt.kdeplot(covid_data2, clip=coordsf.simplify(0.001), levels = 30, cmap='jet',projection=gcrs.AlbersEqualArea(),
                   shade=True, shade_lowest=True,ax=ax1)
    gplt.polyplot(coordsf, edgecolor = 'white', zorder=1, ax=ax1)
    ax1.annotate(column_name.strftime('%m/%d'), xy=(0.2, .2), xycoords='figure fraction', fontsize=15, color='#555555')
    
    ax1.set_title("Allegheny Count COVID-19 dist.")
    norm = mpl.colors.Normalize(vmin=-1,vmax=1)
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap='jet'),ticks=[-1, -.66, -.33, 0, .33, .66, 1],ax=ax1, shrink= .6)
    
    #we choose 50 as an upper bound because all municipalities (except Pittsburgh) remain <50 cases
    cbar.ax.set_yticklabels(['0', '3', '5', '10', '15' ,'20', '50+'])  
    plt.show()


# In[ ]:


#normally we use this to put together frames of the animation
#import imageio
#images = []
#for i in range (3, 71):
#    images.append(imageio.imread('/kaggle/working/cbar' + str(i) + '.png'))
#imageio.mimsave('cbar.gif', images, duration = 0.1)


# # Creating graphs to further visualize global transmission
# In addition to visualizing COVID-19 on the local level, we'd also like to find insights related to global transmission. The Johns Hopkins COVID-19 dataset includes earlier recorded cases and documents places that patients have traveled from in the recent weeks. Using this information, we can create a forced directed graph of the cases recorded.

# In[ ]:


#installing some modules to help us text-mine by geographic keywords
get_ipython().system('pip install geopy')
get_ipython().system('pip install geotext')
from geotext import GeoText
from geopy.geocoders import Nominatim


# We look at the location of each case, as well as its corresponding summary. In the summary, sometimes travel is recorded, so if we see a geographic keyword such as "Wuhan," we will have that location represent the origin of transmission.

# In[ ]:


#importing JHU dataset
travel_data = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv")
travel_data = travel_data[['location', 'summary']]
travel_data = travel_data.dropna().reset_index(drop=True)
#travel_data = travel_data.reset_index(drop=True)
travel_data2 = travel_data

#text mining for names of cities
from geotext import GeoText
for i in range (0, 1000):
    travel_data2['summary'][i] = GeoText((travel_data2['summary'][i])).cities
travel_data2.head()

#pruning data
for i in range (0, 1000):
    if travel_data2['summary'][i] == []:
        travel_data2 = travel_data2.drop([i])

pd.set_option('display.max_rows', None)

#more reformatting
travel_data2 = travel_data2.reset_index(drop=True)
travel_data3 = travel_data2
travel_data4 = travel_data3[['location', 'summary']]
#travel_data4 = travel_data4.dropna().reset_index(drop=True)

#from the list of cities and visual inspection, we find that the second
#  city in the list is most often the origin location in the JHU data
for i in range (0, 500):
    if len(travel_data4['summary'][i]) < 2:
        travel_data4['summary'][i] = travel_data4['summary'][i][0]
    else:
        travel_data4['summary'][i] = travel_data4['summary'][i][1]
        
travel_data4  = travel_data4.head(500)
travel_data4.head(5)


# With our formatted data, now we can create a simple graph of the earliest 50 cases in the JHU dataset:

# In[ ]:


import matplotlib.pyplot as plt
import networkx as nx

G=nx.Graph()

#adding cities as nodes
for i in range (0, 50):
    G.add_node(travel_data4['location'][i])
    
#adding edges from case data
for i in range (0, 50):
    edge = (travel_data4['location'][i], travel_data4['summary'][i])
    G.add_edge(*edge)

#using nx.spring_layout for a radial graph
pos = nx.spring_layout(G)
colors = range(2)
nx.draw(G, pos=pos, node_size = 50,
        with_labels = True, font_size = 10, k=0.3*1/np.sqrt(len(G.nodes())),iterations=50, node_color='#EDEDED'
       )


plt.figure(3,figsize=(120,120)) 


# From the graph above, it is clear that Wuhan is the epicenter. We can also start to distinguish some other hotspots such as Sichuan, as well as other areas where cases are beginning to spread. 
# 
# We can go further and analyze the first 500 cases, but it starts to become difficult to differentiate between nodes:

# In[ ]:


import matplotlib.pyplot as plt
import networkx as nx

G=nx.Graph()
#same as above, but we did a bit of parsing to get rid of self-loops
for i in range (0, 500):
    if (travel_data4['location'][i] == travel_data4['summary'][i]):
        pass
    else:
        G.add_node(travel_data4['location'][i])
for i in range (0, 500):
    if (travel_data4['location'][i] == travel_data4['summary'][i]):
        pass
    else:
        edge = (travel_data4['location'][i], travel_data4['summary'][i])
        G.add_edge(*edge)


pos = nx.spring_layout(G)
colors = range(2)
nx.draw(G, pos=pos, node_size = 50,
        with_labels = True, font_size = 10, k=1,iterations=50, node_color='#EDEDED'
       )

plt.figure(3,figsize=(120,120)) 


# In[ ]:


nx.draw(G, pos=pos, node_size = 50,
        with_labels = False, font_size = 10, k=1,iterations=50, node_color='blue'
       )


# While not very visually effective, one can still make out a few patterns. One, we see a further spread of cases from the epicenter in Wuhan. Secondly, we can start to see cases in further regions such as Europe. Lastly, we can start to see the cluster on the lower left becoming another group of locations where cases are spreading, and rather predictably Barcelona is a place a lot of transmissions seem to be tied to.
# 
# Moving forward, our goal is to expand the size of the graph to capture as much data as possible to find patterns in global transmission while also retaining the viewability of the graph. We also look to animate the graph vs. time using the dates of each case to visualize the spread of the virus within the nodes, as described in [this article](https://www.citylab.com/transportation/2013/02/weve-been-looking-spread-global-pandemics-all-wrong/4782/).

# # Conclusion
# Given a set of data on COVID-19 cases in a specific U.S. county, we needed a way to display the data in both an informative and digestible format. We found that ordinary methods of visualizing geographic data were not effective, as Allegheny County didn't have the level of geographic detail that higher level maps such as the United States have, which made it difficult to pinpoint where cases were occuring. By utilizing interpolations, we were able to make our data much more apparent in the final result, allowing for smoother and more detailed visualizations that are tailored for humans to understand.
# 
# Looking to the future, we plan on making this and our other available COVID-19 work with Allegheny County open-source by publishing it on GitHub. We also plan on further developing the radial graph mentioned here, adding more detail and animation to visualize transmission on a global scale.
# 
# We hope that our efforts in developing this type of map make it easier in the future to better understand and predict where COVID-19 transmission and cases are most prevalent, as well as better coordinate healthcare response by providing geographic information.  

# # References
# * [Allegheny Health Department](https://www.alleghenycounty.us/Health-Department/Resources/COVID-19/COVID-19.aspx)
# * [Johns Hopkins Dataset](https://www.kaggle.com/sudalairajkumar/novel-corona-virus-2019-dataset)
# * [Spacial and temporal interpolation](http://web.eecs.utk.edu/~audris/papers/jasis.pdf)
# * [Bivariate kernel density estimation](https://bookdown.org/egarpor/NP-UC3M/kde-ii-mult.html)
# * [Graphing epidemics](https://www.citylab.com/transportation/2013/02/weve-been-looking-spread-global-pandemics-all-wrong/4782/)
# * [Guidelines on reporting coronavirus](https://ksj.mit.edu/coronavirus-reporting-resources/)

# If you've read this far, thank you and please let us know your thoughts on our work - we'd love to hear your feedback!
