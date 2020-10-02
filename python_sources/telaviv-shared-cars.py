#!/usr/bin/env python
# coding: utf-8

# The idea behind this kernel is to try the use of network analysis on the locations where cars are parked to identify significant locations and important pairs of locations. <br>
# This information can then be used in secondary analysis to determine if more parking lots should be made available. <p>
# 
# In the first box, the data is imported along with some modules that will be used.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import networkx as nx # for network analysis
#visualization
import matplotlib.pyplot as plt
import folium

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
data = pd.read_csv('../input/sample_table.csv')


# The first thing to do then is to determine all car parking locations. <br>
# Unlike the larger carparks that some people might be used to, some of the carparks here appear to be street parking. <p>
# 
# In the first 2 lines, a sanity check on the location data. <br>
# Then each carpark location is stored as a string in a new column. This is done to make node calling easier during network analysis. <br>
# To make the analysis more readable, some carparks that are near each other are combined by rounding the long and lat values. <br>
# For a long and lat with 3 decimal places, each carpark is located to within 50m accuracy which is well within walking distance for the average person. <br>
# Finally, check the number of unique carpark locations.

# In[ ]:


#first find the number of unique carparks, check that the longs and lats are clean
print('Check that data is clean: Lat has', data['latitude'].isna().sum(), 'NAs and Long has', data['longitude'].isna().sum(), 'NAs.')
#Note: 3113 carparks if round(3), 78 carparks if round(2), 3068 carparks if apply floor
# data[['long','lat']] = data[['longitude','latitude']].apply(lambda x: x*1000).apply(np.floor).apply(lambda x: x/1000)
data[['long','lat']] = data[['longitude','latitude']].round(3)
data['carpark'] = '(' + data['lat'].map(str) + ',' + data['long'].map(str) + ')'
print('Number of unique carparks:', data['carpark'].nunique())


# In[ ]:


#Just to have a quick look at the data
data.head(5)


# In the next step, some locations of interest are determined. <br>
# This includes locations where 4 or more cars can be parked. <br>
# The smaller subset is used to make the analysis easier to read, but this could be extended to more carparks in general, at the expense of performance and human readability.

# In[ ]:


data_subset = data[data['total_cars'] > 3].sort_values(['timestamp']).reset_index(drop = True)

big_carparks = {}

for row in range(len(data_subset)):
    carpark = data_subset['carpark'][row]
    car_count = data_subset['total_cars'][row]
    if carpark in big_carparks.keys():
        if car_count > big_carparks[carpark]:
            big_carparks[carpark] = car_count
    else:
        big_carparks[carpark] = car_count

def color(number):
    if number == 4:
        return 'blue'
    elif number == 5:
        return 'green'
    elif number == 6:
        return 'orange'
    else:
        return 'red'
        
m = folium.Map([32.089, 34.797], zoom_start = 12, width = '40%')

for key in big_carparks.keys():
    key1, key2 = key.lstrip('(').strip(')').split(',')
    folium.Marker([key1,key2], popup = key, 
                  icon = folium.Icon(color = color(big_carparks[key]), icon = 'car', prefix = 'fa')
                 ).add_to(m)

m


# The next step is to collect all the trips that the cars have made.<br>
# What this does is that it makes a count of all the trips made by any car between 2 parking locations. <br>
# This is particularly important because we wish to know which carparks are more important.

# In[ ]:


trips = {}

for row in range(len(data)):
    cars = data['carsList'][row].lstrip('[').strip(']').replace(' ','').split(',')
    if len(cars) < 1:
        continue
    for car in cars:
        if car in trips.keys():
            trips[car].append(data['carpark'][row])
        else:
            trips[car] = []
            trips[car].append(data['carpark'][row])


# Then using that trip information, it is possible to build directed and undirected graphs. <br>
# These graphs yield different information of the network. <br>
# Where an undirected graph would give centrality information of a node, the directed graph adds an extra layer of information, taking into account the the direction and number of trips made between nodes.<br>
# The number of trips made will be used as the edge weight between nodes during analysis.

# In[ ]:


directed_trip = {}
undirected_trip = {}

carpark_links = nx.Graph()
directed_carpark_links = nx.DiGraph()

for car in trips.keys():
    for i in range(1, len(trips[car]), 1):
        start = trips[car][i-1]
        end = trips[car][i]
        if start == end:
            continue
        trip = (start, end)
        if trip in directed_trip.keys():
            directed_trip[trip] += 1
        else:
            directed_trip[trip] = 1
        if start < end:
            tripp = (start, end)
        else:
            tripp = (end, start)
        if tripp in undirected_trip.keys():
            undirected_trip[tripp] += 1
        else:
            undirected_trip[tripp] = 1
        directed_carpark_links.add_edge(start,end, weight = directed_trip[start,end])
        carpark_links.add_edge(start,end, weight = undirected_trip[tripp])


# A subgraph is created using only the large carparks, but all weights were applied according to the actual trips made between carparks.
# Note that this disctinction is important as we do not want to accidentally consider only the paths between big carparks, which would happen if the subset of data was used to build the network graph.

# In[ ]:


carparks = [key for key in big_carparks.keys()]

carpark_links_subgraph = nx.subgraph(carpark_links, carparks)
directed_carpark_links_subgraph = nx.subgraph(directed_carpark_links, carparks)


# Below, a typical networkx type graph is shown. The outlier can quickly be identified and analysed. <br>
# In this case, we may be interested in looking at (32.089, 34.797) and (32.063, 34.796). <br>
# The placement suggests that these are under-used parking locations.

# In[ ]:


plt.figure(figsize = (10,12))
pos_kk = nx.kamada_kawai_layout(carpark_links_subgraph)
nx.draw(carpark_links_subgraph, pos = pos_kk, with_labels = True, label = undirected_trip)


# On a map, the locations are shown in red. <br>
# The reason for their low connectivity is not yet understood, but perhaps some centrality measures will help.

# In[ ]:


m2 = folium.Map([32.089, 34.797], zoom_start = 12, width = '40%')

def underused_color(key):
    underused = ['(32.089,34.797)','(32.063,34.796)']
    if key in underused:
        return 'red'
    else:
        return 'blue'

for key in big_carparks.keys():
    key1, key2 = key.lstrip('(').strip(')').split(',')
    folium.Marker([key1,key2], popup = key, 
                  icon = folium.Icon(color = underused_color(key), icon = 'car', prefix = 'fa')
                 ).add_to(m2)

m2


# Unsurprisingly, the betweeness of these carparks is very low.

# In[ ]:


betweenness_weighted = nx.betweenness_centrality(carpark_links_subgraph, weight = 'weight')
max_betweenness_weighted = 0
maxnode_weighted = 'empty'
min_betweenness_weighted = 1
minnode_weighted = 'empty'
for node in betweenness_weighted.keys():
    if betweenness_weighted[node] > max_betweenness_weighted:
        max_betweenness_weighted = betweenness_weighted[node]
        maxnode_weighted = node
    if betweenness_weighted[node] < min_betweenness_weighted:
        min_betweenness_weighted = betweenness_weighted[node]
        minnode_weighted = node

print('Max:', maxnode_weighted , '&', max_betweenness_weighted)

underused = ['(32.089,34.797)','(32.063,34.796)']
for node in underused:
    print('Betweenness of', node, '=', betweenness_weighted[node])


# Note the effect of edgeweights. If omitted, the expected result will be different as shown by the betweenness scores here. <br>
# The two outliers identified in the network diagram no longer have the lowest betweenness scores.

# In[ ]:


#this is without edgeweights
betweenness = nx.betweenness_centrality(carpark_links_subgraph)
max_betweenness = 0
maxnode = 'empty'
min_betweenness = 1
minnode = 'empty'
for node in betweenness.keys():
    if betweenness[node] > max_betweenness:
        max_betweenness = betweenness[node]
        maxnode = node
    if betweenness[node] < min_betweenness:
        min_betweenness = betweenness[node]
        minnode = node


print('Max:', maxnode, '&', max_betweenness)
print('Min:', minnode, '&', min_betweenness)

for node in underused:
    print('Betweenness of', node, '=', betweenness[node])


# As a final point, we can consider the trips made between carparks to see which carparks are more important and perhaps more carspark spaces need to be provided. <p>
# As shown below, the blue nodes are big carparks which can house 7 or more cars. <br>
# It is surprising though to learn that the most number of trips, denoted by the thickness of the lines, are made between smaller carparks.
# Here the carparks are mapped roughly according to the real location on the map.

# In[ ]:


pos_map = {}
for node in carpark_links_subgraph.nodes:
    node1, node2 = node.lstrip('(').strip(')').split(',')
    pos_map[node] = [float(node2), float(node1)]
    
carpark_links_subgraph_edgelabel = []

for key in carpark_links_subgraph.edges:
    key1, key2 = key
    if key1 > key2:
        key = (key2, key1)
    carpark_links_subgraph_edgelabel.append(undirected_trip[key]/500)
    
big_carparks_of_interest = [key for key in big_carparks.keys() if big_carparks[key] > 6]
node_color = ['b' if key in big_carparks_of_interest else 'r' for key in carpark_links_subgraph.nodes]
    
plt.figure(figsize = (18,18))
nx.draw(carpark_links_subgraph, pos = pos_map, with_labels = True, node_color = node_color, width = carpark_links_subgraph_edgelabel)


# We can then check the betweenness scores of these carparks. Take the top 5 carparks by betweenness and compare that against the betweenness of the big carparks. <br>
# Unsurprisingly, the betweenness score confirms that these carparks do see more traffic.

# In[ ]:


maxnodes_weighted = sorted(betweenness, key=betweenness.get, reverse=True)[:5]
maxnodes_unweighted = sorted(betweenness, key=betweenness.get, reverse=True)[:5]

carparks_of_interest = []
for node in maxnodes_unweighted:
    if node not in carparks_of_interest:
        carparks_of_interest.append(node)
for node in maxnodes_weighted:
    if node not in carparks_of_interest:
        carparks_of_interest.append(node)

all_carparks_of_interest = big_carparks_of_interest + carparks_of_interest
print('Key \t\t\t Max. Cars \t Unweighted \t\t Weighted')
for key in all_carparks_of_interest:
    print(key, '  \t', big_carparks[key], '\t\t', betweenness[key], '\t', betweenness_weighted[key])


# These high connectivity carparks are colored in green.

# In[ ]:


node_color = []

for key in carpark_links_subgraph.nodes:
    if key in big_carparks_of_interest:
        node_color.append('b')
    elif key in maxnodes_unweighted or key in maxnodes_weighted:
        node_color.append('g')
    else:
        node_color.append('r')
    
plt.figure(figsize = (18,18))
nx.draw(carpark_links_subgraph, pos = pos_map, with_labels = True, node_color = node_color, width = carpark_links_subgraph_edgelabel)


# Finally, the highly connected routes can be shown on a map to help figure out why these locations have such high connectivty.
# The top 10 routes of high connectivty are shown.

# In[ ]:


m3 = folium.Map([32.089, 34.797], zoom_start = 12, width = '40%')

def color_node(position):
    if position in big_carparks_of_interest:
        return 'blue'
    elif position in carparks_of_interest:
        return 'green'
    else:
        return 'red'

for key in big_carparks.keys():
    key1, key2 = key.lstrip('(').strip(')').split(',')
    folium.Marker([key1,key2], popup = key + '\n Max.:' + str(big_carparks[key]) + ' cars', 
                  icon = folium.Icon(color = color_node(key))
                 ).add_to(m3)

def check_keys(key1, key2):
    if key1 > key2:
        key = (key2, key1)
    else:
        key = (key1, key2)
    return key

undirected_trip_subset = {}
for key in carpark_links_subgraph.edges:
    key1, key2 = key
    key = check_keys(key1, key2)
    undirected_trip_subset[key] = undirected_trip[key]

high_volume = sorted(undirected_trip_subset, key = undirected_trip_subset.get, reverse = True)[:10]

for edge in high_volume:
    node1, node2 = edge
    point1, point2 = node1.lstrip('(').strip(')').split(',')
    point3, point4 = node2.lstrip('(').strip(')').split(',')
    folium.PolyLine([([float(point1),float(point2)]),([float(point3),float(point4)])],
                    color="black", 
                    weight= 3).add_to(m3)

print('Trip \t\t\t\t\t Count')
for trip in high_volume:
    print(trip, '\t', undirected_trip_subset[trip])
    
m3


# It might then be interesting to figure out why drivers make such short trips (as shown below) with such high frequency.

# In[ ]:


m4 = folium.Map([32.116, 34.84], zoom_start = 15, width = '50%', height = '50%')

shorttrip = [[32.118,34.839],[32.114,34.842]]

for location in shorttrip:
    key1, key2 = location
    key = '('+str(key1)+','+str(key2)+')'
    folium.Marker(location, popup = key + '\n Max.:' + str(big_carparks[key]) + ' cars', 
                  icon = folium.Icon(color = color_node(key))
                 ).add_to(m4)

folium.PolyLine(shorttrip, 
                color="black", 
                weight= 3).add_to(m4)
string = ('(32.114,34.842)','(32.118,34.839)')
print('Trip between', trip, 'was made', undirected_trip_subset[string], 'times.')
m4


# In fact, we can check to show that most trips between carpark pairs only occur les than 1000 times and that anything above 200 is rare (and interesting)

# In[ ]:


plot_highvolume = pd.DataFrame([undirected_trip_subset[key] for key in undirected_trip_subset.keys()])
plot_highvolume.plot.hist('0')

