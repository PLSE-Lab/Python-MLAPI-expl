#!/usr/bin/env python
# coding: utf-8

# # Plotting Areas/Polygons on a Map: Milanese Neighbourhoods
# 
# ## What is this for?
# 
# Do you want to plot polygons on a map (usually Countries or their subdivisions)? You are on the right track. This notebook deals exactly with that with an easy to understand example.
# We will deal with Milanese neighbourhoods, plot them on a Folium map using a choropleth, and we will add markers at the approximate centre of each neighbourhood to display their names. You can even save the generated map as an html file and place it wherever you want. Neat, right?
# 
# ## Dataset for this project
# 
# You can find the GeoJSON file with the Milanese neighbourhoods [Here](http://dati.comune.milano.it/it/dataset/ds61_infogeo_nil_localizzazione_/resource/af78bd3f-ea45-403a-8882-91cca05087f0).
# 
# ## Libraries for this project
# 
# Just a handful in this case: **folium** for the map part of this map project (it is that important for this Notebook), **shapely** to find the _representative point_ of each neighbourhood (a centre which is not really a centre but we like it because it's a good enough impostor and computationally not so difficult to calculate), and **json** to load and save the various GeoJSON files (yes, GEOjson, for they contain coordinates and other information). We are lucky, as the GeoJSON file that we are going to use no longer adopts the **EPSG:32632 WGS 84 / UTM zone 32N** as in previous versions (I had to convert its coordinates from that standard to latitude and longitude for a similar project), even though they are actually swapped: it's longitude and latitude, instead of the more common opposite. We need to swap them, or else all that we will be doing is going to end up somewhere in Somalia.
# 
# ## Functions created for the project
# 
# **swap_geojson_coordinates** : it swaps the coordinates and creates a new GeoJSON file with said swapped coordinates.
# 
# **create_neighbourhoods_dictionary** : it opens a GeoJSON file and searches for \['properties'\]\['NIL'\] (it's very likely to be different in other GeoJSON files, maybe \['properties'\]\['province'\], or \['properties'\]\['country'\]) to generate a dictionary with the neighbourhood name as the key, and the coordinates as the corresponding value.
# 
# **create_neighbourhood_centres_dictionary**: we feed this function the previously generated dictionary of neighbourhoods. It will find the approximate centres of each neighbourhood to then return a dictionary where said centres are the values, and the neighbourhood names are the keys.

# In[ ]:


import folium
from folium.features import Choropleth
from shapely.geometry import Polygon, Point, mapping
import json


# In[ ]:


# Hats off to folium/leaflet for making this necessary, as the Milanese neighbourhoods plotted using the choropleth otherwise will end up quite literally in Somalia.
def swap_geojson_coordinates(original_filename, new_filename):
    # We will open the original file
    with open(original_filename, 'r') as f:
        data = json.load(f)
    #going through each "layer"
    for feature in data['features']:
        coords = feature['geometry']['coordinates']
        #coordList is a list of coordinates identifying each polygon
        for coordList in coords:
            #each point, expressed as a latitude, longitude pair
            for coordPair in coordList:
                coordPair[0],coordPair[1] = coordPair[1], coordPair[0]
    # here is the new file
    with open(new_filename, 'w') as f:
        json.dump(data, f)
        
def create_neighbourhoods_dictionary(filename):
    with open(filename) as f:
        neighbourhood_dictionary = {}
        js = json.load(f)
        for feature in js['features']:
            coordinates = [(l[0], l[1]) for l in feature['geometry']['coordinates'][0]]
            neighbourhood_dictionary[feature['properties']['NIL']] = coordinates
    return neighbourhood_dictionary
    
def create_neighbourhood_centres_dictionary(neigh_dict):
    neigh_centres_dict = {}
    for key, value in neigh_dict.items():
        neigh_centres_dict[key] = mapping(Polygon(value).representative_point())['coordinates']
    return neigh_centres_dict


# We swap the coordinates in order to get them to be in Milan according to the latitude-longitude convention.
# if on Kaggle swap_geojson_coordinates('../input/nilzone.geojson', 'nilzone_swapped.geojson')
swap_geojson_coordinates('../input/nilzone.geojson', 'nilzone_swapped.geojson')
NIL_coordinates = create_neighbourhoods_dictionary('nilzone_swapped.geojson')  
neighbourhoods_centres = create_neighbourhood_centres_dictionary(NIL_coordinates)


# ## Finally, the map
# 
# And we get a copy to keep as an html file.

# In[ ]:


neighbourhoods_map = folium.Map(location=[45.464211, 9.191383], tiles="cartodbdark_matter", zoom_start=13)
for key, value in neighbourhoods_centres.items():
    popup = str(key)
    folium.Marker([value[0], value[1]], popup=popup).add_to(neighbourhoods_map)

# We have to use the "longitude-latitude" coordinates, that is, the original ones, otherwise off to Somalia they go.
Choropleth(geo_data='../input/nilzone.geojson', fill_color='gray', line_color='green', fill_opacity=0.4,
            line_weight=3).add_to(neighbourhoods_map)

neighbourhoods_map.save(outfile= "milanese_neighbourhoods.html")
neighbourhoods_map


# In[ ]:




