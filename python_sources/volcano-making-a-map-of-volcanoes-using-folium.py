#!/usr/bin/env python
# coding: utf-8

# # Volcano Eruptions
# This notebook uses Folium to create a map of volcanoes across the world. Scroll down to the end of the notebook to see the final map created with Folium.

# In[ ]:


import numpy as np
import pandas as pd
import folium
import matplotlib.pyplot as plt


# # Import Data

# In[ ]:


DATA_DIR = '../input/volcano-eruptions/'


# In[ ]:


data_eruptions = pd.read_csv(DATA_DIR + 'eruptions.csv')
data_events = pd.read_csv(DATA_DIR + 'events.csv')
data_sulfur = pd.read_csv(DATA_DIR + 'sulfur.csv')
data_tree_rings = pd.read_csv(DATA_DIR + 'tree_rings.csv')
data_volcano = pd.read_csv(DATA_DIR + 'volcano.csv')


# # Volcano Data
# Firstly we will have a quick look at the volcano data that we have, which can be found in the data_volcano dataframe. There is 958 different volcanoes in this dataframe. We have 26 different features. The most important of these for plotting a map will be the latitude and longitude. Other useful features include the volcano name which would be useful to users of the map for identifying the volcanoes they are looking at.

# In[ ]:


data_volcano.head(5)


# In[ ]:


print('Rows:   ', data_volcano.shape[0])
print('Columns:', data_volcano.columns.values)


# # Creating A Map
# We will create a simple map of volcanoes below. To do this we only need the latitude and longitude for each volcano, so that we can create a marker. We will also add the volcano name in a pop-up so that users can see the name of volcanoes by clicking on them. Each marker must be added to the map by using the add_to function.

# In[ ]:


volcano_map = folium.Map()

# Add each volcano to the map
for i in range(0, data_volcano.shape[0]):
    volcano = data_volcano.iloc[i]
    folium.Marker([volcano['latitude'], volcano['longitude']], popup=volcano['volcano_name']).add_to(volcano_map)

volcano_map


# # The Traditional Way
# Let's just compare that to a traditional way of viewing geographical data without a map. How would we do it? Fortunately there is a country feature in the dataset, so we can look at the number of volcanoes per country. There is also a region feature, so we can do the same with that.

# In[ ]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))

volcano_country = pd.DataFrame(data_volcano.groupby(['country']).size()).sort_values(0, ascending=True)
volcano_country.columns = ['Count']
volcano_country.tail(10).plot(kind='barh', legend=False, ax=ax1)
ax1.set_title('Number of Volcanoes per Country')
ax1.set_ylabel('')
ax1.set_xlabel('count')

volcano_region = pd.DataFrame(data_volcano.groupby(['region']).size()).sort_values(0, ascending=True)
volcano_region.columns = ['Count']
volcano_region.tail(10).plot(kind='barh', legend=False, ax=ax2)
ax2.set_title('Number of Volcanoes per Region')
ax2.set_ylabel('')
ax2.set_xlabel('count')

plt.tight_layout()
plt.show()


# # Eruption Data
# 
# Hopefully you have got to this point and you are thinking that the Folium map is superior. Volcanoes don't respect national borders, after all. But the map doesn't give us very much information on each volcano. Another piece of information we might want is data on eruptions for each volcano. So let's have a quick look at the eruption data.

# In[ ]:


data_eruptions.columns


# In[ ]:


data_eruptions[data_eruptions['vei'] > 6]


# # Volcano VEI Map
# 
# After looking at the eruption data we can notice many more things to plot on a map. One of these things is the VEI (Volcanic Explosivity Index), on which [more information can be found here](https://en.wikipedia.org/wiki/Volcanic_Explosivity_Index). We will use this data to create a map which shows the maximum VEI from the eruptions for each volcano.
# 
# To do this, we must first add the max VEI data to the data_volcano dataframe.

# In[ ]:


# Get max VEI for each volcano
volcano_max_vei = data_eruptions.groupby(['volcano_number'])['vei'].max().reset_index()

# Merge these values into the volcano dataframe
data_volcano = pd.merge(data_volcano, volcano_max_vei, on='volcano_number')


# Now we can plot the volcanoes with their maximum VEI data. This is done in following cell. The map can be found below, and some more explanation of parameters and functions used will be below that.

# In[ ]:


def vei_radius(vei):
    return 2 ** (int(vei) - 4) + 3 if not np.isnan(vei) else 1
    
volcano_with_vei = data_volcano#.dropna(subset=['vei'])

# Create the map
volcano_vei_map = folium.Map()

# Create layers
layers = []
for i in range(8):
    layers.append(folium.FeatureGroup(name='VEI: '+str(i)))
layers.append(folium.FeatureGroup(name='VEI: NaN'))

# Add each volcano to the correct layer
for i in range(0, volcano_with_vei.shape[0]):
    volcano = volcano_with_vei.iloc[i]
    # Create marker
    marker = folium.CircleMarker([volcano['latitude'],
                                  volcano['longitude']],
                                  popup=volcano['volcano_name'] + ', VEI: ' + str(volcano['vei']),
                                  radius=vei_radius(volcano['vei']),
                                  color='red' if not np.isnan(volcano['vei']) and int(volcano['vei']) == 7 else 'blue',
                                  fill=True)
    # Add to correct layer
    if np.isnan(volcano['vei']):
        marker.add_to(layers[8])
    else:
        marker.add_to(layers[int(volcano['vei'])])

# Add layers to map
for layer in layers:
    layer.add_to(volcano_vei_map)
folium.LayerControl().add_to(volcano_vei_map)

volcano_vei_map


# Now for a little more explanation...
# * We have created different layers using FeatureGroup(), markers are added to feature groups instead of directly to the map
# * Layers can be toggled interactively by adding a LayerControl() to the map
# * We have used the max VEI to determine the radius (a higher max VEI makes a higher radius), and to do this we created a separate function
# * The color parameters markers has been used to separately display VEI 7 volcanoes when there are many markers on the map
# * The VEI was also added to the pop-up (as an easy way for users to see the VEI of a specific explanation
# * Not every volcano had an eruption with a registered VEI, so we needed a few clauses to handle NaN values

# # Adding Tectonic Plates
# Now we will introduce another dataset, the [Tectonic Plate Boundaries set](https://www.kaggle.com/cwthompson/tectonic-plate-boundaries/settings). With a little bit of domain knowledge we could know that volcanoes are often found near tectonic plate boundaries. So we are now going to add the tectonic plate boundaries to our map. Firstly, we will import the data.

# In[ ]:


tectonic_plates = pd.read_csv('../input/tectonic-plate-boundaries/all.csv')
tectonic_plates.head()


# Now we are going to plot the tectonic plate boundaries. Since the data gives the boundaries for each plate, we are going to add the plates separately. To do this, we use PolyLines.
# 
# If you just collect the points and try to add PolyLines you may notice a problem; some of the boundaries wrap around the world (since the world is round) and that causes a huge line across our map. To deal with this, we try to spot where this wrap-around happens and we split that plate up. This allows us to plot the two or more parts of the plates which look as though they are on the opposite sides of the world on our map (but are in fact next to each other).

# In[ ]:


plate_map = folium.Map()

plates = list(tectonic_plates['plate'].unique())
for plate in plates:
    plate_vals = tectonic_plates[tectonic_plates['plate'] == plate]
    lats = plate_vals['lat'].values
    lons = plate_vals['lon'].values
    points = list(zip(lats, lons))
    indexes = [None] + [i + 1 for i, x in enumerate(points) if i < len(points) - 1 and abs(x[1] - points[i + 1][1]) > 300] + [None]
    for i in range(len(indexes) - 1):
        folium.vector_layers.PolyLine(points[indexes[i]:indexes[i+1]], popup=plate, color='green', fill=False).add_to(plate_map)

plate_map


# That was great. Now let's add it all together!

# In[ ]:


def vei_radius(vei):
    return 2 ** (int(vei) - 4) + 3 if not np.isnan(vei) else 1
    
volcano_with_vei = data_volcano#.dropna(subset=['vei'])

# Create the map
complete_map = folium.Map()

# Add tectonic plates to map
plate_layer = folium.FeatureGroup(name='Tectonic Plates')
plates = list(tectonic_plates['plate'].unique())
for plate in plates:
    plate_vals = tectonic_plates[tectonic_plates['plate'] == plate]
    lats = plate_vals['lat'].values
    lons = plate_vals['lon'].values
    points = list(zip(lats, lons))
    indexes = [None] + [i + 1 for i, x in enumerate(points) if i < len(points) - 1 and abs(x[1] - points[i + 1][1]) > 300] + [None]
    for i in range(len(indexes) - 1):
        folium.vector_layers.PolyLine(points[indexes[i]:indexes[i+1]], popup=plate, color='green', fill=False).add_to(plate_layer)
plate_layer.add_to(complete_map)

# Create layers
layers = []
for i in range(8):
    layers.append(folium.FeatureGroup(name='VEI: '+str(i)))
layers.append(folium.FeatureGroup(name='VEI: NaN'))

# Add each volcano to the correct layer
for i in range(0, volcano_with_vei.shape[0]):
    volcano = volcano_with_vei.iloc[i]
    # Create marker
    marker = folium.CircleMarker([volcano['latitude'],
                                  volcano['longitude']],
                                  popup=volcano['volcano_name'] + ', VEI: ' + str(volcano['vei']),
                                  radius=vei_radius(volcano['vei']),
                                  color='red' if not np.isnan(volcano['vei']) and int(volcano['vei']) == 7 else 'blue',
                                  fill=True)
    # Add to correct layer
    if np.isnan(volcano['vei']):
        marker.add_to(layers[8])
    else:
        marker.add_to(layers[int(volcano['vei'])])

# Add layers to map
for layer in layers:
    layer.add_to(complete_map)

# Add layer control
folium.LayerControl().add_to(complete_map)

complete_map


# And there we have it, our completed map! Hopefully you now have a little more understanding of Folium, volcanoes, or tectonic plates. If you found it useful, please upvote the notebook.
