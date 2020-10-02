#!/usr/bin/env python
# coding: utf-8

# ## After my [ first](https://www.kaggle.com/somesnm/new-york-parties-eda) exploration of a [dataset](https://www.kaggle.com/somesnm/partynyc) of party related noise complaints in New York City, I focused on establishing zones of high concentration of bars, pubs, and clubs. 
# 
# ## I produced and added to the dataset the table that contains locations of places in a category "Club/Bar/Restaurant" that have at least 10 noise complaints in 2016.  This is a short overview of gathered data.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import folium
from folium import features
from folium.plugins import HeatMap
from folium.plugins import MarkerCluster

import seaborn


# ## Locations of bars, restaurants, clubs, and pubs with at least 10 noise complaints in 2016.

# In[ ]:


bars = pd.read_csv('../input/partynyc/bar_locations.csv')
bars.head()


# ## Manhattan has the highest number of locations, followed by Brooklyn.

# In[ ]:


bars.Borough.value_counts(ascending=True).tail().plot.barh();


# ## The heatmap shows the zones of high concentration of popular drinking  establishments.

# In[ ]:


pubs_map = folium.Map(location=[40.742459, -73.971765], zoom_start=12)
data = [[x[0], x[1], 1] for x in np.array(bars[['Latitude', 'Longitude']])]
HeatMap(data, radius = 20).add_to(pubs_map)
pubs_map


# ## This map shows that the Lower Manhattan and Williamsburg has the highest concentration of pubs and bars among Ney York negihborhoods.

# In[ ]:


map_wb = folium.Map(location=[40.742459, -73.971765],zoom_start=12)#, tiles='Active nightlife zone')
mc = MarkerCluster()
for ind,row in bars.iterrows():
    mc.add_child(folium.CircleMarker(location=[row['Latitude'],row['Longitude']],
                        radius=1,color='#3185cc'))
map_wb.add_child(mc)
map_wb


# ## The insights from this data can be used to establish zones of active nightlife and include that information in modeling of the Taxi rides.
