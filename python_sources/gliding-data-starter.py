#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# This kernel gives a starter over the contents of the Gliding Data dataset.
# 
# The dataset includes metadata and calculated phases for over 100000 gliding flights from 2016 to 2019, mostly in the region of France but also Belgium, Switzerland and others. In total there are more than 6 million flights phases recorded.
# 
# ## Gliding
# 
# Gliding is a leisure aviation activity and sport where pilots fly unpowered aircraft known as gliders or sailplanes.
# 
# The principle of gliding is to climb using some sort of lift and convert the altitude gained into distance. Repeating this process allows for very long distance flights, often above 500km or even 1000km - the current [free distance world record](https://www.fai.org/record/7605) being just over 3000km in Patagonia. This is the same process birds follow to reduce the amount of effort required to travel great distances.
# 
# The most used sources of lift include:
# * thermals: where the air rises due to heating from the sun, often marked by cumulus clouds at the top
# * wave: created by strong winds and stable air layers facing a mountain or a high hill, often marked by lenticular clouds
# 
# with thermalling being by far the most common, and this dataset reflects that.
# 
# ![](http://www.flybc.org/thermals2010.jpg)
# 
# ## Data Recorders
# 
# When flying pilots often carry some sort of GPS recording device which generates a track of points collected every few seconds. Each point contains fields for latitude, longitude and altitude (pressure and GPS) among several others, often stored in [IGC](http://www.ukiws.demon.co.uk/GFAC/documents/tech_spec_gnss.pdf) format. Often these tracks are uploaded to online competitions where they are shared with other pilots, and can be easily visualized.
# 
# ![](https://raw.githubusercontent.com/ezgliding/goigc/master/docs/images/track.png)
# 
# The data available in this dataset was scraped from the online competition [Netcoupe](https://netcoupe.net). The scraping, parsing and analysis was done using [goigc](https://github.com/ezgliding/goigc), an open source flight parser and analyser.

# # Getting Started
# 
# Let's start by loading the different files in the dataset and taking a peek at the records.
# 
# The available files include:
# * flight_websource: the metadata exposed in the online competition website, including additional information to what is present in the flight track
# * flight_track: the metadata collected directly from the GPS/IGC flight track
# * phases: the flight phases (cruising, circling) calculated from the GPS/IGC flight track (this is a large file, we load a subset below)
# * handicaps: a mostly static file with the different handicaps attributed to each glider type by the IGC, important when calculating flight performances

# In[ ]:


import datetime
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


flight_websource = pd.read_csv("../input/gliding-data/flight_websource.csv")
flight_track = pd.read_csv("../input/gliding-data/flight_track.csv")
flight_phases = pd.read_csv("../input/gliding-data/phases.csv", skiprows=lambda i: i>0 and random.random() > 0.5)


# # Flight Metadata
# 
# There's a lot of information contained in the two flight metadata files (websource and track).

# In[ ]:


flight_websource.head(1)


# In[ ]:


flight_track.head(1)


# The most useful information comes from the *websource* file as this information is passed directly by the pilot when submitting the flight to the online competition. Things like *Country* or *Region* provide useful statistics on how popular the sport is in different areas. As an example, what are the most popular regions considering the total number of flights? What about the most popular *Takeoff* location?

# In[ ]:


flight_websource.groupby(['Country', 'Region'])['Region'].value_counts().sort_values(ascending=False).head(3)


# In[ ]:


flight_websource.groupby(['Country', 'Year', 'Takeoff'])['Takeoff'].value_counts().sort_values(ascending=False).head(3)


# The three regions above match the Alps area, which is an expected result given this is a gliding Meca. The second result shows *Vinon* as the most popular takeoff in 2016, a big club also in the Southern Alps. But it's interesting to notice that more recently a club near Montpellier took over as the one with the most activity in terms of number of flights.
# 
# Gliding is a seasonal activity peaking in summer months. 

# In[ ]:


flight_websource['DayOfWeek'] = flight_websource.apply(lambda r: datetime.datetime.strptime(r['Date'], "%Y-%m-%dT%H:%M:%SZ").strftime("%A"), axis=1)
flight_websource.groupby(['DayOfWeek'])['DayOfWeek'].count().plot.bar()


# In[ ]:


flight_websource['Month'] = flight_websource.apply(lambda r: datetime.datetime.strptime(r['Date'], "%Y-%m-%dT%H:%M:%SZ").strftime("%m"), axis=1)
flight_websource.groupby(['Month'])['Month'].count().plot.bar()


# ## Merging Data
# 
# We can get additional flight metadata information from the *flight_track* file, but you can expect this data to be less reliable as it's often the case that the metadata in the flight recorder is not updated before a flight. It is very useful though to know more about what type of recorder was used, calibration settings, etc. It is also the source of the flight tracks used to generate the data in the *phases* file.
# 
# In some cases we want to handle columns from both flight metadata files, so it's useful to join the two sets. We can rely on the ID for this purpose.

# In[ ]:


flight_all = pd.merge(flight_websource, flight_track, how='left', on='ID')
flight_all.head(1)


# # Flight Phases
# 
# In addition to the flight metadata provided in the files above, by analysing the GPS flight tracks we can generate a lot more interesting data.
# 
# Here we take a look at flight phases, calculated using the [goigc](https://github.com/ezglding/goigc) tool. As described earlier to travel further glider pilots use thermals to gain altitude and then convert that altitude into distance. In the *phases* file we have a record of each individual phase detected for each of the 100k flights, and we'll focus on:
# * Circling (5): phase where a glider is gaining altitude by circling in an area of rising air
# * Cruising (3): phase where a glider is flying straight converting altitude into distance
# 
# These are indicated by the integer field *Type* below. Each phase has a set of additional fields with relevant statistics for each phase type: while circling the average climb rate (vario) and duration are interesting; while cruising the distance covered and LD (glide ratio) are more interesting.

# In[ ]:


flight_phases.head(1)


# # Data Preparation
# 
# As a quick example of what is possible with this kind of data let's take try to map all *circling* phases as a HeatMap.
# 
# First we need to do some treatment of the data: convert coordinates from radians to degrees, filter out unreasonable values (climb rates above 15m/s are due to errors in the recording device), convert the date to the expected format and desired grouping. In this case we're grouping all thermal phases by week.

# In[ ]:


phases = pd.merge(flight_phases, flight_websource[['TrackID', 'Distance', 'Speed']], on='TrackID')
phases['Lat'] = np.rad2deg(phases['CentroidLatitude'])
phases['Lng'] = np.rad2deg(phases['CentroidLongitude'])

phases_copy = phases[phases.Type==5][phases.AvgVario<10][phases.AvgVario>2].copy()
phases_copy.head(2)

#phases_copy['AM'] = phases_copy.apply(lambda r: datetime.datetime.strptime(r['StartTime'], "%Y-%m-%dT%H:%M:%SZ").strftime("%p"), axis=1)
#phases_copy['Day'] = phases_copy.apply(lambda r: datetime.datetime.strptime(r['StartTime'], "%Y-%m-%dT%H:%M:%SZ").strftime("%j"), axis=1)
#phases_copy['Week'] = phases_copy.apply(lambda r: datetime.datetime.strptime(r['StartTime'], "%Y-%m-%dT%H:%M:%SZ").strftime("%W"), axis=1)
#phases_copy['Month'] = phases_copy.apply(lambda r: r['StartTime'][5:7], axis=1)
#phases_copy['Year'] = phases_copy.apply(lambda r: r['StartTime'][0:4], axis=1)
#phases_copy['YearMonth'] = phases_copy.apply(lambda r: r['StartTime'][0:7], axis=1)
#phases_copy['YearMonthDay'] = phases_copy.apply(lambda r: r['StartTime'][0:10], axis=1)

# use the corresponding function above to update the grouping to something other than week
phases_copy['Group'] = phases_copy.apply(lambda r: datetime.datetime.strptime(r['StartTime'], "%Y-%m-%dT%H:%M:%SZ").strftime("%W"), axis=1)
phases_copy.head(1)


# ## Visualization
# 
# Once we have the data ready we can visualize it over a map. We rely on folium for this.

# In[ ]:


# This is a workaround for this known issue:
# https://github.com/python-visualization/folium/issues/812#issuecomment-582213307
get_ipython().system('pip install git+https://github.com/python-visualization/branca')
get_ipython().system('pip install git+https://github.com/sknzl/folium@update-css-url-to-https')


# In[ ]:


import folium
from folium import plugins
from folium import Choropleth, Circle, Marker
from folium.plugins import HeatMap, HeatMapWithTime, MarkerCluster

# folium.__version__ # should be '0.10.1+8.g4ea1307'
# folium.branca.__version__ # should be '0.4.0+4.g6ac241a'


# ### Single HeatMap

# In[ ]:


# we use a smaller sample to improve the visualization
# a better alternative is to group entries by CellID, an example of this will be added later
phases_single = phases_copy.sample(frac=0.01, random_state=1)
m_5 = folium.Map(location=[47.06318, 5.41938], tiles='stamen terrain', zoom_start=7)
HeatMap(
    phases_single[['Lat','Lng','AvgVario']], gradient={0.5: 'blue',  0.7:  'yellow', 1: 'red'},
    min_opacity=5, max_val=phases_single.AvgVario.max(), radius=4, max_zoom=7, blur=4, use_local_extrema=False).add_to(m_5)

m_5


# ### HeatMap over Time
# 
# Another cool possibility is to visualize the same data over time.
# 
# In this case we're grouping weekly and playing the data over one year.
# 
# Both the most popular areas and times of the year are pretty clear from this animation.

# In[ ]:


m_5 = folium.Map(location=[47.06318, 5.41938], tiles='stamen terrain', zoom_start=7)

groups = phases_copy.Group.sort_values().unique()
data = []
for g in groups:
    data.append(phases_copy.loc[phases_copy.Group==g,['Group','Lat','Lng','AvgVario']].groupby(['Lat','Lng']).sum().reset_index().values.tolist())
    
HeatMapWithTime(
    data,
    index = list(phases_copy.Group.sort_values().unique()),
    gradient={0.1: 'blue',  0.3:  'yellow', 0.8: 'red'},
    auto_play=True, scale_radius=False, display_index=True, radius=4, min_speed=1, max_speed=6, speed_step=1,
    min_opacity=1, max_opacity=phases_copy.AvgVario.max(), use_local_extrema=True).add_to(m_5)

m_5


# In[ ]:




