#!/usr/bin/env python
# coding: utf-8

# # COVID-19 May be Affected by Congested Waters.

# COronaVIrus Disease 2019 (COVID-19) was first reported in December 2019, and has now engulfed and locked down the entire world.
# 
# The Covid-19 data so far point out that Covid-19 started and has remained prominent in Wuhan in China, Lombardy in Italy, New York on the East Coast of USA, and Seattle on the West Coast of USA. These 4 areas may be treated as oulier areas.
# 
# As an attempt towards a root cause analysis of Covid-19, this notebook plots the world geo map of lakes along with markers for Covid-19 areas, and then presents snapshots of geographic maps and satellite images of the outlier areas.
# 
# The geo map and the snapshots show that the outlier areas are in close vicinity of narrow oceanic bay areas or long narrow lakes. A manual observation of the satellite images shows that waters surrounding these areas appear to be very congested and possible breeding places for various microbes.
# 
# It will be definitely worthwhile to investigate whether waters congested in narrow oceanic bay areas or long narrow lakes around the outlier areas played any role in leading to Covid-19.
# 
# If time permits, I would like to apply Deep Learning techniques to analyze turbidity and pollution of waters around various Covid-19 areas in the world.

# ### Packages and Files

# In[ ]:


from __future__ import print_function
import os
import pandas as pd
import numpy as np
import math
import datetime as dt
import json
# import plotly.express as px
import folium
from urllib.request import urlopen
from IPython.core.display import display, HTML


# In[ ]:


zfLk = "../input/geofiles/ne_50m_lakes.geojson"
# zdir = "../input/outlier-areas/" {This did not work.}
zdir = "http://lipy.us/docs/OutlierAreas/"
zfCc = zdir + "WorldCovidCases.csv"
zfWr = zdir + "WuhanRoads.JPG"
zfWv = zdir + "WuhanWaters1.JPG"
zfWw = zdir + "WuhanWaters2.JPG"
zfIr = zdir + "ItalyRoads.JPG"
zfIv = zdir + "ItalyWaters1.JPG"
zfIw = zdir + "ItalyWaters2.JPG"
zfNr = zdir + "NewYorkRoads.JPG"
zfNv = zdir + "NewYorkWaters1.JPG"
zfNw = zdir + "NewYorkWaters2.JPG"
zfSr = zdir + "SeattleRoads.JPG"
zfSv = zdir + "SeattleWaters1.JPG"
zfSw = zdir + "SeattleWaters2.JPG"


# ### Dataframe of Covid-19 Cases

# In[ ]:


df = pd.read_csv(zfCc, encoding='latin-1')
df = df.fillna("x")
df['Latitude'] = df['Latitude'].apply(lambda x: int(x))
df['Longitude'] = df['Longitude'].apply(lambda x: int(x))
df = df.groupby(by=['Latitude','Longitude']).agg({'Cases':'sum','Deaths':'sum','Country':'max','Code':'max','Capital':'max'})
df = df.sort_values(by=['Cases'], ascending=False)
df = df.reset_index()
df.head(5)


# ### Covid-19 Data and Lakes on Geo Map

# In[ ]:


with open(zfLk) as resp:
    zjso = json.loads(resp.read())
    
zlls = []
for zloc in zjso['features']:
    zlls.append([zloc['properties']['name'],zloc['geometry']['coordinates'][0][0][0],zloc['geometry']['coordinates'][0][1][0]])
zlls = pd.DataFrame(data=zlls, columns=['name','lat','long'])
zlls.head(3)


# In[ ]:


zmap = folium.Map([20,0], zoom_start=2, tiles='cartodbpositron')

colorC = {0:'orange', 1:'beige', 2:'purple', 3:'darkpurple', 4:'lightred', 5:'red', 6:'darkred'}
# green,darkgreen,lightgreen,blue,darkblue,cadetblue,lightblue,white,pink,gray,black,lightgray

for Lat,Lon,Cas,Dea,Cou,Cap in zip(df['Latitude'],df['Longitude'],df['Cases'],df['Deaths'],df['Country'],df['Capital']):
    
    Val = int(math.log(Cas+9, 8))
    
    folium.CircleMarker(
        location = [Lat,Lon],
        radius = Val,
        popup = str(Cou) + '<br>' + str(Cap) + '<br>' +str(Cas) + '<br>' + str(Dea),
        threshold_scale = [0,1,2,3,4,5,6],
        color = colorC[Val],
        fill_color = colorC[Val],
        fill = False,
        fill_opacity = 1
    ).add_to(zmap)

folium.GeoJson(zjso).add_to(zmap)
# folium.LatLngPopup().add_to(zmap)

zmap


# ### Satellite Images of Outlier Locations

# In[ ]:


zimg = [[zfWr,zfWv,zfWw],[zfIr,zfIv,zfIw],[zfNr,zfNv,zfNw],[zfSr,zfSv,zfSw]]

zhtm = "<table><tr><td>.</td><td>Wuhan China</td><td>Lombardy Italy</td><td>New York USA</td><td>Seattle</td></tr>"
for i in range(3):
    zhtm = zhtm + "<tr><td>" + ["Roads","Sat Image","sat Image"][i] + "</td>"
    for j in range (4):
        zhtm = zhtm + "<td><img style='width:120px;height:180px;margin:0px;float:left;border:5px solid green;' src='" + zimg[j][i] + "' /></td>"
    zhtm = zhtm + "</tr>"
zhtm = zhtm + "</table>"
    
display(HTML(zhtm))


# ### Observations

# The Covid-19 data so far point out that Covid-19 started and has remained prominent in Wuhan in China, Lombardy in Italy, New York on the East Coast of USA, and Seattle on the West Coast of USA. These 4 areas may be treated as oulier areas.
# 
# The world geo map of lakes along with markers for Covid-19 areas, snapshots of geographic maps of the outlier areas, and snapshots of satellite images of the outlier areas show that the outlier areas are in close vicinity of narrow oceanic bay areas or long narrow lakes.
# 
# A manual observation of the satellite images shows that waters surrounding these areas appear to be very congested and possible breeding places for various microbes.
# 
# It will be definitely worthwhile to investigate whether waters congested in narrow oceanic bay areas or long narrow lakes around the outlier areas played any role in leading to Covid-19.
# 
# If time permits, I would like to apply Deep Learning techniques to analyze turbidity and pollution of waters around various Covid-19 areas in the world.

# ### References

# GovCDC: https://www.cdc.gov/coronavirus/2019-nCoV/index.html
# 
# GovWHOCovid: https://www.who.int/health-topics/coronavirus
# 
# OrgWikiCovid: https://en.wikipedia.org/wiki/Coronavirus_disease_2019
# 
# OrgHopkinsM: https://www.hopkinsmedicine.org/coronavirus/
# 
# ComNaturalE: https://www.naturalearthdata.com/
# 
# ComHighCharts: http://code.highcharts.com/mapdata/
