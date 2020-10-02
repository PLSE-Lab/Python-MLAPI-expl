#!/usr/bin/env python
# coding: utf-8

# ![](https://www.airpano.com/files/barcelona_02_big.jpg)

# ># <font color="#116191"><b>Introduction</b></font>
# **Barcelona**  is a city in Spain. It is the capital and largest city of Catalonia. 
# In this kernel, I will be studying the city in a detailed manner. We will take a look at Barcelona's population, immigration, transaport system, air quality etc.

# ># <font color="#116191"><b>Loading required libraries and datasets</b></font>

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

import plotly
import plotly.plotly as py
import cufflinks as cf
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
from plotly import tools
import plotly.tools as tls

import folium
from folium import plugins
from folium.plugins import HeatMap

import warnings
warnings.filterwarnings('ignore')

import os
print(os.listdir("../input"))


# ># <font color="#116191"><b>Barcelona Population</b></font>

# In[ ]:


population_df = pd.read_csv("../input/population.csv")
population_df.head()


# In[ ]:


population_df.info()


# In[ ]:


total_pop = population_df.loc[population_df['Year'] == 2017]['Number'].sum()
print("Total Population of Barcelona =",total_pop)


# In[ ]:


temp_df = population_df.loc[population_df['Year'] == 2017].groupby(['Gender'])['Number'].sum()

trace = go.Pie(labels = temp_df.index,
               values = temp_df.values,
               marker = dict(colors=['#E53916','#2067AD'], line = dict(color='#FFFFFF', width=2.5))
              )

data = [trace]
layout = go.Layout(title="Gender-Wise Distribution for Year-2017")
fig = go.Figure(data=data, layout=layout)

plotly.offline.iplot(fig)


# In[ ]:


male = population_df.loc[population_df['Gender'] == 'Male'].groupby(['Year'])['Number'].sum()
female = population_df.loc[population_df['Gender'] == 'Female'].groupby(['Year'])['Number'].sum()

trace0 = go.Bar(x = male.index,
                y= male.values,
                name = "Male",
                marker = dict(color='rgb(236,154,41)'),
                opacity = 0.8
               )

trace1 = go.Bar(x = female.index,
                y = female.values,
                name = "Female",
                marker = dict(color='rgb(168,32,26)'),
                opacity = 0.8
               )

data = [trace0,trace1]
layout = go.Layout(barmode = 'group',
                   xaxis = dict(tickangle=-30),
                   title="Gender-Wise Distribution Across the Years",
                      )
fig = go.Figure(data=data,layout=layout)

plotly.offline.iplot(fig)


# # <font color="#116191">Insights</font>
# 1. In Barcelona, women outnumber men.
# 2. Over the five years, there has not been significant change in the population of the city.

# # <font color="#116191">Which district holds the highest number of population?</font>

# In[ ]:


dist_df = population_df.loc[population_df['Year'] == 2017].groupby(['District.Name'])['Number'].sum()

trace0 = go.Bar(x = dist_df.index,
                y = dist_df.values,
                marker = dict(color=list(dist_df.values),
                                  colorscale='Reds'),
                )

data = [trace0]
layout = go.Layout(xaxis = dict(tickangle=-30),
                   title="District-Wise Distribution of population (Year 2017)",
                      )
fig = go.Figure(data=data,layout=layout)

plotly.offline.iplot(fig)


# # <font color="#116191">Insights</font>
#  'Eixample' is the most populated district in the city.

# ># <font color="#116191"><b>Immigrants by Nationality</b></font>

# In[ ]:


immig_df = pd.read_csv("../input/immigrants_by_nationality.csv")
immig_df.head()


# In[ ]:


im_df = immig_df.loc[immig_df['Year'] == 2017].groupby(['Nationality'])['Number'].sum().sort_values(axis=0, ascending=False)[:25]

trace0 = go.Bar(x = im_df.index,
                y = im_df.values,
                marker = dict(color=list(dist_df.values),
                                  colorscale='Portland'),
                )

data = [trace0]
layout = go.Layout(xaxis = dict(tickangle=-30),
                   title="Immigration By Nationality (Year 2017)",
                      )
fig = go.Figure(data=data,layout=layout)

plotly.offline.iplot(fig)


# ># <font color="#116191"><b>Transport</b></font>

# In[ ]:


transport_df =pd.read_csv("../input/transports.csv")
transport_df.head()


# # <font color="#116191"><b>Underground Metro</b></font>

# In[ ]:


underground = transport_df.loc[transport_df['Transport'] == 'Underground']
underground = underground[['Latitude','Longitude','Station']]

barcelona_coordinates = [41.3851, 2.1734]

map_undergound = folium.Map(location=barcelona_coordinates, tiles='OpenStreetMap', zoom_start=12)

for each in underground.iterrows():
    folium.CircleMarker([each[1]['Latitude'],each[1]['Longitude']],
                        radius=5,
                        color='blue',
                        popup=each[1]['Station'],
                        fill=True).add_to(map_undergound)
map_undergound


# # <font color="#116191"><b>Bus Stops</b></font>

# In[ ]:


busstop_df =pd.read_csv("../input/bus_stops.csv")
busstop_df.head()


# In[ ]:


daybus = busstop_df.loc[busstop_df['Transport'] == 'Day bus stop']

map_busstop = folium.Map(location=barcelona_coordinates, tiles='OpenStreetMap', zoom_start=12)

for each in daybus[:100].iterrows():
    folium.Marker([each[1]['Latitude'],each[1]['Longitude']],
                  popup=str(each[1]['Bus.Stop']),
                  icon=folium.Icon(color='blue', icon='stop')).add_to(map_busstop)
    
map_busstop


# # <font color="#116191"><b>Accidents (Year 2017)</b></font>

# In[ ]:


accidents_df = pd.read_csv("../input/accidents_2017.csv")
accidents_df.head()


# In[ ]:


accidents_df['killed+injured'] = accidents_df['Mild injuries'] + accidents_df['Serious injuries'] + accidents_df['Victims']
temp_df = accidents_df.groupby(['District Name'])['killed+injured'].sum().sort_values(axis=0, ascending=False)

trace0 = go.Bar(x = temp_df.index,
                y = temp_df.values,
                marker = dict(color=list(temp_df.values))
                )

data = [trace0]
layout = go.Layout(xaxis = dict(tickangle=-30),
                   title="Top Districts Where People Killed and Injured in Accidents",
                      )
fig = go.Figure(data=data,layout=layout)

plotly.offline.iplot(fig)


# In[ ]:


wkday = accidents_df.groupby(['Weekday']).        agg({'Mild injuries':'sum', 'Serious injuries':'sum','Victims':'sum'}).reset_index()
wkday

trace0 = go.Bar(x = wkday['Weekday'],
                y= wkday['Mild injuries'],
                name = "Mild injuries",
                marker = dict(color='rgb(108, 52, 131)')
               )

trace1 = go.Bar(x = wkday['Weekday'],
                y = wkday['Serious injuries'],
                name = "Serious injuries",
                marker = dict(color='rgb(241, 196, 15)')
               )

trace2 = go.Bar(x = wkday['Weekday'],
                y = wkday['Victims'],
                name = "Victims",
                marker = dict(color='rgb(211, 84, 0)')
               )

data = [trace0,trace1,trace2]
layout = go.Layout(barmode = 'group',
                   xaxis = dict(tickangle=-30),
                   title="Weekday-Wise Accidents in Barcelona",
                       
                      )
fig = go.Figure(data=data,layout=layout)

plotly.offline.iplot(fig)


# # <font color="#116191">Insights</font>
# 1. In 2017, most of the accidents occured on Friday.
# 2. The least number of accidents occured on Sunday.

# In[ ]:


month = accidents_df.groupby(['Month']).        agg({'Mild injuries':'sum', 'Serious injuries':'sum','Victims':'sum'}).reset_index()
month

trace0 = go.Bar(x = month['Month'],
                y= month['Mild injuries'],
                name = "Mild injuries",
                marker = dict(color='rgb(205, 92, 92)')
               )

trace1 = go.Bar(x = month['Month'],
                y = month['Serious injuries'],
                name = "Serious injuries",
                marker = dict(color='rgb(75, 0, 130)')
               )

trace2 = go.Bar(x = month['Month'],
                y = month['Victims'],
                name = "Victims",
                marker = dict(color='rgb(34, 139, 34)')
               )

data = [trace0,trace1,trace2]
layout = go.Layout(barmode = 'group',
                   xaxis = dict(tickangle=-30),
                   title="Month-Wise Accidents in Barcelona")
fig = go.Figure(data=data,layout=layout)

plotly.offline.iplot(fig)


# # <font color="#116191">Insights</font>
# 1. In 2017, majority of accidents occured in the month of November.
# 2. The least number of accidents occured in the month of August

# # <font color="#116191">Heatmap of Accidents' Locations</font>

# In[ ]:


barcelona_coordinates = [41.406141, 2.168594]

from folium.plugins import HeatMap

map_accidents = folium.Map(location=barcelona_coordinates, tiles='CartoDB Dark_Matter', zoom_start=13)

lat_long_df = accidents_df[['Latitude','Longitude']]

lat_long = [[row['Latitude'],row['Longitude']] for index,row in lat_long_df.iterrows()]

#map_accidents.add_children(plugins.HeatMap(lat_long))

HeatMap(lat_long, min_opacity=0.5, radius=15).add_to(map_accidents)
map_accidents


# ># <font color="#116191"><b>Air Quality (Year 2017)</b></font>

# In[ ]:


airquality_df = pd.read_csv("../input/air_quality_Nov2017.csv")
airquality_df.dropna(how='any', inplace=True)
airquality_df.head()


# In[ ]:


df = airquality_df.loc[:,['O3 Value','NO2 Value','PM10 Value']]
plt.style.use('fivethirtyeight')
g = sns.pairplot(df)


# In[ ]:


data = [
    {
        'x': airquality_df["Longitude"],
        'y': airquality_df["Latitude"],
        'text': airquality_df["Station"],
        'mode': 'markers',
        'marker': {
            'color': airquality_df["PM10 Value"],
            'size': airquality_df["PM10 Value"],
            'showscale': True,
            'colorscale':'Viridis'
        }
    }
]

layout= go.Layout(title= 'Air Quality',
                  xaxis= dict(title= 'Longitude'),
                  yaxis=dict(title='Latitude'))

fig = go.Figure(data=data, layout=layout)
iplot(fig)


# In[ ]:


trace1 = go.Box(y = airquality_df.loc[airquality_df['Air Quality'] == "Good"]['PM10 Value'],
                name = "Good",
                marker = dict(color='#AF4040')
                )

trace2 = go.Box(y = airquality_df.loc[airquality_df['Air Quality'] == "Moderate"]['PM10 Value'],
                name = "Moderate",
                marker = dict(color='#4986B6'))

data = [trace1,trace2]

layout = go.Layout(xaxis = dict(title='Air Quality'), 
                   yaxis=dict(title='PM10 Values'),
                   title="Air Quality in Barcelona (PM10 Values)")

fig = go.Figure(data=data, layout=layout)
plotly.offline.iplot(fig)


# # <font color="#116191">**To Be Continued**</font>
