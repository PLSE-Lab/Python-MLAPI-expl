#!/usr/bin/env python
# coding: utf-8

# ##Plotting Location of Attacks
# This is my first kernel, just wanted to explore some cyber data. Open to any suggestions for improving my analysis or ways to take this analysis further.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import IFrame
import plotly
import plotly.offline as py
import plotly.graph_objs as go
from mpl_toolkits.basemap import Basemap
get_ipython().system('conda install basemap-data-hires --yes')

plt.style.use('ggplot')


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('../input/AWS_Honeypot_marx-geo.csv', parse_dates=['datetime'])
df_orig = df
df.head()


# Read data and examine first few lines

# In[ ]:


df.isnull().sum()


# Examine the number of nulls in data. we see that type and unnamed are mostly null. They will not be important to our analysis anyway so we will drop them

# In[ ]:


df = df.drop(['type', 'Unnamed: 15'], axis=1)


# In[ ]:


df = df.dropna(subset=['latitude'])
df = df.dropna(subset=['longitude'])
df.isnull().sum()


# Drop any NA values for lat and long columns since we will be using that column to do our geographic analysis

# In[ ]:


df = df.set_index(pd.DatetimeIndex(df['datetime']))


# Set index of dataframe to the date column to do time series analysis

# In[ ]:


df['num_attacks'] = 1
grouped_times = df.resample('1H').sum()
grouped_times.fillna(0, inplace=True)
grouped_times['num_attacks'].head(10)


# Here we split the data into 1 hour chunks of time for our time series analysis. The way to count attacks is a bit messy as we just set a new column to 1 since each row represents one attack. This makes it easy to then count all '1's in an hour chunk of time to get the total number of attacks for that chunk.

# In[ ]:


# create the plot space upon which to plot the data
fig, ax= plt.subplots()

# add the x-axis and the y-axis to the plot
ax.bar(grouped_times.index.values, 
        grouped_times['num_attacks'], 
        color = 'red',
        alpha=0.3)

# rotate tick labels
plt.setp(ax.get_xticklabels(), rotation=45)

# set title and labels for axes
ax.set(xlabel="Date",
       ylabel="Attacks",
       title="Number of Attacks Per Hour by Date");


# Plot number of attacks per hour across whole dataset. We see some very major spikes on a few days throught the span of data colection.

# In[ ]:


grouped_times['log_value'] = np.log(grouped_times['num_attacks']+1)

break_ins = go.Scatter(
                x=grouped_times.index,
                y=grouped_times['log_value'],
                name='Flagged Break-Ins'
                )

layout = go.Layout(
    title = 'Time Series of Break-In Attempts',
    xaxis = go.layout.XAxis(
        title=go.layout.xaxis.Title(text='Date'),
        tickformat = '%d %B %Y'
    ),
    yaxis=go.layout.YAxis(
        title=go.layout.yaxis.Title(text='Number of Break-Ins')
    )
)

data = [break_ins]

fig = go.Figure(
    data=data,
    layout=layout
)

py.iplot(fig, filename = 'break-ins-over-time')


# Another time series analysis, but this time we take the log to smooth out some of the high spikes.

# In[ ]:


top_countries = df['country'].value_counts()
c_name = list(top_countries.index)
c_count = list(top_countries.values)
top_countries.head(10)


# List top countries where attacks originate

# In[ ]:


top_countries = pd.DataFrame({'name':c_name, 'num':c_count})
top_countries['tup'] = list(zip(top_countries.name, top_countries.num))
top_countries['log_value'] = np.log(top_countries['num'])


# Turn top countries series into a usable dataframe for plotting later

# In[ ]:


#color ramp for chloropleth map
scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'], [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]

fig = go.Figure(
data = [ dict(
        type='choropleth',
        colorscale = 'blues',
        locations = top_countries['name'],
        z = top_countries['log_value'],
        locationmode = ('country names'),
        text = ('country: '+top_countries['name'] + '<br>' +\
               'number of attacks: '+top_countries['num'].apply(str)),
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 0.5
            )
        ),
        colorbar = dict(
            title = "Log of<br>Frequency"
        )
    ) ],

layout = dict(
        title = 'Honeypot Attacks by Country',
        geo = dict(
            scope='world',
            projection=dict( type='natural earth' ),
            showlakes = True,
            landcolor = 'lightgray',
            showland = True,
            showcountries = True,
            countrycolor = 'gray',
            countrywidth = 0.5,
            lakecolor = 'rgb(255, 255, 255)',
        ),
    )
)
py.iplot(fig, filename='IP-world-map')


# Generate a heatmap of attack origins

# In[ ]:


df.drop(df[df['latitude'] > 90].index, inplace=True)
df['latitude'].describe()


# There is a value in the latitude column that is much higher than 90. Since latitudes above 90 do not exist, we drop this row so we can plot using the lat long coordinates

# In[ ]:


#Create basemap
lat = df['latitude'].values
lon = df['longitude'].values

# buffer to add to the range
margin = 20 
lat_min = min(lat) - margin
lat_max = max(lat) + margin
lon_min = min(lon) - margin
lon_max = max(lon) + margin

plt.figure(figsize=(30,15))
m = Basemap(llcrnrlon=lon_min,
            llcrnrlat=lat_min,
            urcrnrlon=lon_max,
            urcrnrlat=lat_max,
            lat_0=(lat_max - lat_min)/2,
            lon_0=(lon_max-lon_min)/2,
            projection='merc',
            resolution = 'h',
            area_thresh=10000.,
           )
m.drawcoastlines()
m.drawcountries()
m.drawstates()
m.drawmapboundary(fill_color='#46bcec')
m.fillcontinents(color = 'white',lake_color='#46bcec')
# convert lat and lon to map projection coordinates
lons, lats = m(lon, lat)
# plot points as red dots
m.scatter(lons, lats, marker = '.', color='r', zorder=5, s=3)
plt.show()


# Basemap with dots indicating where each attack originated

# In[ ]:


df['srcstr'].value_counts().nlargest(10)


# Here we see the top 10 IP addresses in terms of attack volume

# In[ ]:


df_ip1 = df[df.srcstr == '175.146.199.252']
df_ip2 = df[df.srcstr == '2.186.189.218']
df_ip3 = df[df.srcstr == '203.178.148.19']
df_ip4 = df[df.srcstr == '128.9.168.98']
df_ip5 = df[df.srcstr == '129.82.138.44']
df_ip6 = df[df.srcstr == '183.91.14.60']
df_ip7 = df[df.srcstr == '96.254.171.2']
df_ip8 = df[df.srcstr == '68.145.164.27']
df_ip9 = df[df.srcstr == '123.151.42.61']
df_ip10 = df[df.srcstr == '220.225.17.46']


# Create different dataframes for each of the top 10 bad IP addresses

# In[ ]:


def generate_timeline(df,ip):
    grouped_times = df.resample('1H').sum()
    grouped_times.fillna(0, inplace=True)
    #grouped_times['log_value'] = np.log(grouped_times['num_attacks']+1)

    break_ins = go.Scatter(
                    x=grouped_times.index,
                    y=grouped_times['num_attacks'],
                    name='Attacks'
                    )

    layout = go.Layout(
        title = 'Time Series of Break-In Attempts ('+ip+')',
        xaxis = go.layout.XAxis(
            title=go.layout.xaxis.Title(text='Date'),
            tickformat = '%d %B %Y'
        ),
        yaxis=go.layout.YAxis(
            title=go.layout.yaxis.Title(text='Number of Break-Ins')
        )
    )

    data = [break_ins]

    fig = go.Figure(
        data=data,
        layout=layout
    )

    py.iplot(fig, filename = 'break-ins-over-time')


# Helper function to consolidate some code since we are just using our Plotly graph from earlier to plot these specific IP addresses

# In[ ]:


generate_timeline(df_ip1,'175.146.199.252')
generate_timeline(df_ip2,'2.186.189.218')
generate_timeline(df_ip3,'203.178.148.19')
generate_timeline(df_ip4,'128.9.168.98')
generate_timeline(df_ip5,'129.82.138.44')
generate_timeline(df_ip6,'183.91.14.60')
generate_timeline(df_ip7,'96.254.171.2')
generate_timeline(df_ip8,'68.145.164.27')
generate_timeline(df_ip9,'123.151.42.61')
generate_timeline(df_ip10,'220.225.17.46')


# We see some interesting trends here. Some of the top offenders only have one period of time they run their attacks. This may be an indicator of a DoS type attack. Some are very sporadic and some try to attack very consistently

# In[ ]:


df_orig['proto'].value_counts()


# In[ ]:


df_orig.head()


# In[ ]:


df_mapbox = df_orig.drop(['src','spt','dpt','cc','locale','localeabbr','postalcode'], axis=1)
df_mapbox = df_orig.dropna(subset=['latitude'])
df_mapbox = df_orig.dropna(subset=['longitude'])
df1 = df_orig[df_orig['proto']=='TCP']
df2 = df_orig[df_orig['proto']=='UDP']
df3 = df_orig[df_orig['proto']=='ICMP']


# In[ ]:


mapbox_access_token = 'pk.eyJ1IjoianNjZWFyY2UiLCJhIjoiY2p5azhidjh3MGJ1azNxbGlyeXJrNDA3ZCJ9.6UXtubZtsMny5_wlho0IaA'


# In[ ]:


data = [
    go.Scattermapbox(
        name='TCP',
        lat=df1['latitude'],
        lon=df1['longitude'],
        mode='markers',
        text = df1['srcstr'],
        marker=dict(
            size=6,
            color='orange',
            opacity=0.5
        )),
    go.Scattermapbox(
        name='UDP',
        lat=df2['latitude'],
        lon=df2['longitude'],
        mode='markers',
        text = df2['srcstr'],
        marker=dict(
            size=6,
            color='blue',
            opacity=0.5
        )),
    go.Scattermapbox(
        name='ICMP',
        lat=df3['latitude'],
        lon=df3['longitude'],
        mode='markers',
        text = df3['srcstr'],
        marker=dict(
            size=6,
            color='beige',
            opacity=0.5
        ))
        ]
layout = go.Layout(
    autosize=True,
    hovermode='closest',
    mapbox=dict(
        accesstoken=mapbox_access_token,
        bearing=0,
        center=go.layout.mapbox.Center(
            lat=0,
            lon=0
        ),
        pitch=20,
        zoom=1,
        style='satellite'
    ),
)

fig = dict(data=data, layout=layout)
py.iplot(fig, filename='Attack Mapbox')


# Quick view of lat long coordinates for attacks colored by the protocol of attack
