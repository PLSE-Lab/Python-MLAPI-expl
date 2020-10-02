#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install geopandas


# In[ ]:


pip install geoplot


# In[ ]:


pip install plotly


# In[ ]:


pip install geoplotlib


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import geoplot as gplt
import geoplot.crs as gcrs
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from shapely.geometry import Point, Polygon
import calendar
import matplotlib.font_manager as fm

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you fwrite to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')


# In[ ]:


data.shape


# In[ ]:


data.describe().T


# In[ ]:


data.head(5)


# In[ ]:


pd.to_datetime(data['last_review']).max()


# In[ ]:


len(data['name'].unique().tolist())


# In[ ]:


len(data['host_id'].unique().tolist())


# In[ ]:


data = data.dropna()


# In[ ]:


data.corr().style.background_gradient(cmap='coolwarm')


# In[ ]:


NeighbourHoodGrpCnt = data['neighbourhood_group'].value_counts()
NeighbourHoodGrpCnt


# In[ ]:


colors = ['mediumturquoise', 'gold', 'darkorange', 'lightgreen', 'red']
fig = go.Figure(data=[go.Pie(labels = data['neighbourhood_group'].value_counts().index,
                             values = NeighbourHoodGrpCnt.values)])
fig.update_traces(hoverinfo='label+percent', textinfo='value+percent', textfont_size=20,
                  marker=dict(colors=colors, line=dict(color='#000000', width=1)))
fig.show()


# In[ ]:


# Most people live in Brooklyn or Manhattan
NGroup = ['Manhattan','Brooklyn']
data = data[data.neighbourhood_group.isin(NGroup)]


# In[ ]:


roomdf = data.groupby(['neighbourhood_group','room_type']).count()['id']
roomdf = pd.DataFrame(roomdf)
roomdf = roomdf.reset_index()
roomdf = roomdf.rename(columns={"id": "Count"})
roomdf


# In[ ]:


# Listed room-types in Manhattan and Brooklyn
colors = ['mediumturquoise', 'gold', 'darkorange', 'lightgreen', 'red']
roomType = ['Entire home/apt', 'Private room', 'Shared room']
neighbourhood_grp = roomdf['neighbourhood_group'].unique().tolist()
fig = go.Figure()
x = 0
for locality in NGroup:
    fig.add_trace(go.Bar(x = roomType, y = roomdf[roomdf.neighbourhood_group == locality].Count.to_list(), name=locality, marker_color=colors[x]))
    x = x + 1
fig.update_layout(barmode='stack',xaxis={'categoryorder':'category ascending'},font=dict(
        family="Segoe UI, monospace",
        size=15,
        color="#7f7f7f"))
fig.show()


# In[ ]:


# Price distribution of room-types in Neighbourhood-group
fig = go.Figure()
x = 0
colors = [ 'gold', 'mediumturquoise', 'darkorange', 'lightgreen', 'red']
roomData = data[data.price < 500]
for locality in neighbourhood_grp:
    fig.add_trace(go.Box(
        y = roomData[roomData.neighbourhood_group == locality].price,
        x = roomData[roomData.neighbourhood_group == locality].room_type,
        name=locality,
        marker_color = colors[x]
    ))
    x = x + 1 
fig.update_layout(
    yaxis_title='Room-Price',
    boxmode='group',
    font=dict(
        family="Segoe UI, monospace",
        size=15,
        color="#7f7f7f")
)
fig.show()
# Average price in Manhattan is more than that of Brooklyn


# In[ ]:


data.head(5)
data['Day'] = pd.to_datetime(data['last_review']).dt.day_name()
data.head(5)


# In[ ]:


# Price distribution of Rooms in Days
fig = go.Figure()
dayData = data[data.price < 500]
fig.add_trace(go.Box(y = dayData.price,x = dayData.Day,marker_color = 'darkorange'
    ))
fig.update_layout(
    yaxis_title='Room-Price',
    boxmode='group',
    font=dict(
        family="Segoe UI, monospace",
        size=15,
        color="#7f7f7f")
)
fig.show()


# In[ ]:


NGroup = ['Manhattan','Brooklyn']
AData = data[data.neighbourhood_group.isin(NGroup)]
AData.groupby(['neighbourhood_group','neighbourhood']).mean()


# In[ ]:


df_top_prices_by_neighbourhood = data.groupby(['neighbourhood_group','neighbourhood']).mean().sort_values('price')['price'].reset_index()
df_top_availability_by_neighbourhood = data.groupby(['neighbourhood_group','neighbourhood']).mean().sort_values('availability_365')['availability_365'].reset_index()
df_top_availability_by_neighbourhood


# In[ ]:


fig = make_subplots(rows=2, cols=1, vertical_spacing = 0.25)
xcnt = 0
rcnt = 1
for ngroup in df_top_prices_by_neighbourhood['neighbourhood_group'].unique().tolist():
    fig.add_trace(
        go.Bar(
            x=df_top_prices_by_neighbourhood[df_top_prices_by_neighbourhood.neighbourhood_group == ngroup].neighbourhood,
            y=df_top_prices_by_neighbourhood[df_top_prices_by_neighbourhood.neighbourhood_group == ngroup].price,
            textposition='auto',
            name=ngroup,
            marker_color = colors[xcnt],
            text = round(df_top_prices_by_neighbourhood[df_top_prices_by_neighbourhood.neighbourhood_group == ngroup].price)),row = rcnt, col =1)
    rcnt = rcnt + 1
    xcnt = xcnt + 1
fig.update_layout(height=1000, width=1400, title_text="Price By Neighbourhood",font=dict(
        family="Segoe UI, monospace",
        size=15,
        color="#7f7f7f"))
fig.show()


# In[ ]:


fig = make_subplots(rows=2, cols=1, vertical_spacing = 0.25)
xcnt = 0
rcnt = 1
for ngroup in df_top_availability_by_neighbourhood['neighbourhood_group'].unique().tolist():
    fig.add_trace(
        go.Bar(
            x=df_top_availability_by_neighbourhood[df_top_availability_by_neighbourhood.neighbourhood_group == ngroup].neighbourhood,
            y=df_top_availability_by_neighbourhood[df_top_availability_by_neighbourhood.neighbourhood_group == ngroup].availability_365,
            textposition='auto',
            name=ngroup,
            marker_color = colors[xcnt],
            text = round(df_top_availability_by_neighbourhood[df_top_availability_by_neighbourhood.neighbourhood_group == ngroup].availability_365)),row = rcnt, col =1)
    rcnt = rcnt + 1
    xcnt = xcnt + 1
fig.update_layout(height=1000, width=1400, title_text="Availability By Neighbourhood",font=dict(
        family="Segoe UI, monospace",
        size=15,
        color="#7f7f7f"))
fig.show()


# In[ ]:


data.head(5)


# In[ ]:


data.minimum_nights.unique()


# In[ ]:


# Remove more than 60
hdata = data[data.minimum_nights<30]
hdata.minimum_nights.unique()


# In[ ]:


colors = ['gold', 'mediumturquoise', 'darkorange']
roomType = ['Private room','Entire home/apt','Shared room']
x1 = hdata[hdata.room_type == 'Private room'].minimum_nights.to_list()
x2 = hdata[hdata.room_type == 'Entire home/apt'].minimum_nights.to_list()
x3 = hdata[hdata.room_type == 'Shared room'].minimum_nights.to_list()
hist_data = [x1, x2, x3]
    
# Create distplot with curve_type set to 'normal'
fig = ff.create_distplot(hist_data, roomType, colors=colors, bin_size=2, show_rug=False,show_hist=False)

# Add title
fig.update_layout(title_text='Distribution Plot for Minimum_Nights')
fig.show()


# 

# In[ ]:


boroughs = gpd.read_file(gplt.datasets.get_path('nyc_boroughs'))
boroughs = boroughs.to_crs(epsg=4326)
boroughs


# In[ ]:


# designate coordinate system
crs = {'init':'espc:4326'}
# zip x and y coordinates into single feature
geometry = [Point(xy) for xy in zip(data['longitude'], data['latitude'])]
# create GeoPandas dataframe
geo_df = gpd.GeoDataFrame(data,
 crs = crs,
 geometry = geometry)
geo_df


# In[ ]:


df_top_prices_by_neighbourhood = data.groupby(['neighbourhood_group','neighbourhood']).mean().sort_values('price').reset_index()
df_top_prices_by_neighbourhood
# designate coordinate system
crs = {'init':'espc:4326'}
# zip x and y coordinates into single feature
geometry = [Point(xy) for xy in zip(df_top_prices_by_neighbourhood['longitude'], df_top_prices_by_neighbourhood['latitude'])]
# create GeoPandas dataframe
geo_df = gpd.GeoDataFrame(df_top_prices_by_neighbourhood,
 crs = crs,
 geometry = geometry)
geo_df


# In[ ]:


fig,ax = plt.subplots()
boroughs.plot(ax=ax,alpha=0.4,edgecolor='black')
geo_df.plot(column='price',ax=ax,legend=True,cmap='Reds',markersize=50)
#ax.set_ylabel("Median Population", fontname="Segoe UI", fontsize=12)
#ax.set_title("Average Price of AirBnb Listings in Neighbourhood",fontname="Segoe UI", fontsize=20)
#plt.title("Average Price of AirBnb Listings in neighbourhood")
plt.axis('off')


# In[ ]:


fig,ax = plt.subplots()
boroughs.plot(ax=ax,alpha=0.4,edgecolor='black')
geo_df.plot(column='availability_365',ax=ax,legend=True,cmap='summer_r',markersize=50)
#ax.set_ylabel("Median Population", fontname="Segoe UI", fontsize=12)
#ax.set_title("Average Price of AirBnb Listings in Neighbourhood",fontname="Segoe UI", fontsize=20)
#plt.title("Average Price of AirBnb Listings in neighbourhood")
plt.axis('off')

