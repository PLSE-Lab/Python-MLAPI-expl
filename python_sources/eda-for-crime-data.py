#!/usr/bin/env python
# coding: utf-8

# * [1.DataOverview](#1.DataOverview)
# * [2.Visualization](#2.Visualization)
#     * [2.1 visualization for crime type](#21)
#     * [2.2 visualization for the number of crime](#22)
#     * [2.3 visualization for 'traffic-accident'](#23)
#     * [2.4 visualization for 'larceny'](#24)
#     * [2.5 visualization for 'NEIGHBORHOOD_ID'](#25)
# * [3.Suggestion](#3.Suggestion)

# In[ ]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import os 
import seaborn as sns
import geopandas as gpd
import folium
from folium import plugins
import datetime
import math


# In[ ]:


os.listdir('../input/denver-crime-data')


# # 1.DataOverview

# The crime data size is 509346,and the variables include INCIDENT_ID,OFFENSE_ID,and so on.
# 
# 'LAST_OCCURRENCE_DATE', 'INCIDENT_ADDRESS', 'GEO_LAT', 'GEO_LON','GEO_Y', 'GEO_X' have missing value
# 
# If you want to use these variables,you have to watch out missing value.
# 
# In this kernel,I will use the Geo_LON and GEO_LAT to visualize the data,so I will remove the missing value from these variables.

# In[ ]:


data=pd.read_csv('../input/denver-crime-data/crime.csv')
data.head()


# In[ ]:


y=data.isnull().sum().sort_values(ascending=False)[:6].index
x=data.isnull().sum().sort_values(ascending=False)[:6]
plt.figure(figsize=(8,8))
sns.barplot(x,y)
plt.title("counts of missing value",size=20)


# In[ ]:


data=data.dropna(subset=['GEO_LAT','GEO_LON'])
data.isnull().sum()


# In[ ]:


data['REPORTED_DATE']=data.REPORTED_DATE.apply(lambda x:datetime.datetime.strptime(x,'%m/%d/%Y %I:%M:%S %p'))
data['year']=data.REPORTED_DATE.apply(lambda x:x.strftime('%Y'))
data['month']=data.REPORTED_DATE.apply(lambda x:x.strftime('%m'))
data['hour']=data.REPORTED_DATE.apply(lambda x:x.strftime('%H'))
data.head()


# I remove the data whcih LAT is less than 39

# In[ ]:


data=data[data.GEO_LAT>39]


# # 2.Visualization

# <h3 id="21">2.1 visualization for crime type</h3>

# From the bar chart, we know that the most of crime type is traffic-accident in this data.

# In[ ]:


Top10_crime_type=data[data['OFFENSE_CATEGORY_ID'].isin(list(data.OFFENSE_CATEGORY_ID.value_counts()[:10].index[:10]))]
fig,ax=plt.subplots(2,2,figsize=(20,20))
y=Top10_crime_type.OFFENSE_CATEGORY_ID.value_counts().index
x=Top10_crime_type.OFFENSE_CATEGORY_ID.value_counts()
sns.barplot(x=x,y=y,ax=ax[0,0])
ax[0,0].set_title("Top 10 crime type by counts",size=20)
ax[0,0].set_xlabel('counts',size=18)
ax[0,0].set_ylabel('')


Top10_crime_type.groupby(['year','OFFENSE_CATEGORY_ID'])['INCIDENT_ID'].agg('count').unstack('OFFENSE_CATEGORY_ID').plot(ax=ax[0,1])
ax[0,1].legend(loc=0, ncol=1, fontsize=14,bbox_to_anchor=(1.10,1))
ax[0,1].set_title("Top 10 crime type counts by year",size=20)
ax[0,1].set_ylabel('counts',size=18)
ax[0,1].set_xlabel('year',size=18)

Top10_crime_type.groupby(['month','OFFENSE_CATEGORY_ID'])['INCIDENT_ID'].agg('count').unstack('OFFENSE_CATEGORY_ID').plot(ax=ax[1,0])
ax[1,0].legend(loc=0, ncol=1, fontsize=14,bbox_to_anchor=(-0.25,1))
ax[1,0].set_title("Top 10 crime type counts by month",size=20)
ax[1,0].set_ylabel('counts',size=18)
ax[1,0].set_xlabel('month',size=18)

sns.scatterplot(x="GEO_LON", y="GEO_LAT", hue="OFFENSE_CATEGORY_ID",data=Top10_crime_type,ax=ax[1,1])
ax[1,1].legend(loc=0, ncol=1, fontsize=14,bbox_to_anchor=(1.10,1))
ax[1,1].set_title("The distribution of Top 10 crime type",size=20)
ax[1,1].set(xlabel='Longitude', ylabel='LATITUDE')


# I use the top 2000 data to visualize on map,you can see the crime's location and type on map.

# In[ ]:


data_2000=data[:2000]
Long=data_2000.GEO_LON.mean()
Lat=data_2000.GEO_LAT.mean()
data_map=folium.Map([Lat,Long],zoom_start=12)

data_crime_map=plugins.MarkerCluster().add_to(data_map)
for lat,lon,label in zip(data_2000.GEO_LAT,data_2000.GEO_LON,data_2000.OFFENSE_CATEGORY_ID):
    folium.Marker(location=[lat,lon],icon=None,popup=label).add_to(data_crime_map)
data_map.add_child(data_crime_map)

data_map


# <h3 id="22">2.2 visualization for the number of crime</h3>

# According to the following chart,we know that the crime almost are happened at 12~18 o'clock.

# In[ ]:


fig,ax=plt.subplots(2,2,figsize=(20,20))
y=data.year.value_counts()
x=data.year.value_counts().index
sns.barplot(x=x,y=y,ax=ax[0,0])
ax[0,0].set_title("The number of crimes by year",size=20)
ax[0,0].set_ylabel('counts',size=18)
ax[0,0].set_xlabel('')


y=data.month.value_counts()
x=data.month.value_counts().index
sns.barplot(x=x,y=y,ax=ax[0,1])
ax[0,1].set_title("The number of crimes by month",size=20)
ax[0,1].set_ylabel('counts',size=18)
ax[0,1].set_xlabel('')

y=data.hour.value_counts()
x=data.hour.value_counts().index
sns.barplot(x=x,y=y,ax=ax[1,0])
ax[1,0].set_title("The number of crimes by hour",size=20)
ax[1,0].set_ylabel('counts',size=18)
ax[1,0].set_xlabel('')




y=data.NEIGHBORHOOD_ID.value_counts()[:10].index
x=data.NEIGHBORHOOD_ID.value_counts()[:10]
sns.barplot(x=x,y=y,ax=ax[1,1])
ax[1,1].set_title("The 10 NEIGHBORHOOD_ID by the number of crimes",size=20)
ax[1,1].set_xlabel('counts',size=18)
ax[1,1].set_ylabel('')


# In[ ]:


map_all=folium.Map([39.7,-105],zoom_start=12)
crime_new=pd.DataFrame({"Lat":data['GEO_LAT'],"Long":data['GEO_LON']})
crime_new=crime_new[:20000]
map_all.add_child(plugins.HeatMap(data=crime_new))
map_all


# <h3 id="23">2.3 visualization for 'traffic-accident'</h3>

# In[ ]:


data_traf=data[data.OFFENSE_CATEGORY_ID=='traffic-accident']
data_traf.head()


# In[ ]:


fig,ax=plt.subplots(2,2,figsize=(20,20))
y=data_traf.year.value_counts()
x=data_traf.year.value_counts().index
sns.barplot(x=x,y=y,ax=ax[0,0])
ax[0,0].set_title("The number of traffic-accident by year",size=20)
ax[0,0].set_ylabel('counts',size=18)
ax[0,0].set_xlabel('')


y=data_traf.month.value_counts()
x=data_traf.month.value_counts().index
sns.barplot(x=x,y=y,ax=ax[0,1])
ax[0,1].set_title("The number of traffic-accident by month",size=20)
ax[0,1].set_ylabel('counts',size=18)
ax[0,1].set_xlabel('')

y=data_traf.hour.value_counts()
x=data_traf.hour.value_counts().index
sns.barplot(x=x,y=y,ax=ax[1,0])
ax[1,0].set_title("The number of traffic-accident by hour",size=20)
ax[1,0].set_ylabel('counts',size=18)
ax[1,0].set_xlabel('')


sns.scatterplot(x="GEO_LON", y="GEO_LAT", hue="NEIGHBORHOOD_ID",data=data_traf,ax=ax[1,1])
ax[1,1].legend(loc=0, ncol=1, fontsize=14,bbox_to_anchor=(1.10,2.5))
ax[1,1].set_title("The distribution of traffic-accident",size=20)
ax[1,1].set(xlabel='Longitude', ylabel='LATITUDE')


# In[ ]:


data_IS_TRAFFIC=data[data.IS_TRAFFIC==1]
data_IS_TRAFFIC.head()


# According to the map,you can see the distribution of traffic-accident by year clearly.If the circle is more large,the traffic-accidents are happened more frequently in this area. I only show top 2000 data on the map.Since it can not show on the map if the data size too large in KAGGLE.

# In[ ]:


data_IS_TRAFFIC=data_IS_TRAFFIC[:2000]
colors = {'2014' : 'red', '2015' : 'blue','2016' :'green','2017':'brown','2018':'plum','2019':'purple'}
Long=data_IS_TRAFFIC.GEO_LON.mean()
Lat=data_IS_TRAFFIC.GEO_LAT.mean()
data_IS_TRAFFIC_map=folium.Map([Lat,Long],zoom_start=12)
for i in range(len(data_IS_TRAFFIC.groupby(['GEO_LAT','GEO_LON'])['INCIDENT_ID'].agg('count').index)):
    lat,lon=data_IS_TRAFFIC.groupby(['GEO_LAT','GEO_LON'])['INCIDENT_ID'].agg('count').index[i]
    folium.Circle(location=[lat,lon],
    popup=data_IS_TRAFFIC.iloc[i]['OFFENSE_TYPE_ID'],
    radius=int(data_IS_TRAFFIC.groupby(['GEO_LAT','GEO_LON'])['INCIDENT_ID'].agg('count')[i])*70,
    fill=True,
    fill_color=colors[data_IS_TRAFFIC['year'].iloc[i]],
    fill_opacity=0.7,).add_to(data_IS_TRAFFIC_map)

data_IS_TRAFFIC_map


# In[ ]:


data_IS_TRAFFIC=data[data.IS_TRAFFIC==1]
plt.figure(figsize=(20,20))
for i in range(6):
    traffic=data_IS_TRAFFIC[data_IS_TRAFFIC.year==str(2014+i)]
    plt.subplot(3,2,i+1)
    plt.scatter('GEO_LON', 'GEO_LAT', data=traffic, c=colors[traffic['year'].iloc[0]])
    plt.title("The distribution of traffic-accident in "+str(2014+i),size=20)
    plt.xlabel('Longitude')
    plt.ylabel('LATITUDE')


# <h3 id="24">2.4 visualization for 'larceny'</h3>

# In[ ]:


data_larceny=data[data.OFFENSE_CATEGORY_ID=='larceny']
data_larceny.head()


# In[ ]:


fig,ax=plt.subplots(2,2,figsize=(20,20))
y=data_larceny.year.value_counts()
x=data_larceny.year.value_counts().index
sns.barplot(x=x,y=y,ax=ax[0,0])
ax[0,0].set_title("The number of larceny by year",size=20)
ax[0,0].set_ylabel('counts',size=18)
ax[0,0].set_xlabel('')


y=data_larceny.month.value_counts()
x=data_larceny.month.value_counts().index
sns.barplot(x=x,y=y,ax=ax[0,1])
ax[0,1].set_title("The number of larceny by month",size=20)
ax[0,1].set_ylabel('counts',size=18)
ax[0,1].set_xlabel('')

y=data_larceny.hour.value_counts()
x=data_larceny.hour.value_counts().index
sns.barplot(x=x,y=y,ax=ax[1,0])
ax[1,0].set_title("The number of larceny by hour",size=20)
ax[1,0].set_ylabel('counts',size=18)
ax[1,0].set_xlabel('')


sns.scatterplot(x="GEO_LON", y="GEO_LAT", hue="NEIGHBORHOOD_ID",data=data_larceny,ax=ax[1,1])
ax[1,1].legend(loc=0, ncol=1, fontsize=14,bbox_to_anchor=(1.10,2.5))
ax[1,1].set_title("The distribution of larency",size=20)
ax[1,1].set(xlabel='Longitude', ylabel='LATITUDE')


# <h3 id="25">2.5 visualization for 'NEIGHBORHOOD_ID'</h3>

# From the rank plot, the neighborhood is dangerous if the area of the plot is large.

# In[ ]:


def rank_NEIGHBORHOOD(NEIGHBORHOOD_ID):
    year=['2014','2015','2016','2017','2018','2019']
    B={}
    for i in range(len(year)):
        A=data[data.year==year[i]]
        value=A.groupby(['NEIGHBORHOOD_ID'])['OFFENSE_ID'].agg('count')
        rank=A.groupby(['NEIGHBORHOOD_ID'])['OFFENSE_ID'].agg('count').rank(method='min',ascending=False)
        new=pd.DataFrame({'rank':rank,'value':value})
        B['rank '+year[i]]=str(new[new.index==NEIGHBORHOOD_ID].iloc[0,0])+"/"+str(max(rank))
        B['value '+year[i]]=str(new[new.index==NEIGHBORHOOD_ID].iloc[0,1])

    return B


# In[ ]:


def rank_plot(NEIGHBORHOOD_ID):
    ID=rank_NEIGHBORHOOD(NEIGHBORHOOD_ID)
    y=[]
    x=[]
    n=[]
    for i in range(6):
        r1,r2=ID['rank '+str(i+2014)].split('/')
        R=float(r1)/float(r2)
        R=1-R
        y.append(1.5+R*math.sin(0+i*2*math.pi/6))
        x.append(1.5+R*math.cos(0+i*2*math.pi/6))
        n.append('rank '+str(i+2014)+' '+ID['rank '+str(i+2014)])
    
    x.append(x[0])
    y.append(y[0])
    plt.plot(x,y, marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=2)
    for i, txt in enumerate(n):
        plt.annotate(txt, (x[i], y[i]))
        plt.xlim(0.45,2.7)
        plt.ylim(0.45,2.7)
        plt.fill(x, y,"plum")
        plt.plot( 1.5, 1.5, marker='o', markerfacecolor='blue', markersize=8, linewidth=2)
        plt.title("The rank of the number of crime by year in "+NEIGHBORHOOD_ID,size=18) 


# In[ ]:


plt.figure(figsize=(20,30))
plt.subplot(3,2,1)
rank_plot('five-points')
plt.subplot(3,2,2)
rank_plot('stapleton')
plt.subplot(3,2,3)
rank_plot('cbd')
plt.subplot(3,2,4)
rank_plot('capitol-hill')
plt.subplot(3,2,5)
rank_plot('virginia-village')
plt.subplot(3,2,6)
rank_plot('city-park')


# # 3.Suggestion

# 1.From the rank plot,the neighborhood is dangerous if the area of the plot is large.Police have to pay more attention to five-points,stapleton,cbd,and capitol-hill.
# 
# 2.According to the following chart from 2.2,we know that the crime almost are happened at 12~18 o'clock.
# 
# 3.From the bar chart of 2.1, we know that the most of crime type is traffic-accident in this data.
