#!/usr/bin/env python
# coding: utf-8

# # San Francisco Crime Classification

# In this notebook I will analyse the San Francisco Crime Classification dataset.Its a huge dataset  with more than 800k entries.Lets see what's this data is to offer.

# ## Overview

# From 1934 to 1963, San Francisco was infamous for housing some of the world's most notorious criminals on the inescapable island of Alcatraz.
# 
# Today, the city is known more for its tech scene than its criminal past. But, with rising wealth inequality, housing shortages, and a proliferation of expensive digital toys riding BART to work, there is no scarcity of crime in the city by the bay.
# 
# From Sunset to SOMA, and Marina to Excelsior, this  dataset provides nearly 12 years of crime reports from across all of San Francisco's neighborhoods

# ## Getting the Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
get_ipython().run_line_magic('matplotlib', 'inline')
import folium
from folium import plugins
from folium.plugins import HeatMap
from folium.plugins import FastMarkerCluster
from folium.plugins import MarkerCluster

from subprocess import check_output
print(check_output(['ls','../input']).decode('utf8'))


# ## Loading the Data

# In[2]:


train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')


# ## What is the shape of the data??

# In[3]:


print('The shape of the training data is:',train.shape)
print('The shape of the test data is :',test.shape)


# 878049 observations.A huge dataset

# ## Glimpse of the data

# Printing the top 5 observation of the train data

# In[4]:


train.head(5)


# Is there any null value present in the train data ?

# In[5]:


train.columns[train.isnull().any()]


# ## Feature Engineering

# ### Extracting Feature from Dates Feature

# Use the date feature to get the year,month and hour features

# In[6]:


train=pd.read_csv('../input/train.csv',parse_dates=['Dates'])


# Year Feature

# In[8]:


train['Year']=train['Dates'].dt.year


# Month Feature

# In[9]:


train['Month']=train['Dates'].dt.month


# Hour Feature

# In[10]:


train['Hour']=train['Dates'].dt.hour


# In[11]:


train.head()


# ### Extracting Feature from the Address Feature

# In[12]:


def street_addr(x):
    street=x.split(' ')
    return (' '.join(street[-2:]))


# In[13]:


train['addr']=train['Address'].apply(lambda x:street_addr(x))
train['addr'].head()


# ## Exploratory Data Analysis

# ### Most Common Crimes(Top 10)

# Lets look for the top crimes of San Fransisco

# In[14]:


commo_crime=train['Category'].value_counts().sort_values(ascending=False).reset_index().head(10)
commo_crime.columns=['Crime','Count']
data = [go.Bar(
            x=commo_crime.Crime,
            y=commo_crime.Count,
             opacity=0.6
    )]

py.iplot(data, filename='basic-bar')


# ### Districts most vulnerable to a Crime

# In[15]:


train['PdDistrict'].value_counts()


# In[16]:


commo_dis=train['PdDistrict'].value_counts().sort_values(ascending=False).reset_index().head(10)
commo_dis.columns=['District','Count']
data = [go.Bar(
            y=commo_dis.District,
            x=commo_dis.Count,
             opacity=0.6,
             orientation = 'h'
    )]

py.iplot(data, filename='basic-bar')


# ### Crime Distribution over the years

# In[17]:


train['Year'].value_counts()


# We will see this crime data over the years so see what is the trean

# In[18]:


year_count=train['Year'].value_counts().reset_index().sort_values(by='index')
year_count.columns=['Year','Count']
# Create a trace
trace = go.Scatter(
    x = year_count.Year,
    y = year_count.Count
)

data = [trace]

py.iplot(data, filename='basic-line')


# We can clearly see that there is growing criminal cases from year 2010 to 2014.The dip in 2015 is may be due to less data from year 2015.

# ### Top address with most crimes

# Here we will see the street names where most crime took place.

# In[19]:


train['addr'].value_counts().head(10)


# Lets form a pie chart to plot the data

# In[20]:


year_count=train['addr'].value_counts().reset_index().sort_values(by='index').head(10)
year_count.columns=['addr','Count']
# Create a trace
tag = (np.array(year_count.addr))
sizes = (np.array((year_count['Count'] / year_count['Count'].sum())*100))
plt.figure(figsize=(15,8))

trace = go.Pie(labels=tag, values=sizes)
layout = go.Layout(title='Top Address with Most Crimes')
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="Activity Distribution")


# ### Crime Distribution over the Month

# In[21]:


month=train['Month'].value_counts().sort_values(ascending=False).reset_index()
month.columns=['Month','Count']
data = [go.Bar(
            x=month.Month,
            y=month.Count,
             opacity=0.6
    )]

py.iplot(data, filename='basic-bar')


# This plot shows the crime of every month for all the years.Lets plot the criminal activity for every month of the year.

# In[22]:


data=[]
for i in range(2003,2015):
    year=train[train['Year']==i]
    year_count=year['Month'].value_counts().reset_index().sort_values(by='index')
    year_count.columns=['Month','Count']
    trace = go.Scatter(
    x = year_count.Month,
    y = year_count.Count,
    name = i)
    data.append(trace)
    

py.iplot(data, filename='basic-line')
    


# ### Crime Distribution of Districts over the Year

# In[23]:


val=train['PdDistrict'].value_counts().reset_index()
val.columns=['District','Count']
x=val.District
data=[]
for i in x:
    district=train[train['PdDistrict']==i]
    year_count=district['Year'].value_counts().reset_index().sort_values(by='index')
    year_count.columns=['Year','Count']
    trace = go.Scatter(
    x = year_count.Year,
    y = year_count.Count,
    name = i)
    data.append(trace)
    

py.iplot(data, filename='basic-line')


# Southern District seems to be a criminal hotspot over the years.There is a sharp rise in criminal activity of Tenderloin from 2006 to 2008

# ## Visualize the Criminal Activity on the Map

# In[24]:


m = folium.Map(
    location=[train.Y.mean(), train.X.mean()],
    tiles='Cartodb Positron',
    zoom_start=13
)

marker_cluster = MarkerCluster(
    name='Crime Locations',
    overlay=True,
    control=False,
    icon_create_function=None
)
for k in range(1000):
    location = train.Y.values[k], train.X.values[k]
    marker = folium.Marker(location=location,icon=folium.Icon(color='green'))
    popup = train.addr.values[k]
    folium.Popup(popup).add_to(marker)
    marker_cluster.add_child(marker)

marker_cluster.add_to(m)

folium.LayerControl().add_to(m)

m.save("marker cluster south asia.html")

m


# Lets create a heatmap to show criminal hotspot.Here we will be using top 1000 crimes

# In[25]:


M= folium.Map(location=[train.Y.mean(), train.X.mean() ],tiles= "Stamen Terrain",
                    zoom_start = 13) 

# List comprehension to make out list of lists
heat_data = [[[row['Y'],row['X']] 
                for index, row in train.head(1000).iterrows()] 
                 for i in range(0,11)]
#print(heat_data)
# Plot it on the map
hm = plugins.HeatMapWithTime(heat_data,auto_play=True,max_opacity=0.8)
hm.add_to(M)

hm.save('world MPI heatmap.html')

# Display the map
M


# ## Locations of Top 5 Crimes

# In[26]:


top_crime=train['Category'].value_counts().reset_index().head(5)
top_crime


# ### Location for Occurence of Larceny/Theft

# In[27]:


cat='LARCENY/THEFT'
new=train[train['Category']==cat]
m = folium.Map(
    location=[train.Y.mean(), train.X.mean()],
    tiles='Cartodb Positron',
    zoom_start=13
)

marker_cluster = MarkerCluster(
    name='Crime Locations',
    overlay=True,
    control=False,
    icon_create_function=None
)
for k in range(1000):
    location = new.Y.values[k], new.X.values[k]
    marker = folium.Marker(location=location,icon=folium.Icon(color='green'))
    popup = new.addr.values[k]
    folium.Popup(popup).add_to(marker)
    marker_cluster.add_child(marker)

marker_cluster.add_to(m)

folium.LayerControl().add_to(m)

m.save("marker cluster south asia.html")

m


# ### Location for occurrence of Other Offenses

# In[28]:


cat='OTHER OFFENSES'
new=train[train['Category']==cat]
m = folium.Map(
    location=[train.Y.mean(), train.X.mean()],
    tiles='Cartodb Positron',
    zoom_start=13
)

marker_cluster = MarkerCluster(
    name='Crime Locations',
    overlay=True,
    control=False,
    icon_create_function=None
)
for k in range(1000):
    location = new.Y.values[k], new.X.values[k]
    marker = folium.Marker(location=location,icon=folium.Icon(color='green'))
    popup = new.addr.values[k]
    folium.Popup(popup).add_to(marker)
    marker_cluster.add_child(marker)

marker_cluster.add_to(m)

folium.LayerControl().add_to(m)

m.save("marker cluster south asia.html")

m


# ### Location for Non-Criminal Activity

# In[29]:


new=train[train['Category']=='NON-CRIMINAL']
M= folium.Map(location=[train.Y.mean(), train.X.mean() ],tiles= "Stamen Terrain",
                    zoom_start = 13) 

# List comprehension to make out list of lists
heat_data = [[[row['Y'],row['X']] 
                for index, row in new.head(1000).iterrows()] 
                 for i in range(0,11)]
#print(heat_data)
# Plot it on the map
hm = plugins.HeatMapWithTime(heat_data,auto_play=True,max_opacity=0.8)
hm.add_to(M)

hm.save('SanFran heatmap.html')

# Display the map
M


# ### Locations For Assault

# In[30]:


new=train[train['Category']=='ASSAULT']
M= folium.Map(location=[train.Y.mean(), train.X.mean() ],tiles= "Stamen Terrain",
                    zoom_start = 13) 

# List comprehension to make out list of lists
heat_data = [[[row['Y'],row['X']] 
                for index, row in new.head(1000).iterrows()] 
                 for i in range(0,11)]
#print(heat_data)
# Plot it on the map
hm = plugins.HeatMapWithTime(heat_data,auto_play=True,max_opacity=0.8)
hm.add_to(M)

hm.save('SanFran heatmap.html')

# Display the map
M


# ## More to come...
