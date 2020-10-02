#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df = pd.read_csv("../input/world-cities-database/worldcitiespop.csv")
df.head()


# Import Country Codes and combine with dataset

# In[ ]:


codes = pd.read_csv('../input/countries-iso-codes/wikipedia-iso-country-codes.csv')
codes.head()


# Convert df['Country'] values to to upper case for joining with codes['Alpha-2 code]

# In[ ]:


df['Alpha-2 code'] = df['Country'].apply(str.upper)


# Zaire was renamed Democratic Republic of the Congo, this dataset contains old codes and will be replaced.

# In[ ]:


df['Alpha-2 code'] = df['Alpha-2 code'].replace('ZR', 'CD')
df = df.merge(codes, on='Alpha-2 code', how='outer')


# Check for Null Population, Lat, and Long, Alpha-2 code, English short name lower case, and City Values

# In[ ]:


def check_for_null(df,columns):
    for i in columns:
        print('Null {value} values found:'.format(value=i))
        print(df[df[i].isnull()].shape[0])

check_for_null(df,['Population','Latitude','Longitude','Alpha-2 code','English short name lower case','City'])


# In[ ]:


df[df['Alpha-2 code'].isnull()]


# In[ ]:


df[df['English short name lower case'].isnull()].head()


# Null values found, removing from dataset.

# In[ ]:


to_remove = ['Population','City','English short name lower case']
df = df.dropna(subset=to_remove)
check_for_null(df,to_remove)


# Select cities with populations greater than 1 million.

# In[ ]:


df = df[df["Population"] > 1000000]
df.head()


# In[ ]:


from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go

init_notebook_mode(connected=True)


# In[ ]:


def combine(ls):
    text = ls[0] + ': ' + str(int(ls[1]))
    return text

df['text'] = df[['AccentCity','Population']].apply(combine,axis=1)


# In[ ]:


trace = dict(type='scattergeo', 
             lon = df['Longitude'], 
             lat = df["Latitude"],
             text = df['text'],
             marker=dict(size = df['Population']/1000000),
                         mode = 'markers')

iplot([trace])


# In[ ]:


limits = [(0,2),(3,7),(8,16),(17,32),(33,64)]
colors = ["rgb(0,116,217)","rgb(255,65,54)","rgb(133,20,75)","rgb(255,133,27)","lightgrey"]
cities = []
scale = 100000

for i in range(len(limits)):
    lim = limits[i]
    min_pop = lim[0]
    max_pop = lim[1]
    
    df_sub = df[(df['Population']/1000000 > lim[0]) & (df['Population']/1000000 < lim[1])]
    
    city = dict(
        type = 'scattergeo',
        lon = df_sub['Longitude'],
        lat = df_sub['Latitude'],
        text = df_sub['text'],
        marker = dict(
            size = df_sub['Population']/scale,
            color = colors[i],
            line = dict(width=0.5, color='rgb(40,40,40)'),
            sizemode = 'area'
        ),
        name = '{0} - {1}'.format(lim[0],lim[1]) )
    cities.append(city)

layout = dict(
        title = 'Cities with populations greater than 1 Million<br>(Click legend to toggle traces)',
        showlegend = True)

fig = dict( data=cities, layout=layout )
iplot( fig, validate=False)

