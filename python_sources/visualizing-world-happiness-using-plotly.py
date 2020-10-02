#!/usr/bin/env python
# coding: utf-8

# In[6]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.cluster import KMeans
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[9]:


df = pd.read_csv('../input/world-happiness/2015.csv')
print(df.head())
map_data = pd.read_csv('../input/countries/countries.csv')
map_data.head()


# 
# ### Choropleth
# This map shows the happiness rank of all the countries in the world in 2015. The darker the colour, the higher the rank, i.e. the happier the people in that country. It can be seen that the countries of North America (the US, Mexico and Canada), Australia, New Zealand and the western countries of Europe have the happiest citizens. Countries that have lower happiness scores are the ones that are either war struck (eg., Iraq) or are highly underdeveloped as can be said of the countries in Africa like Congo and Chad. The top 5 countries are:
# 1. Switzerland
# 2. Iceland
# 3. Denmark
# 4. Norway
# 5. Canada
# 

# In[10]:


scl = [[0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],            [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(240, 210, 250)"]]


# scl = [[0.0, 'rgb(50,10,143)'],[0.2, 'rgb(117,107,177)'],[0.4, 'rgb(158,154,200)'],\
#             [0.6, 'rgb(188,189,220)'],[0.8, 'rgb(218,208,235)'],[1.0, 'rgb(250,240,255)']]
data = dict(type = 'choropleth', 
            colorscale = scl,
            autocolorscale = True,
            reversescale = True,
           locations = df['Country'],
           locationmode = 'country names',
           z = df['Happiness Rank'], 
           text = df['Country'],
           colorbar = {'title':'Happiness'})
layout = dict(title = 'Global Happiness', 
             geo = dict(showframe = False, 
                       projection = {'type': 'Orthographic'}))

choromap3 = go.Figure(data = [data], layout=layout)
iplot(choromap3)


# ### Symbol Map
# To take our analysis further, we plot a symbol map where the size of the circles represents the Happiness Score while the colour represents the GDP of the countries. Larger the circle, happier the citizens while darker the circle, higher the GDP. From this plot, we can see that the top 5 countries we have seen above definitely have a much higher GDP. **This seems to imply that more well off a country is economically, the happier its citizens are.** Also, the underdeveloped countries like Chad, Congo, Burundi and Togo have a very low GDP and also a very low happiness score. While we cannot directly say that low GDP implies lower happiness, it seems like an important factor. This trend remain consistent throughout all the countries. **There are almost no countries that have a high GDP but low happiness index or vice versa.**
# 
# Also geographic location and neighbours may be playing an important role. We can clusters/regions with countries having similar GDPs and similar Happiness Scores. **So countries that have a good economy and good relations with their neighbours benefit from mutual growth and this is also reflected in their happiness scores. Countries that have disturbed neighbourhoods, like in middle east Asia (Iraq, Afghanistan, etc.), show much lower growth/economic prosperity as well as lower happiness scores.**
# 
# One thing that can also be noted is that in general, countries which are known for their lower population densities (https://www.worldatlas.com/articles/the-10-least-densely-populated-places-in-the-world-2015.html) like Denmark and Iceland are much happier than the more densly populated countries.

# In[11]:


df = df.merge(map_data, how='left', on = ['Country'])
df.head()


# In[12]:


df['Happiness Score'].min(), df['Happiness Score'].max()


# In[13]:


df['text']=df['Country'] + '<br>Happiness Score ' + (df['Happiness Score']).astype(str)
scl = [ [0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],    [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"] ]

data = [ dict(
        type = 'scattergeo',
        locationmode = 'country names',
        lon = df['Longitude'],
        lat = df['Latitude'],
        text = df['text'],
        mode = 'markers',
        marker = dict(
            size = df['Happiness Score']*3,
            opacity = 0.8,
            reversescale = True,
            autocolorscale = False,
            symbol = 'circle',
            line = dict(
                width=1,
                color='rgba(102, 102, 102)'
            ),
            colorscale = scl,
            cmin = 0,
            color = df['Economy (GDP per Capita)'],
            cmax = df['Economy (GDP per Capita)'].max(),
            colorbar=dict(
                title="GDP per Capita"
            )
        ))]

layout = dict(
        title = 'Happiness Scores by GDP',
        geo = dict(
#             scope='usa',
            projection=dict( type='Mercator' ),
            showland = True,
            landcolor = "rgb(250, 250, 250)",
#             subunitcolor = "rgb(217, 217, 217)",
#             countrycolor = "rgb(217, 217, 217)",
            countrywidth = 0.5,
            subunitwidth = 0.5
        ),
    )

symbolmap = go.Figure(data = data, layout=layout)

iplot(symbolmap)


# In[ ]:




