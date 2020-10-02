#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import string
from plotly.offline import init_notebook_mode, iplot
from plotly.graph_objs import *
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import os
print(os.listdir("../input"))
df = pd.read_csv('../input/FastFoodRestaurants.csv')


# In[ ]:


df.head()


# In[ ]:


print('Total no of columns:',df.shape[1])
print('Total no of rows:',df.shape[0])


# In[ ]:


print('countries are:',df.country.unique())
print('city are:',df.city.unique())


# In[ ]:


cityplt=df.city.value_counts()[:10].plot.bar(title='Top 10 cities')
cityplt.set_xlabel('city',size=15)
cityplt.set_ylabel('count',size=15)


# In[ ]:


provplt=df.province.value_counts()[:10].plot.bar(title='Top 10 province')
provplt.set_xlabel('province',size=15)
provplt.set_ylabel('count',size=15)


# In[ ]:


df.name.value_counts()[:10]


# In[ ]:


df.name=df.name.apply(lambda x: x.lower())
df.name=df.name.apply(lambda x:''.join([i for i in x 
                            if i not in string.punctuation]))

df.name.value_counts()[:10]


# In[ ]:


nameplt=df.name.value_counts()[:10].plot.bar(title="Top 10 Restaurants")
nameplt.set_xlabel('Restaurant',size=15)
nameplt.set_ylabel('count',size=15)


# In[ ]:


df['text'] = df['name'] + ',' + df['province'] + ', ' + df['country']

scl = [ [0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],    [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"] ]

data = [ dict(
        type = 'scattergeo',
        locationmode = 'USA-states',
        lon = df['longitude'],
        lat = df['latitude'],
        text = df['text'],
        mode = 'markers',
        marker = dict(
            size = 4,
            opacity = 0.5,
            reversescale = True,
            autocolorscale = False,
            symbol = 'circle',
            line = dict(
                width=.5,
                color='rgba(102, 102, 102)'
            )))]
layout = dict(
        title = 'Restaurants across the country',
        colorbar = True,
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showland = True,
            landcolor = "rgb(250, 250, 250)",
            subunitcolor = "rgb(217, 217, 217)",
            countrycolor = "rgb(217, 217, 217)",
            countrywidth = 0.5,
            subunitwidth = 0.5
        ),
    )

fig = dict( data=data, layout=layout )
iplot( fig, validate=False, filename='all restaurants' )


# In[ ]:


mapbox_access_token='pk.eyJ1IjoibmF2ZWVuOTIiLCJhIjoiY2pqbWlybTc2MTlmdjNwcGJ2NGt1dDFoOSJ9.z5Jt4XxKvu5voCJZBAenjQ'


# In[ ]:


mcd=df[df.name =='mcdonalds']
mcd_lat = mcd.latitude
mcd_lon = mcd.longitude
mcd_city = mcd.city

data = [
    go.Scattermapbox(
        lat=mcd_lat,
        lon=mcd_lon,
        mode='markers',
        marker=dict(
            size=5,
            color='rgb(255, 0, 0)',
            opacity=0.3
        ))]
layout = go.Layout(
    title='Mcdonalds Restaurants',
    autosize=True,
    hovermode='closest',
    showlegend=False,
    mapbox=dict(
        accesstoken=mapbox_access_token,
        bearing=0,
        center=dict(
            lat=38,
            lon=-94
        ),
        pitch=0,
        zoom=3,
        style='light'
    ),
)

fig = dict(data=data, layout=layout)
iplot(fig, filename='Mcdonalds restaurants')


# In[ ]:


bg=df[df.name =='burger king']
bg_lat = bg.latitude
bg_lon = bg.longitude
bg_city = bg.city

data = [
    go.Scattermapbox(
        lat=bg_lat,
        lon=bg_lon,
        mode='markers',
        marker=dict(
            size=5,
            color='rgb(0,255, 0)',
            opacity=0.8
        ))]
layout = go.Layout(
    title='Burgerking Restaurants',
    autosize=True,
    hovermode='closest',
    showlegend=False,
    mapbox=dict(
        accesstoken=mapbox_access_token,
        bearing=0,
        center=dict(
            lat=38,
            lon=-94
        ),
        pitch=0,
        zoom=3,
        style='light'
    ),
)

fig = dict(data=data, layout=layout)
iplot(fig, filename='burgerking restaurants')


# In[ ]:


tb=df[df.name =='taco bell']
tb_lat = tb.latitude
tb_lon = tb.longitude
tb_city = tb.city

data = [
    go.Scattermapbox(
        lat=tb_lat,
        lon=tb_lon,
        mode='markers',
        marker=dict(
            size=5,
            color='rgb(0,0,255)',
            opacity=0.8
        ))]
layout = go.Layout(
    title='Tacobell Restaurants',
    autosize=True,
    hovermode='closest',
    showlegend=False,
    mapbox=dict(
        accesstoken=mapbox_access_token,
        bearing=0,
        center=dict(
            lat=38,
            lon=-94
        ),
        pitch=0,
        zoom=3,
        style='light'
    ),
)

fig = dict(data=data, layout=layout)
iplot(fig, filename='Tacobell restaurants')


# Now we are gonna visualize with Tableau...
# 
# Check out the following link which explains how to obatain tableau plots in python..
# http://datawisesite.wordpress.com/2017/06/26/how-to-embed-tableau-in-jupyter-notebook/
# 

# In[ ]:


get_ipython().run_cell_magic('HTML', '', "<div class='tableauPlaceholder' id='viz1535352712112' style='position: relative'><noscript><a href='#'><img alt='Restaurants ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Re&#47;Restaurantlocations&#47;Sheet1&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='Restaurantlocations&#47;Sheet1' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Re&#47;Restaurantlocations&#47;Sheet1&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1535352712112');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")


# Thankyou for visiting.......Yoloyolo
