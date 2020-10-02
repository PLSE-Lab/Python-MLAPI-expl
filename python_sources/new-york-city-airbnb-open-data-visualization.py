#!/usr/bin/env python
# coding: utf-8

# ![](https://66.media.tumblr.com/fac4306e758efc68ce8ec40b5489c34c/tumblr_p7qz2jdSSN1ruxyndo3_1280.jpg)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.offline as pyo
import plotly.graph_objs as go
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

get_ipython().run_line_magic('matplotlib', 'inline')
from ipywidgets import widgets


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv("../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")


# In[ ]:


df.head()


# In[ ]:


df.info()


# # Removing Outliers 

# In[ ]:


df= df.drop(df[(df['minimum_nights']>365)].index)
df= df.drop(df[(df['number_of_reviews']>500)].index)
df= df.drop(df[(df['calculated_host_listings_count']>100)].index)
df= df.drop(df[(df['price']>1800)].index)
df= df.drop(df[(df['price']<1)].index)

df.describe()


# # Lets Plot something now ;)

# Types of Neighbourhood groups and their price

# In[ ]:



df.neighbourhood_group.unique()


# In[ ]:


neighbour_group_df =df.pivot_table('price', ['neighbourhood_group'], aggfunc='mean').reset_index()


# In[ ]:


fig = px.bar(neighbour_group_df, x='neighbourhood_group', y='price',
             hover_data=['price'], color='price',  barmode ='relative',
             labels={'pop':'Neighbourhood group and their pricing'}, height=400, width=800)

fig.show()


# In[ ]:



fig = px.histogram(df, x="neighbourhood_group", color = 'neighbourhood_group', height=600, width=800, )
fig.update_layout(showlegend = True)
fig.show()


# Checking the types of room in the NewYork Aibnbs

# In[ ]:


roomdf = df.groupby('room_type').size()/df['room_type'].count()*100
labels = roomdf.index
values = roomdf.values

# Use `hole` to create a donut-like pie chart
fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
fig.show()


# In[ ]:


neighbour_df =df.pivot_table(['price', 'number_of_reviews', 'calculated_host_listings_count', 'neighbourhood_group' ] , ['neighbourhood'], aggfunc='mean').reset_index()


# In[ ]:


fig = px.scatter(neighbour_df, x="neighbourhood", y="price", color="calculated_host_listings_count",
                 size='price', height=500, width=800)
fig.update_layout(showlegend = False)
fig.show()


# The one Neighbourhood which stands out from the above visualization is **Murray hill**  having the maximum average host listing counts.

# In[ ]:


fig = px.histogram(df, x="price", color = 'neighbourhood_group',marginal="rug",  hover_data=df.columns, height=600, width=800, )
fig.update_layout(showlegend = False)
fig.show()


# We can clearly see that the **Staten Island ** have most airbnbs with minimum price 
# 
# Now checking the Airbnbs with low pricing and high number of reviews

# In[ ]:


airbnb_100 = df.nsmallest(200,'price')
fig = px.scatter(airbnb_100, x="host_name", y="reviews_per_month", color="price", size = 'calculated_host_listings_count', height=500, width=800)
fig.update_layout(showlegend = False)
fig.show()


# Of all the Airbnbs with low price there are some which stands out. Here is the listing of Top 5 Airbnbs with low price and high reviews per month.
#  
# 1. Beautiful furnished private studio with backyard hosted by Melissa having the price **20 dollars**
# 2. 8mins to JFK airport, separate door & bathroom hosted by Modesta having the price  **25 dollars**
# 3. Private room with visit to queens #4 hosted by Sonia having the price **25 dollars**
# 4. Happy Home 3 by Raquek having the price **13 dollars**
# 5. Spacious 2-bedroom Apt in Heart of Greenpoint hosted by Vishanti and Jeremy having the price **10 dollars**
# 
# ----------------------------------------------------------------------------------------------------------------

#  Let's check the Airbnbs with high price and lets try to find some insights

# In[ ]:


large_airbnb_200 = df.nlargest(200,'price')
fig = px.scatter(large_airbnb_200, x="price", y="reviews_per_month", color="neighbourhood_group", size = 'calculated_host_listings_count', 
                 hover_data=large_airbnb_200.columns, height=500, width=800)
fig.update_layout(showlegend = False)
fig.show()


# From the above Visualization it's clearly visible that most Airbnbs are in **Manhattan** .
# Out of all these Airbnbs the one which stands out is **Empire city- king Lux King room** hosted by Gabriel.
# 
# Feel free to hover the mouse over these data points and extract out the information

# In[ ]:





# Now lets plot these areas and see what information can be extracted

# In[ ]:


import plotly.express as px

mapbox_access_token = 'pk.eyJ1IjoiYmlkZHkiLCJhIjoiY2pxNWZ1bjZ6MjRjczRhbXNxeG5udzkyNSJ9.xX6QLOAcoBmXZdUdocAeuA'
px.set_mapbox_access_token(mapbox_access_token)
fig = px.scatter_mapbox(df, lat="latitude", lon="longitude", color="neighbourhood_group", size = 'price', opacity= 0.8,
                  color_continuous_scale=px.colors.cyclical.IceFire, size_max=16, zoom=9.2,height=400, width=800 )
fig.update_layout(
    mapbox_style="white-bg",
    showlegend = False,
    mapbox_layers=[
        {
            "below": 'traces',
            "sourcetype": "raster",
            "source": [
                "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"
            ]
        },
      ]
)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":4})
fig.show()


# Done for the day, I'll upload some more interactive visualization tomorrow. Give me a thumbs up if you like it :)
