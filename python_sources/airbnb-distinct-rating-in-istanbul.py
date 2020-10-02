#!/usr/bin/env python
# coding: utf-8

# # INTRODUCTION
# In this work, i present a way to rating istanbul distincts for guest who liked to use airbnb

# ## my naive rating methodology
# To make a good rating system based on the data i have, i have used price, calculated_host_listings_count,	availability_365,	number_of_reviews and total_houses in that distinct. 
# I calculated the maximum value for each parameter as 1 point and divided each parameter of each distinct to the maximum value. Later then, i sum all the values for each distinct and it gives a a score for each distinct. 
# Enjoy it.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.express as px
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
get_ipython().run_line_magic('matplotlib', 'inline')
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df=pd.read_csv("/kaggle/input/airbnb-istanbul-dataset/AirbnbIstanbul.csv")
df.sample(5)


# ### Vizualization of each airbnber on the map

# In[ ]:


fig = px.scatter_mapbox(df, lat="latitude", lon="longitude", hover_name="neighbourhood", hover_data=["price", "room_type"],
                              color_discrete_sequence=["fuchsia"], zoom=8, height=300)
fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r":0,"t":1,"l":0,"b":0})

fig.show()


# ### Create a dataframe based on mean of used parameters.

# In[ ]:


x=pd.DataFrame(df.groupby(['neighbourhood'])[['price', 'calculated_host_listings_count', 'availability_365', 'number_of_reviews', 'longitude', 'latitude']].mean())
y=pd.DataFrame(df.groupby('neighbourhood')['room_type'].value_counts().unstack().fillna(0))
z=pd.concat([y, x.reindex(y.index)], axis=1)
z['total_houses']= z['Entire home/apt'] + z['Private room'] + z['Shared room']
z.sample(5)


# ### Creating rating system 

# In[ ]:


z['price_score']=z.price/z.price.max()
z['host_score']=z.calculated_host_listings_count/z.calculated_host_listings_count.max()
z['availability_score']=z.availability_365/z.availability_365.max()
z['number_of_reviews_score']=z.number_of_reviews/z.number_of_reviews.max()
z['total_houses_score']=z.total_houses/z.total_houses.max()
z['total_score']=z['total_houses_score'] + z['number_of_reviews_score'] + z['availability_score'] + z['host_score'] + z['price_score']
z.sample(3)


# ### Showing the total score on a map.

# In[ ]:


fig = px.scatter_mapbox(z, lat="latitude", lon="longitude",  hover_name=z.index, hover_data=["total_score"], color="total_score",
                         zoom=8, height=300, size='total_score')
fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r":0,"t":1,"l":0,"b":0})

fig.show()


# ### Let's see scores for calculated host list, number of reviews and availability.

# ### host score

# In[ ]:


fig = px.scatter_mapbox(z, lat="latitude", lon="longitude",  hover_name=z.index, hover_data=["host_score"], color="host_score",
                         zoom=8, height=300, size='host_score')
fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r":0,"t":1,"l":0,"b":0})

fig.show()


# ### Availablity Score

# In[ ]:


fig = px.scatter_mapbox(z, lat="latitude", lon="longitude",  hover_name=z.index, hover_data=["availability_score"], color="availability_score",
                         zoom=8, height=300, size='availability_score')
fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r":0,"t":1,"l":0,"b":0})

fig.show()


# ### Number of Reviews Score

# In[ ]:


fig = px.scatter_mapbox(z, lat="latitude", lon="longitude",  hover_name=z.index, hover_data=["number_of_reviews_score"], color="number_of_reviews_score",
                         zoom=8, height=300, size='number_of_reviews_score')
fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r":0,"t":1,"l":0,"b":0})

fig.show()


# ### Maximum annual income calculation.

# In[ ]:


z['max_annual_income']=round(z['price']*z['availability_365'])
fig = px.scatter_mapbox(z, lat="latitude", lon="longitude",  hover_name=z.index, hover_data=["max_annual_income"], color="max_annual_income",
                         zoom=8, height=300, size='max_annual_income')
fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r":0,"t":1,"l":0,"b":0})

fig.show()


# ### Thank you
