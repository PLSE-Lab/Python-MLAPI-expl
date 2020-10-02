#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from google.cloud import bigquery
import bq_helper 
import numpy as np
import pandas as pd


# In[ ]:


# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")


# In[ ]:


list(accidents.head('accident_2016').columns)


# In[ ]:


query = """SELECT state_name,
            COUNT(consecutive_number) AS total
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
            GROUP BY 1
            ORDER BY COUNT(consecutive_number) DESC
        """
accidents_by_states = accidents.query_to_pandas_safe(query)


# In[ ]:


accidents_by_states['state_name'].unique()


# ## Top 10 States by Traffic fatalities by 2016 in USA

# In[ ]:


accidents_by_states.head(10)


# In[ ]:


import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
number = accidents_by_states.total.head(10)
name = accidents_by_states.state_name.head(10)


trace0 = go.Bar(x=number, y=name, orientation = 'h')
data = [trace0]

layout = go.Layout(title='Top 10 States Total Traffic Fatalities in 2016')

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='Fatalities_2016')


# In[ ]:


get_ipython().system('pip install folium')


# ## Mapping states fatalities 

# In[ ]:


query = """SELECT DISTINCT(state_name),
             latitude, longitude
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
                     
        """
location = accidents.query_to_pandas_safe(query)


# In[ ]:


states_cordinates = location.groupby(['state_name'])['latitude', 'longitude'].mean()


# In[ ]:


states_cordinates = states_cordinates.reset_index()


# In[ ]:


states_cordinates.columns


# In[ ]:


accidents_by_states.sort_values('state_name', inplace=True)


# In[ ]:


accidents_by_states


# In[ ]:


sates_df = states_cordinates.merge(accidents_by_states,on='state_name', how='inner')


# In[ ]:


sates_df


# In[ ]:


import folium
from folium import plugins
map_obj = folium.Map(location=[42.50, -99.45], tiles='cartodbpositron', zoom_start=3.5)
for j, rown in sates_df.iterrows():
    rown = list(rown)
    folium.CircleMarker([float(rown[1]), float(rown[2])], popup="<b>State:</b>" + rown[0].title() +"<br> <b>Fatalites:</b> "+str(int(rown[3])), radius=float(rown[3])*0.001, color='#be0eef', fill=True).add_to(map_obj)
map_obj

