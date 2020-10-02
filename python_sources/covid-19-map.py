#!/usr/bin/env python
# coding: utf-8

# # Covid-19 maps
# What else to do during quarantine is how not to improve your visualization skills. Fortunately, plotly allows us to make beautiful maps in just a few lines of code. Maps are scaled, you can select areas and, when you hover over the cursor, the information corresponding to each country appears. 
# 

# In[ ]:


import pandas as pd
import plotly.express as px


# In[ ]:


data = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')


# In[ ]:


data.rename(columns = {'Country/Region':'Country', 'ObservationDate':'Date'},inplace = True)
data = data.groupby(['Date','Country']).sum().reset_index()


# In[ ]:


fig = px.choropleth(data, 
                    locations="Country", 
                    locationmode = "country names",
                    color="Confirmed", 
                    hover_name="Country",
                    hover_data = ['Confirmed','Deaths','Recovered'],
                    animation_frame="Date",
                    color_continuous_scale=px.colors.sequential.Inferno,
                    template="plotly_dark")

fig.update_layout(
    title_text = 'Covid-19 timelapse',
    title_x = 0.5,
    geo=dict(
        showframe = False,
        showcoastlines = False,
    ))


fig.show()


# In[ ]:


fig = px.scatter_geo(data, locations="Country", 
                     locationmode = "country names", color="Confirmed", 
                     hover_name="Country", hover_data = ['Deaths', 'Confirmed'], size="Confirmed",
                     animation_frame="Date",
                     projection="natural earth",
                     color_discrete_sequence=["green", "goldenrod", "red"], size_max=80,
                     color_continuous_scale=px.colors.sequential.thermal,
                     template="plotly")

fig.update_layout(
    title_text = 'Coronavirus scatter timeline',
    title_x = 0.5)


fig.show()


# I will do a detailed analysis of the coronavirus and publish it later. Thanks for watching!
