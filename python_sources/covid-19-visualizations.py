#!/usr/bin/env python
# coding: utf-8

# # **COVID-19 VISUALIZATIONS**

# Coronavirus disease or COVID-19 was first identified in December 2019 in Wuhan, China. Since then, everything has almost shutdown to reduce the its transmission.
# 
# I am using visualization techniques to try to see how it sprayed from one country to the rest of the world over time. We will use Choropleth Maps for the all visuals.
# 

# In[ ]:


#Importing Relevant Libraries
import numpy as np 
import pandas as pd 
import plotly as py
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

# Input data files are available in the "../input/" directory.

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


# Read Data
covid_data = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv")


# In[ ]:


#Showing first rows of data
covid_data.head(3)


# In[ ]:


covid_data.info()


# In[ ]:


#let's rename columnw of Country/region and Observations for simplicity

covid_data = covid_data.rename(columns={'Country/Region':'Country'})
covid_data= covid_data.rename(columns={'ObservationDate':'Date'})


# In[ ]:


# Manipulating Dataframe
covid_countries = covid_data.groupby(['Country', 'Date']).sum().reset_index().sort_values('Date', ascending=False)
covid_countries = covid_countries.drop_duplicates(subset = ['Country'])
covid_countries = covid_countries[covid_countries['Confirmed']>0]


# In[ ]:


# Create the Choropleth ....Static
fig = go.Figure(data=go.Choropleth(
    locations = covid_countries['Country'],
    locationmode = 'country names',
    z = covid_countries['Confirmed'],
    colorscale = 'Reds',
    marker_line_color = 'black',
    marker_line_width = 0.5,
))


# In[ ]:


fig.update_layout(
    title_text = 'Confirmed Cases By April 12nd, 2020',
    title_x = 0.5,
    geo=dict(
        showframe = False,
        showcoastlines = False,
        projection_type = 'equirectangular'
    )
)


# As you can see from the above plot, it is static representation of COVID-19 cases by 12 April. We can create animated choropleth map similar to it. we are looking on number of confirmed cases by country over a time.  

# In[ ]:


# Manipulating the original dataframe
covid_countrydate = covid_data[covid_data['Confirmed']>0]
covid_countrydate = covid_countrydate.groupby(['Date','Country']).sum().reset_index()
covid_countrydate.head(3)


# In[ ]:


# Creatinge animated visualization
fig = px.choropleth(covid_countrydate, 
                    locations="Country", 
                    locationmode = "country names",
                    color="Confirmed", 
                    hover_name="Country", 
                    animation_frame="Date"
                   )


# In[ ]:


fig.update_layout(
    title_text = 'Global Spread of Coronavirus',
    title_x = 0.5,
    geo=dict(
        showframe = False,
        showcoastlines = False,
    ))
    
fig.show()


# Now that we can see easily number of cases by region easily(Just move cursor around the regions of interests), let's do the same for the number of deaths

# In[ ]:


#Plotting Deaths by date

# Manipulating the original dataframe, deaths>=0 to show all countries with Zeros cases
covid_deathsdate = covid_data[covid_data['Deaths']>=0]
covid_deathsdate = covid_deathsdate.groupby(['Date','Country']).sum().reset_index()
#covid_countrydate
covid_deathsdate.head(3)


# In[ ]:


fig_deaths = px.choropleth(covid_deathsdate, 
                    locations="Country", 
                    locationmode = "country names",
                    color="Deaths", 
                    hover_name="Country", 
                    animation_frame="Date"
                   )


# In[ ]:


fig_deaths.update_layout(
    title_text = 'Number of Deaths As Of April 12nd',
    title_x = 0.5,
    geo=dict(
        showframe = False,
        showcoastlines = False,
    ))
    
fig_deaths.show()


# > You can now see how cases and deaths increased from one country to the rest of the world. I was inspired by @TerenceShin to do these visuals. 

# In[ ]:




