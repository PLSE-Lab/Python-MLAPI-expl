#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import folium


# In[ ]:


import pandas as pd


# In[ ]:


country_geo  = '../input/coordinatescountry/countries-coordinates.json'


# In[ ]:


data = pd.read_csv('../input/world-development-indicators/Indicators.csv')


# In[ ]:


data.head()


# In[ ]:


hist_indicator = 'CO2 emissions \(metric'


# In[ ]:


hist_year = 2011


# In[ ]:


mask1 = data['IndicatorName'].str.contains(hist_indicator)
mask2 = data['Year'].isin([hist_year])
stage = data[mask1 & mask2]
stage.head()


# In[ ]:


plot_data = stage[['CountryCode','Value']]
plot_data.head()


# In[ ]:


hist_indicator = stage.iloc[0]['IndicatorName']


# here i am trying to visualize the map and using the choropleth maps to bind Pandas Data Frames and json geometries.
# 
# ### *but it is throwing some exception that i am not able to resolve*

# In[ ]:


map = folium.Map(location=[100,0],zoom_start = 1.5)


# In[ ]:


get_ipython().run_line_magic('pinfo', 'map.choropleth')


# In[ ]:


map.choropleth(country_geo, data=plot_data,
             columns=['CountryCode', 'Value'],
             key_on='features.id',
             fill_color='YlGnBu', fill_opacity=0.7, line_opacity=0.2,
             legend_name=hist_indicator)


# In[ ]:





# In[ ]:





# In[ ]:




