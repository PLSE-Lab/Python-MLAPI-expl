#!/usr/bin/env python
# coding: utf-8

# # Exploring the geographic distribution of accidents
# 
# ### This is my initial kernel exploring the BigQuery US Traffic Fatality data. A work in progress I will be adding to, suggestions welcomed!
# 
# In the next few days I will be adding to the data visualizations and attempting some statistical analyses of factors correlated with drunk driving fatalities. I thought I would make this public as suggestions arewelcome!
# 
# ## Imports

# In[ ]:


import bq_helper

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap


# ## Big query explore

# In[ ]:


traffic_data = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")


# In[ ]:


traffic_data.list_tables()


# In[ ]:


traffic_data.head('drimpair_2016')


# In[ ]:


traffic_data.head('distract_2016')


# In[ ]:


traffic_data.head('vehicle_2016')


# In[ ]:


list(traffic_data.head('vehicle_2016').columns)


# In[ ]:


traffic_data.head('vehicle_2016')[['body_type_name',
                                    'vehicle_make_name',
                                    'vehicle_model_year',
                                    'travel_speed',
                                    'previous_dwi_convictions',
                                    'previous_speeding_convictions',
                                    'previous_other_moving_violation_convictions',
                                    'driver_drinking',]]


# In[ ]:


traffic_data.head('accident_2016')


# In[ ]:


list(traffic_data.head('accident_2016').columns)


# In[ ]:


traffic_data.head('accident_2016')[['number_of_drunk_drivers', 
                                    'related_factors_crash_level_1_name',
                                    'related_factors_crash_level_2_name',
                                    'school_bus_related',
                                    'latitude',
                                    'longitude',
                                    'state_name',]]


# ## SQL inner join to extract the data of interest

# In[ ]:


join_accidents_and_vehicles = """
SELECT a.consecutive_number AS a_consecutive_number, a.latitude, a.longitude, 
        a.number_of_drunk_drivers, a.city, a.timestamp_of_crash,
        a.number_of_fatalities, a.related_factors_crash_level_1_name,
        a.related_factors_crash_level_2_name, a.related_factors_crash_level_3_name,
        v.consecutive_number AS v_consecutive_number,
        v.vehicle_make_name, v.vehicle_model_year, v.travel_speed,
        v.previous_dwi_convictions, v.previous_speeding_convictions,
        v.previous_other_moving_violation_convictions, v.driver_drinking,
        v.body_type_name
FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016` a INNER JOIN `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016` v
    ON a.consecutive_number = v.consecutive_number
"""


# In[ ]:


traffic_data.estimate_query_size(join_accidents_and_vehicles)


# In[ ]:


a_v_2016 = traffic_data.query_to_pandas_safe(join_accidents_and_vehicles)


# To save for later use

# ## Save the SQL data to a .csv
# just in case I run over my quota, I can still play with this subset later!

# In[ ]:


a_v_2016.to_csv('vehicle_and_accident_data_2016.csv')


# In[ ]:


a_v_2016.head()


# In[ ]:


a_v_2016.columns


# ## Exploring the latitude and longitude data + cleanup

# In[ ]:



a_v_2016.plot(kind='scatter', x='longitude', y='latitude')
plt.ylabel("Latitude", fontsize=14)
plt.xlabel("Longitude", fontsize=14)
plt.legend() 
plt.show()


# 
# There are some erroneous values in the data, as the plots fall far outside the lat and long of the united states! This should remove them

# In[ ]:


a_v_2016 = a_v_2016.drop(a_v_2016[ a_v_2016['longitude'] > 0].index)


# Plotting this, we still ate a little squished on account of alaska and hawaii, I'm going to remove these in order to focus on the continentual united states.

# In[ ]:


a_v_2016.plot(kind='scatter', x='longitude', y='latitude',
			 alpha=0.4,figsize=(10,7), c='black')
plt.ylabel("Latitude", fontsize=14)
plt.xlabel("Longitude", fontsize=14)
plt.legend() 
plt.show()

a_v_2016 = a_v_2016.drop(a_v_2016[ a_v_2016['longitude'] < -130].index)


# Now that we have cleaned the data up a bit, lets take a look at the locations of the fatal car accidents in the dataset
# 
# ## Fatal car accidents in the United States

# In[ ]:


#distribution of fatal car accidents in the continential united states
#show in a mercator projection
m = Basemap(projection='merc',llcrnrlat=20,urcrnrlat=50,            llcrnrlon=-130,urcrnrlon=-60,lat_ts=20,resolution='i')
m.drawcoastlines()
m.drawcountries()
#m.drawstates()
# draw parallels and meridians.
parallels = np.arange(-90., 91., 5.)
# Label the meridians and parallels
m.drawparallels(parallels, labels=[False,True,True,False])
# Draw Meridians and Labels
meridians = np.arange(-180., 181., 10.)
m.drawmeridians(meridians, labels=[True, False, False, True])
m.drawmapboundary(fill_color = 'white')
plt.title('Fatal car accidents in the continential United States, 2016')
x,y = m(a_v_2016['longitude'].values, a_v_2016['latitude'].values) #transform to projection
m.plot(x,y, 'bo', markersize = 0.5)
plt.show()


# The number of accidents is astonishingly large, and seems to roughly follow a population distribution. Lets isolate just the drunk driving incidents and see where they occurred.
# 
# ## Drunk driving distribution

# In[ ]:


#isolate just the drunk driving incidents

drunk_driving = a_v_2016[a_v_2016['driver_drinking'] == 'Drinking']


# In[ ]:


m = Basemap(projection='merc',llcrnrlat=20,urcrnrlat=50,            llcrnrlon=-130,urcrnrlon=-60,lat_ts=20,resolution='i')
m.drawcoastlines()
m.drawcountries()
#m.drawstates()
# draw parallels and meridians.
parallels = np.arange(-90., 91., 5.)
# Label the meridians and parallels
m.drawparallels(parallels, labels=[False,True,True,False])
# Draw Meridians and Labels
meridians = np.arange(-180., 181., 10.)
m.drawmeridians(meridians, labels=[True, False, False, True])
m.drawmapboundary(fill_color = 'white')
plt.title('Fatal car accidents Involving a drunk driver, 2016')
x,y = m(drunk_driving['longitude'].values, drunk_driving['latitude'].values) #transform to projection
m.plot(x,y, 'bo', markersize = 0.5)
plt.show()


# Again, the concentration appears to mirror the location of major cities in the United states.
# 

# ## What kind of cars are involved in the accidents?

# In[ ]:


a_v_2016.vehicle_make_name.unique()


# The category breakdown there is a little ridiculous
# Note this option in the categories:
# 
# `'Other Domestic\nAvanti\nChecker\nDeSoto\nExcalibur\nHudson\nPackard\nPanoz\nSaleen\nStudebaker\nStutz\nTesla (Since 2014)'`
# 
# Hudson ceased to exist in 1954, Packard died in 1956 and studebaker went defunct in 1967... yet these are grouped with Teslas?  The category is sort of a catch all for 'weird cars from america' 
# 

# In[ ]:


a_v_2016.body_type_name.unique()


# The descriptions of the body type are extremly detailed in some instances!
