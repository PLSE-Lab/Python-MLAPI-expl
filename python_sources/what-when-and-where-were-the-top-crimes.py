#!/usr/bin/env python
# coding: utf-8

# <h1>High Level Overview</h1>
# 
# This is a high level overview that will look at:
# 
# * Which crimes were the most common
# * When and where did the most common crimes occur
# 
# 

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap

data = pd.read_csv("../input/crime.csv")
# For this study only look at a few key columns ands remove rows with no data
df = data [['OFFENSE_ID','OFFENSE_TYPE_ID','OFFENSE_CATEGORY_ID','REPORTED_DATE','GEO_LON','GEO_LAT']]
df = df.dropna()


# <H2>What were the Most Common Crimes ?</H2>
# 
# As the next step let's look at the 20 most common crimes.

# In[ ]:


# Get a count of the top 20 crimes, based on the "OFFENSE_TYPE_ID"
crime_cnts = df[['OFFENSE_TYPE_ID','OFFENSE_ID']].groupby(['OFFENSE_TYPE_ID'],as_index=False).count().nlargest(20,['OFFENSE_ID'])

# Plot the most common crimes
ax = crime_cnts.plot(kind='bar', x='OFFENSE_TYPE_ID', title ="Overall Counts of the Type of Crime", figsize=(15, 8), fontsize=12,legend=False)
ax.set_xlabel("Types of Crime", fontsize=12)
ax.set_ylabel("Total Counts", fontsize=12)
plt.show()


# <H2>Traffic Related Crimes were the Most Common</H2>
# 
# The three top crimes were all traffic related so let's group them together and look at when they occurred.

# In[ ]:


import warnings
warnings.filterwarnings('ignore')

# Group all traffic crimes into a data set
traffic = df[df['OFFENSE_TYPE_ID'].str[:4] == 'traf']

# Create columns for month and hour
traffic['MONTH'] = pd.DatetimeIndex(traffic['REPORTED_DATE']).month
traffic['HOUR'] = pd.DatetimeIndex(traffic['REPORTED_DATE']).hour

# Which months are getting the most traffic crimes
traf_month = traffic[['OFFENSE_TYPE_ID','MONTH']].groupby(['MONTH'],as_index=False).count()
ax = traf_month.plot(kind='bar', x='MONTH', title ="Which Month is Getting the Most Traffic Crimes", figsize=(15, 8), fontsize=12,legend=False,)
ax.set_xlabel("Month", fontsize=12)
ax.set_ylabel("Crime/Month", fontsize=12)
plt.show()

# Which hours of the day are getting the most traffic crimes
traf_hour = traffic[['OFFENSE_TYPE_ID','HOUR']].groupby(['HOUR'],as_index=False).count()
ax = traf_hour.plot(kind='bar', x='HOUR', title ="Which Hour of the Day is Getting the Most Traffic Crimes", figsize=(15, 8), fontsize=12,legend=False,)
ax.set_xlabel("Hour", fontsize=12)
ax.set_ylabel("Crime/Hour", fontsize=12)
plt.show()


# <h2>The Time of the Year Isn't A Big Factor</h2>
# 
# THe months of January to May seem to be slightly higher for traffic crimes, but not signifigently.
# 
# <h2>Driving Home from Work is the Worst for Accidents</h2>
# 
# It appears that the hours of 3-6pm are the worst for traffic crimes.
# 
# As the next steps let's look at geographically where the traffic crimes are occuring between 3-6pm.

# In[ ]:


# Add mapping libraries and traffic summaries on a geographic map
import folium
from folium.plugins import HeatMap

map_den = folium.Map(location= [39.76,-105.02], zoom_start = 16)

# Get data from 15:00 to 18:00
den15_18 = traffic[(traffic['HOUR'] >= 15) & (traffic['HOUR'] <= 18)]

# Create a list with lat and long values and add the list to a heat map, then show map
heat_data = [[row['GEO_LAT'],row['GEO_LON']] for index, row in den15_18.iterrows()]
HeatMap(heat_data).add_to(map_den)

map_den


# <H1>Summary</H1>
# 
# The goal of this study was to look at the top crimes and when they occured. Not surprisingly traffic crimes were signifigently higher than any other crimes, and most of these crimes occured on the drive home from work (3-6pm). Traffic crimes were slightly higher in the winter months.
# 
# For this study a geographic heat plot was used to show where the traffic crimes were most common during the evening rush hour. The heat plot can be extremely overwhelming if too much data is used or the map is zoom out too far.
# 
# 
