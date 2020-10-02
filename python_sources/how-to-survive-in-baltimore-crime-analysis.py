#!/usr/bin/env python
# coding: utf-8

# ![Baltimore](https://storage.needpix.com/rsynced_images/urban-534019_1280.jpg)
# # Baltimore crime analysis.
# 
# Under impression from "The Wire" TV Show. But it would be interesting to see real crime image of Baltimore, especially violent crimes.
# So, we have data about crimes from 01/01/2012 to 09/02/2017. Yes, this is another time period than "The Wire" action, but we'll have imagination about the level of crime probably in most dangerous city of the United States.
# Let's take a look at data and discover answers to questions.
# 
# Thanks [Sohier Dane](https://www.kaggle.com/sohier) for the [data](https://www.kaggle.com/sohier/crime-in-baltimore).

# # Questions
# 
# 1. Median number of crimes in Baltimore
# 2. What day was most filled with crimes? Why?
# 3. What day was with minimum number of crimes? Why?
# 4. What crimes are most often in Baltimore?
# 5. What day of the week are most of crimes?
# 6. How to survive in Baltimore? Buiding the heatmap.

# In[ ]:


# Importing the libraries

import pandas as pd
import datetime

# for visualizations

import folium
from folium import plugins
from folium.plugins import HeatMap
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})


# In[ ]:


# Importing the dataset. Checking the names of columns.

df = pd.read_csv('/kaggle/input/crime-in-baltimore/BPD_Part_1_Victim_Based_Crime_Data.csv')
df.info()


# In[ ]:


# Renaming long and uninformative "Description" to laconical "Crime"

df.rename(columns={'Description':'Crime'}, inplace=True)
df['Crime'] = df['Crime'].str.capitalize()


# In[ ]:


# How many accidents we have

df.shape


# # 1. Median number of crimes in Baltimore

# In[ ]:


# To return median value per day

df['CrimeDate'].value_counts().median()


# # 2. What day was minimum number of crimes? Why?

# Discovering what day of min number of crimes.

# In[ ]:


# Take a dates with a minimum of crimes

df['CrimeDate'].value_counts().tail(3)


# * Jan. 23, 2016 - the number of crimes is 39.4% lower than the 2nd least crime day.
# 
# Why?
# THis date is the second day of the biggest showstorm in the history of Baltimore. [1]

# # 3. What day was most filled with the crimes? Why?

# Discovering what day of max number of crimes.

# In[ ]:


# Take a dates with a maximum of crimes

df['CrimeDate'].value_counts().head(3)


# * Apr. 27, 2015 - the number of crimes is 61.3% higher than the 2nd top crime day and 1169.7% hihger than most quiet day.
# 
# What are the reasons? In reallity it's easy to find the answers. Freddy Gray funeral and the following riot took place this day. [2]
# 
# What crimes happened at the day of Freddy Gray funeral?

# In[ ]:


# Analysing the day of Freddy Gray funeral

df_fgray = df['CrimeDate'] == '04/27/2015'
df.loc[df_fgray]['Crime'].value_counts().plot.bar(figsize=(11, 6))


# Too much in compartion with median value.

# In[ ]:


# Median value again

df['CrimeDate'].value_counts().median()


# # 4. What crimes are most often in Baltimore?

# In[ ]:


# Building a bar plot with full stats of Baltimore crimes

crime_num = df['Crime'].value_counts().plot.bar(figsize=(11, 6))


# In[ ]:


# How many crimes and percentage are in called time period.

crime_num = df['Crime'].value_counts()
crime_pct = df['Crime'].value_counts(1) * 100
pd.DataFrame({'Crimes': crime_num, 'Percent' : crime_pct}).round(2)


# In[ ]:


# Return pie plot with cuts of crime

plt.rcParams.update({'font.size': 12})
crime_num = df['Crime'].value_counts().head(15).plot.pie(radius=4, autopct='%1.1f%%', textprops=dict(color="black"))


# # 5. What day of the week are most of crimes?

# In[ ]:


# Taking back the font for the bar plot

plt.rcParams.update({'font.size': 18})

# Date format correction

df['CrimeDate'] = pd.to_datetime(df['CrimeDate'], format='%m/%d/%Y')

# Define day of week and column creation

df['DayOfWeek'] = df['CrimeDate'].dt.day_name()

# What part of the week is most dangerous?

df['DayOfWeek'].value_counts().plot.bar(figsize=(11, 6))


# # 6. How to survive in Baltimore?

# To find the answer for this better to calculate factors which making insafety

# In[ ]:


# Creating dataframe with homicides and returning on the bar plot to find out which neighborhoods are most dangerous by statistics of murders

hc = df['Crime'] == 'Homicide'
df.loc[hc]['Neighborhood'].value_counts().head(35).plot.bar(figsize=(26, 6))


# In[ ]:


# Creating dataframe with shootings and returning on the bar plot to find out which neighborhoods are most dangerous by statistics of shooting

sh = df['Crime'] == 'Shooting'
df.loc[sh]['Neighborhood'].value_counts().head(35).plot.bar(figsize=(26, 6))


# In[ ]:


# Creating dataframe with aggravated assaults and returning in the bar plot to find out which neighborhoods are most dangerous by statistics of aggravated assaults

ass = df['Crime'] == 'Agg. assault'
df.loc[ass]['Neighborhood'].value_counts().head(35).plot.bar(figsize=(26, 6))


# In[ ]:


hc = df['Crime'] == 'Homicide'
df.loc[hc]['CrimeTime'].value_counts()


# In[ ]:


hc = df['Crime'] == 'Shooting'
df.loc[hc]['CrimeTime'].value_counts().head(8)


# In[ ]:


sh = df['Crime'] == 'Homicide'
df.loc[sh]['DayOfWeek'].value_counts().head(7)


# In[ ]:


sh = df['Crime'] == 'Agg. assault'
df.loc[sh]['DayOfWeek'].value_counts().sort_index().plot.bar(figsize=(12, 6))


# The best way to survive in Baltimore is to keep away from Sandtown-Winchester, Coldstream Homestead Montebello and Brooklin from 9 PM to 2 AM at Sunday, Monday and Tuesday.

# # Heatmap with crimes

# In[ ]:


# Dropping crimes with NaN numbers of neccessaried data

df = df[pd.notnull(df['Latitude'])]


# In[ ]:


# Dropping crimes with NaN numbers of neccessaried data

df = df[pd.notnull(df['Longitude'])]


# In[ ]:


# Taking a look how many  accidents left. Enough.

df.shape


# In[ ]:


# Limitation of the area which we need

m = folium.Map(location=[39.3121, -76.6198], zoom_start=13)
m


# In[ ]:


# for index, row in df.iterrows():
#    folium.CircleMarker([row['Latitude'], row['Longitude']],
#                        radius=15,
#                        popup=row['Crime'],
#                        fill_color="#3db7e4", # divvy color
#                       ).add_to(m)


# To understend better where better to avoid - heatmaps:

# homicide

# In[ ]:


# Ensure you're handing it floats
df['Latitude'] = df['Latitude'].astype(float)
df['Longitude'] = df['Longitude'].astype(float)

# Filter the DF for rows, then columns, then remove NaNs
# heat_df = df[df['CrimeDate']=='2015-04-27'] # Reducing data size so it runs faster
heat_df = df[df['Crime']=='Homicide'] # Reducing data size so it runs faster
# heat_df = df[['Latitude', 'Longitude']]
# heat_df = heat_df.dropna(axis=0, subset=['Latitude','Longitude'])

# List comprehension to make out list of lists
heat_data = [[row['Latitude'],row['Longitude']] for index, row in heat_df.iterrows()]

# Plot it on the map
HeatMap(heat_data).add_to(m)

# Display the map
m


# shooting

# In[ ]:


# Ensure you're handing it floats
df['Latitude'] = df['Latitude'].astype(float)
df['Longitude'] = df['Longitude'].astype(float)

# Filter the DF for rows, then columns, then remove NaNs
# heat_df = df[df['CrimeDate']=='2015-04-27'] # Reducing data size so it runs faster
heat_df = df[df['Crime']=='Shooting'] # Reducing data size so it runs faster
# heat_df = df[['Latitude', 'Longitude']]
# heat_df = heat_df.dropna(axis=0, subset=['Latitude','Longitude'])

# List comprehension to make out list of lists
heat_data = [[row['Latitude'],row['Longitude']] for index, row in heat_df.iterrows()]

# Plot it on the map
HeatMap(heat_data).add_to(m)

# Display the map
m


# Arson

# In[ ]:


# Ensure you're handing it floats
df['Latitude'] = df['Latitude'].astype(float)
df['Longitude'] = df['Longitude'].astype(float)

# Filter the DF for rows, then columns, then remove NaNs
# heat_df = df[df['CrimeDate']=='2015-04-27'] # Reducing data size so it runs faster
heat_df = df[df['Crime']=='Arson'] # Reducing data size so it runs faster
# heat_df = df[['Latitude', 'Longitude']]
# heat_df = heat_df.dropna(axis=0, subset=['Latitude','Longitude'])

# List comprehension to make out list of lists
heat_data = [[row['Latitude'],row['Longitude']] for index, row in heat_df.iterrows()]

# Plot it on the map
HeatMap(heat_data).add_to(m)

# Display the map
m


# Interesting there are almost equal heat zones.

# In[ ]:


# df_cr = folium.map.FeatureGroup()

# df_cr = plugins.MarkerCluster().add_to(m)

# loop through the crimes and add each to the incidents feature group
# for lat, lng, label in zip(df.Latitude, df.Longitude, df.Crime):
  #  df_cr.add_child(
   #     folium.CircleMarker(
    #        [lat, lng],
     #       radius=5, # define how big you want the circle markers to be
      #      color='yellow',
       #     fill=True,
        #    popup=label,
         #   fill_color='blue',
          #  fill_opacity=0.6
        #)
    #)

# add incidents to map
# m.add_child(df_cr)
# m


# In[ ]:


df.drop(df[df.CrimeDate > '08-31-2017'].index, inplace=True)


# In[ ]:


df.groupby(
  pd.Grouper(
    key='CrimeDate',
    freq='M'
  )
).size().plot.line(figsize=(24, 8), linewidth=3.5)


# In[ ]:


df.groupby(
  pd.Grouper(
    key='CrimeDate',
    freq='M'
  )
).size().plot.bar(figsize=(25, 5))


# February is the month when bad guys would like to stay home.

# Please add your plots - it's a hundreds of ways how to present this data.

# # Links
# [1] https://www.weather.gov/lwx/winter_storm-pr
# 
# [2] https://en.wikipedia.org/wiki/2015_Baltimore_protests
# 
