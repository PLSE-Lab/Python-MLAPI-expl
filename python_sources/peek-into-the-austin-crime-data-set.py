#!/usr/bin/env python
# coding: utf-8

# # Austin Crime dataset
# This is an example kernel to get you started with this dataset that contains crime reports from the Austin, TX police department from 2002 to 2019.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


df = pd.read_csv('../input/rows.csv')


# In[ ]:


df.dtypes


# In[ ]:


# check out some samples
df.head()


# In[ ]:


# how many samples are in this data set
len(df)


# In[ ]:


# convert the columns with date information into a datetime format
from datetime import datetime
df['report_dt'] = pd.to_datetime(df['Report Date Time'],format='%m/%d/%Y %I:%M:%S %p')
df['occured_dt'] = pd.to_datetime(df['Occurred Date Time'],format='%m/%d/%Y %I:%M:%S %p')
# and drop the columns that we don't need anymore
df.drop(['Report Date Time','Occurred Date Time','Report Date'
         ,'Report Time','Occurred Date','Occurred Time'],axis=1,inplace=True)


# There are a few rows with empty dates. You might want to look into these and correct the problems. For this excercise I'm simply deleting these rows.

# In[ ]:


df['report_dt'].isnull().sum()


# In[ ]:


df['occured_dt'].isnull().sum()


# In[ ]:


df = df[df['report_dt'].isnull() == False]
df = df[df['occured_dt'].isnull() == False]


# ### Let's see some graphs

# In[ ]:


crimes_per_year = df['report_dt'].dt.year.value_counts().sort_index()

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
sns.set_color_codes("pastel")

g = sns.barplot(x=crimes_per_year.index, y=crimes_per_year.values,color='b')
g.set_xticklabels(g.get_xticklabels(),rotation=90)
g.set(xlabel='Year', ylabel='# crimes reported')
plt.title('Number of reported crimes per year')
plt.show()


# there's only one row in the dataset for 2002. Looks like there was a peak in crimes reported in 2008 but since then there's a steady decline. Good to know.

# In[ ]:


crimes_per_tod = df['occured_dt'].dt.hour.value_counts().sort_index()
g = sns.barplot(x=crimes_per_tod.index, y=crimes_per_tod.values, color='b')
g.set(xlabel='Hour', ylabel='# crimes reported')
plt.title('Crime reports per hour of day')
plt.show()


# Interestingly the peak time for crimes is during noon. Expected is the increasing crime rate from evening to night and from 3am things are quieting down.

# In[ ]:


top_crimes = df['Highest Offense Description'].value_counts().head(25)
sns.set(rc={'figure.figsize':(12,8)},style="whitegrid")
g = sns.barplot(y=top_crimes.index, x=top_crimes.values,color='b')
g.set(xlabel='# crimes reported', ylabel='Offense description')
plt.title('Top 25 offenses (2003-2019)')
plt.show()


# Top offense in Austin over all years is stealing cars (wow, over 200k in 15 years)
# 
# ### Visualising crime locations
# I'll be using the Lat/Long columns for these graphs. If you're going to use these for further analysis make sure you check the validity of the data. There seem to be some samples with invalid geo-coordinates. 
# 
# Let's check out where bicycles were stolen in 2018

# In[ ]:


crime_coords = df[(df['Latitude'].isnull() == False) 
                  & (df['Longitude'].isnull() == False)
                 & (df['Highest Offense Description'] == 'THEFT OF BICYCLE')
                 & (df['report_dt'].dt.year == 2018)][['Latitude','Longitude']]


# In[ ]:


import folium
from folium import plugins

map_1 = folium.Map(location=[30.285516,-97.736753 ],tiles='OpenStreetMap', zoom_start=11)
map_1.add_child(plugins.HeatMap(crime_coords[['Latitude', 'Longitude']].values, radius=15))
map_1


# When you zoom into the map you can see some hotspots for stealing bikes: Downtown area but also the Domain in North Austin.
# 
# Just for fun, do the same for stolen vehicles.

# In[ ]:


crime_coords = df[(df['Latitude'].isnull() == False) 
                  & (df['Longitude'].isnull() == False)
                 & (df['Highest Offense Description'] == 'BURGLARY OF VEHICLE')
                 & (df['report_dt'].dt.year == 2018)][['Latitude','Longitude']]


# In[ ]:


map_2 = folium.Map(location=[30.285516,-97.736753 ],tiles='OpenStreetMap', zoom_start=11)
map_2.add_child(plugins.HeatMap(crime_coords[['Latitude', 'Longitude']].values, radius=15))
map_2


# And for shoplifting

# In[ ]:


crime_coords = df[(df['Latitude'].isnull() == False) 
                  & (df['Longitude'].isnull() == False)
                 & (df['Highest Offense Description'] == 'THEFT BY SHOPLIFTING')
                 & (df['report_dt'].dt.year == 2018)][['Latitude','Longitude']]


# In[ ]:


map_3 = folium.Map(location=[30.285516,-97.736753 ],tiles='OpenStreetMap', zoom_start=11)
map_3.add_child(plugins.HeatMap(crime_coords[['Latitude', 'Longitude']].values, radius=15))
map_3


# In[ ]:




