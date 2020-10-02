#!/usr/bin/env python
# coding: utf-8

# In this project, I analyze data from earthquakes took place in Greece in a time span of 118 years
# 
# All the data are from this Kaggle [dataset](https://www.kaggle.com/astefopoulos/earthquakes-in-greece-19012018)
# 

# In[ ]:


# IMPORTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()    
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# LOAD THE DATA
data = pd.read_csv('../input/EarthQuakes in Greece.csv')


# In[ ]:


data.info()


# No missing values and seems that all data types are as suppose to be: numerical values

# What is very interesting is that we have three columns to show the date each earthquake took place. One for the year, one for the month and one for the day. A good idea is to make one column out of these three columns.

# In[ ]:


data.describe()


# We have a time span from 1901 to 2018 this is about 118 year of earthquakes, that will be interesting. The month entries are ok (1 - 12) as the day entries (1 - 31), hour and minutes seem to be ok too. The deviation from the LAT and LON seem to be very small, that means that the earthquakes took place around the same region, but that can we prove it later. The magnitude of the earthquakes has a min of 0, that seems not ok as an earthquake of 0 doesn't exist. But the max is 8 Richter, that is quite a shake.

# A quick look online I found that only one earthquake took place in Greece and was so big - 8 Richter and was around 11 August 1903 near the Island of Kythira, that can we proof now.

# In[ ]:


data[data['MAGNITUDE (Richter)'] == 8]


# Great we proof that we have only one so big earthquake and seem to be right, so our dataset is valid.

# In[ ]:


data.head()


# This is the head of our data, at first look what can we fix is the names, the long and all caps names can be changed with simpler and easier to type names.

# In[ ]:


columns = ['year', 'month', 'day', 'hours', 'minutes', 'LAT', 'LON', 'richter']

data.columns = columns

data['date'] = pd.to_datetime(data[['year','month','day','hours','minutes']])

data.drop(['year', 'month', 'day', 'hours', 'minutes'], axis=1, inplace=True)


# In[ ]:


data.head()


# Very good, our dataset is all tied up, we have a nice column with the complete date, a simple column for the magnitude and for the GEO locations we have simpler names, we can type quicker and with no mistakes.

# Before we go any further let's see what each Richter value means:
# <img src="https://upload.wikimedia.org/wikipedia/commons/d/de/Earthquake_severity.jpg"/>
# 
# source:https://en.wikipedia.org/wiki/Richter_magnitude_scale

# In[ ]:


# PLOTS & EDA


# Now, let's make some data analysis, we can dive deeper into our data to see what is really going on.
# 
# We have 118 years of records but, do we have an equal amount of data for each year?

# In[ ]:


data['date'].hist(bins=30)
plt.yscale('log')
plt.xlabel('Year')
plt.tight_layout()


# It seems no, as we can see we have more records late at 90's than early at 90's. That is normal I think as at these years there were no right foundations ot the needed equipment for this job.
# 
# But this will be no problem for our analysis.

# Now lets see how the magnitude is distibuted

# In[ ]:


data['richter'].hist(bins=8)
plt.yscale('log')
plt.xlabel('Magnitude (Richter)')
plt.tight_layout()


# In[ ]:


sns.boxplot(x='richter', data=data, )


# Nice, the magnitude feature is well distributed, we have a mid-range of earthquakes with max around 8 as we saw earlier a min of 0. Now, this 0 Richter earthquake we have to invest further.

# In[ ]:


data[data['richter'] == 0]


# Ok, its only two entries, we can live with that, or we can drop them. That's up to the scope of the analysis. At the moment I will let them inside. The important thing is to know that there are exist.Ok, its only two entries, we can live with that, or we can drop them. That's up to the scope of the analysis. At the moment I will them inside. The important think is to know that there are there.

# Now let's try to look at how strong the earthquakes are in the pass of years. We do this with a heatmap but first, we have to create a pivot table

# In[ ]:


year = data['date'].apply(lambda x: x.year) #we separate the year from the date
month = data['date'].apply(lambda x: x.month) #we separate the month from the date
hour = data['date'].apply(lambda x: x.hour) #we separate the hours from the date

pivot_year = pd.pivot_table(data, values='richter', index=year)#we group the data for each year in a pivot table
pivot_month = pd.pivot_table(data, values='richter', index=month)#we group the data for each month in a pivot table
pivot_hour = pd.pivot_table(data, values='richter', index=hour)#we group the data for each hour in a pivot table


# In[ ]:


f, axes = plt.subplots(3, 2, figsize=(10, 10))
sns.heatmap(pivot_year,yticklabels='auto', cmap='viridis', ax=axes[0][0])
sns.heatmap(pivot_month,yticklabels='auto', cmap='viridis', ax=axes[1][0])
sns.heatmap(pivot_hour,yticklabels='auto', cmap='viridis', ax=axes[2][0])

pivot_year.plot(ax=axes[0][1])
pivot_month.plot(ax=axes[1][1])
pivot_hour.plot(ax=axes[2][1])


plt.tight_layout()


# Very nice, we see that as years go by we have weaker earthquakes, BUT this is not the case now, we noticed earlier that we have a smaller amount of records for the early '90s perhaps the people back then recorded only the strongest earthquakes and that's why we see this strange effect.
# 
# As consider the months the strongest seem to take place around January and February while the weakest around September, otherwise the rest months are equally distributed so no pattern here.
# 
# The hours are kind of interesting as we can see, late in the night we have the weakest earthquakes and about between 6 - 10 am the strongest.

# Now its time to see the GEO features
# 
# Sometimes making a scatter plot of the features we can see the shape of a location, county, city, etc. Let's try it now to see what happens

# In[ ]:


sns.scatterplot(x='LAT', y='LON', data=data)


# With little further investigation online we find from this [source](http://geodata.gov.gr/en/dataset/periphereies-elladas) the physical boundaries of Greece. Now we can draw these border lines in the plot to have a picture of the earthquakes took place only in Greece 
# 
# South Bound Latitude: 33.957559
# 
# North Bound Latitude: 44.108926
# 
# West Bound Longitude: 18.17496
# 
# East Bound Longitude: 32.061679

# In[ ]:


sns.scatterplot(x='LAT', y='LON', data=data)
plt.axvline(33.957559, color='r')
plt.axvline(44.108926, color='r')
plt.axhline(18.17496, color='r')
plt.axhline(32.061679,color='r')
plt.tight_layout()


# 
# We see that some of the data are not located in Greece but there are only a few compares to the whole dataset.

# Now, we will use Folium an open source library which helps us plot maps

# In[ ]:


import folium


# In[ ]:


m = folium.Map([data['LAT'].mean(), data['LON'].mean()],zoom_start=6)
m


# Ok, now we see the map of Greece and more specifically the mean area where the earthquakes took place. Now if we try to plot each entry we will end up to a high computational process and it would be unnecessary, so let's plot only the earthquakes that are bigger or equal 7 Richters. If we make a search in Wikipedia we will see that an earthquake with a magnitude bigger than 6 Richters described as "strong". So we decide to plot only the "strong" quakes.

# In[ ]:


def make_map(clm):
    lat = clm['LAT']
    lon = clm['LON']
    mag = clm['richter']
    year = clm['date'].year


    folium.Circle(
        radius=2000 * mag,
        location=[lat, lon],
        popup="Year: " + str(year) +" " + "Magnitude: " + " " +str(mag) + " richter",
        color='crimson',
        fill=False,
    ).add_to(m)


# In[ ]:


filter_richter = data['richter'] >= 7
filtered_data = data[filter_richter]

m = folium.Map([data['LAT'].mean(), data['LON'].mean()], zoom_start=6)
_ = filtered_data.apply(lambda x:make_map(x), axis=1)

m


# In[ ]:




