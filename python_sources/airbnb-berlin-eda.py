#!/usr/bin/env python
# coding: utf-8

# The purpose of this kernel is to show EDA steps and initial concludions drawn from the AirBnB Berlin listings data. 
# It includes exploration and summary of the main data types, dealing with missing values as well as some look into the price variable and its potential predictors, such as property type and distance from centre. It also shows how we can visualize the distribution of properties by neighbourhood and by availability. 
# 
# Based on the map visualisations it might be interesting to further consider if there is any noticeable difference in occupancy rates among different areas, what might be the reason?
# 
# 
# Data Source 
# 
# This [data](http://insideairbnb.com/get-the-data.html) was created by Murray Cox.
# 

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from geopy.distance import great_circle
import matplotlib.ticker as ticker
import os


# In[ ]:


df = pd.read_csv("../input/airbnb-berlin/listings.csv")
df.head()


# In[ ]:


df.describe()


# In[ ]:


#To see what data types we have
df.info()


# ### Missing values

# In[ ]:


#To detect the missing data

df.isnull().sum()


# In[ ]:


#To visualise the missing values
fig = plt.figure(figsize=(12,8))
sns.heatmap(df.isnull())
plt.show()


# In[ ]:


#Dealing with missing values
#For the sake of experiment I will remove a couple of price values from the df slice
df3 = df[['neighbourhood','room_type','price']]
df_missingvalues = df3[df3.neighbourhood =='Alexanderplatz'].head(10)
df_missingvalues.loc[48:53,'price'] = 0
df_missingvalues

#use the average value (we work with a slice of the data here)

df_missingvalues[df_missingvalues.room_type == "Entire home/apt"].mean()


# In[ ]:


#replasing the data
#location based imputation
df_missingvalues.loc[48,'price'] = 48.5
df_missingvalues


# In[ ]:


#fill in all missing values simultaneusly with a single value 

df_missingvalues.loc[48:53,'price'] = np.nan 
mean = df_missingvalues['price'].mean() #(use of mean or median would depend on the data and presence of outliers)
df_missingvalues['price'].fillna(mean, inplace=True)
df_missingvalues


# In[ ]:


#Predictive filling using the interpolate() method will perform a linear interpolation to 'find' the missing values
df_missingvalues.loc[48:53,'price'] = np.nan 
df_missingvalues['price'].interpolate(inplace=True)
df_missingvalues


# ### Summary of categorical data

# In[ ]:


df['room_type'].value_counts()


# In[ ]:


df['neighbourhood_group'].value_counts().head()


# In[ ]:


#Using box plots to analyse our data

fig = plt.figure(figsize=(12,8))
sns.boxplot(x='room_type',y='price',data=df, showfliers=False);


# ## Distance from centre

# In[ ]:


#Adding a Distance from centre column

def distance_from_centre(lat, lon):
    berlin_centre = (52.520008, 13.404954)
    apartment_spot = (lat, lon)
    return round(great_circle(berlin_centre, apartment_spot).km, 1)

df["distance"] = df.apply(lambda x: distance_from_centre(x.latitude, x.longitude), axis=1)
df.head()


# ## Distance from centre and property prices 
# The scatter plot confirms the presumption that higher price properties are concentrated in more central areas (let's use areas because a centre is hard to define for the city of Berlin).

# In[ ]:


fig = plt.figure(figsize=(12,8))
plt.scatter(df['distance'],df['price'])
plt.xlim(0,30)
plt.ylim(0,6500)
plt.xlabel('Distance from centre')
plt.ylabel('Price')
plt.show()


# ## Minimum nights distribution 
# The visible concentration of minimum nights, other than the first 7 to 20 days, is around whole months. 

# In[ ]:


f,ax = plt.subplots(figsize=(17,8))
ax = sns.swarmplot(y= df.index,x= df.minimum_nights)
n = 5

for index, label in enumerate(ax.xaxis.get_ticklabels(), 1):
    if index % n != 0:
        label.set_visible(False)

plt.show()


# ## Distribution of price levels
# The histogram shows the distribution of price levels in Berlin.

# In[ ]:


fig = plt.figure(figsize=(12,8))
plt.hist(df['price'], bins=1000)
plt.xlim(0,300)
plt.xlabel('Price')
plt.ylabel('Number of properties')
plt.grid()
plt.show()


# ### Grouping of data to find out more about price levels

# In[ ]:


#To find the average price per neighbourhood

df1 = df[['neighbourhood_group','price', 'distance']]
df_group = df1.groupby(['neighbourhood_group'],as_index=False).mean()
df_group.head(15)


# In[ ]:


#Or group by location and property type to see the average price per each type
df2 = df[['neighbourhood_group','room_type','price']]
df_group2 = df2.groupby(['neighbourhood_group','room_type'],as_index=False).mean()
df_group2.head()


# In[ ]:


#The above can be converted to pivot
df_pivot = df_group2.pivot(index='neighbourhood_group',columns='room_type')
df_pivot.head()


# ### Analysis of variance
# To check the variance of prices in Mitte and Pankow. The F-test score (high) and the p value (low) should confirm that there is no large variance between the two prices.

# In[ ]:


df_var = df1[(df1['neighbourhood_group'] == 'Pankow') | (df1['neighbourhood_group'] == 'Mitte')]
df_vartemp = df_var[['neighbourhood_group','price']].groupby(['neighbourhood_group'])
stats.f_oneway(df_vartemp.get_group('Pankow')['price'],df_vartemp.get_group('Mitte')['price'])


# ### Visualizing the data with folium
# Where are the properties concentrated?

# In[ ]:


import folium
from folium import plugins
from folium.plugins import HeatMap


# In[ ]:


map_test = df.head(200)


# In[ ]:


m = folium.Map(location=[52.52, 13.4], zoom_start = 12)

heat_data = [[row['latitude'],row['longitude']] for index, row in
             map_test[['latitude', 'longitude']].iterrows()]

hh =  HeatMap(heat_data).add_to(m)

m


# ### Property availability by area

# In[ ]:


f,ax = plt.subplots(figsize=(12,8))
ax = sns.scatterplot(y=df.latitude,x=df.longitude,hue=df.availability_365)
plt.show()


# Hope this was helpful:)
