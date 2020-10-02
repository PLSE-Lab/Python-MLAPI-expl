#!/usr/bin/env python
# coding: utf-8

# Our aim is to explore the dataset we have and use it for visual representation on the Indian map to help us have a visual aid in order to understand and analyze the data. The three main tools we will be using for this notebook are numpy, pandas, matplotlib, and a toolkit within matplotlib called basemap (which lets us do the fun stuff)

# In[ ]:


#import the various libraries we require for the analysis 
import numpy as np
import pandas as pd
from numpy import array
import matplotlib as mpl

#import the tools we need to plot
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib import cm #cm is a tool that allows us to gain access to a predefined set of colours in matplotlib


# Reading all the data into the notebook and storing it

# In[ ]:


cities = pd.read_csv("../input/top-500-indian-cities/cities_r2.csv")


# Let us now view the data and get an idea as to what it is that we're dealing with

# In[ ]:


#print the data for the first five cities in the data set
cities.head()


# In[ ]:


#prints an overview of each columns with the number of entries in each column and the data type
cities.info()


# In[ ]:


#gives an even more detailed view into each column
cities.describe()


# Let us now plot a bar graph to get an idea as to how many cities have been taken for analysis from the various states

# In[ ]:


#we shall now set the size of the figure we are plotting
fig = plt.figure(figsize=(20,20))

#sorting the states in ascending order of the number of times a city from that state is present in the data set
states = cities.groupby('state_name')['name_of_city'].count().sort_values(ascending=True) 

#mention the sort of graph we want
states.plot(kind='barh', fontsize=18)
plt.grid(b=True, which='both', color='black', linestyle='-')
plt.xlabel('No of cities taken for analysis', fontsize=20)


# As we can see, most of the cities in the data set have been taken from the states of Uttar Pradesh and West Bengal (each over 60), which is a little surprising because even though Uttar Pradesh is the most populous state in India, Maharashtra comes next

# Since we want to plot the data on an Indian map, we need the latitude and longitude of each city. If we go back and view the data set, we notice that the latitude and longitude is stored together under location. So to make our job much easier, we can split that into latitude and longitude

# In[ ]:


#We split the data in location using the delimiter ',' and since the first index i.e., [0] is the latitude coordinate,
#we store and we did the same for longitude
cities['latitude']=cities['location'].apply(lambda x:x.split(',')[0])
cities['longitude']=cities['location'].apply(lambda x:x.split(',')[1])


# Now let us find the top 10 most populated cities in India from the data set

# In[ ]:


#sort the cities according to population
top_pop_cities=cities.sort_values(by='population_total',ascending=False)

#store the top 10 cities under a separate name
top10_pop_cities=top_pop_cities.head(10)
top10_pop_cities


# Now on to the fun part.
# It is useful to have the data but it would be quite tedious to explain it to someone using just the data set. A visual representation would be much more appropriate and helpful.
# So let us plot this data on a map of India

# In[ ]:


#specify the size of the figure
fig=plt.figure(figsize=(20,15))

#setting up Basemap
#width and height are self-explanatory, projection is the type of map we want
#resolution can be low or high
#llcrnrlon is lower left corner longitude and urcrnrlon is upper right corner longitude
#lat_0 is the central latitude of the map
map = Basemap(width=1200000,height=900000,projection='lcc',resolution='l',llcrnrlon=67,llcrnrlat=5,urcrnrlon=99,urcrnrlat=37,lat_0=22,lon_0=78)

#draws the map boundaries, the countries, and the coastlines
map.drawmapboundary()
map.drawcountries()
map.drawcoastlines()

#storing all the longitudes in a numpy array
lg=array(top10_pop_cities['longitude'])

#storing all the latitudes in a numpy array
lt=array(top10_pop_cities['latitude'])

#storing the population of each of the top 10 cities in a numpy array
pt=array(top10_pop_cities['population_total'])

#storing the names of the top 10 cities in a numpy array
nc=array(top10_pop_cities['name_of_city'])

x,y=map(lg,lt)

#choosing the size of the circle that will be plotted on the map, corresponding to the population
population_size=top10_pop_cities['population_total'].apply(lambda x:int(x/5000))

#s=number of entries, c=value for the population which determines the color, cmap=pre-existing color map, aplha=opacity
map.scatter(x, y, s=population_size, c=population_size, marker="o", cmap=cm.Dark2, alpha=0.7)

#printing the name of each city
for ncs,xpt,ypt in zip(nc,x,y):
    plt.text(xpt+60000,ypt+30000,ncs,fontsize=12, fontweight='bold')

#title of the map
plt.title("Top 10 most populous cities in India",fontsize=18)


# To understand and to put this into context, India is the second most populated country in the world, hence we expect the population density of India to be quite an interesting visual. 
# So let us go ahead and plot the entire population of India on the map

# In[ ]:


#since it is tedious to write the code for drawing a map every single time, let us go ahead and write a function that draws a map
#we can then call this function every single time we require a map

def plot_map(sizes, colorbarValue):
    plt.figure(figsize=(19,20))
    f, ax = plt.subplots(figsize=(19,20))
    map = Basemap(width=5000000,height=3500000,projection='lcc',resolution='l',llcrnrlon=67,llcrnrlat=5,urcrnrlon=99,urcrnrlat=37,lat_0=22,lon_0=78,ax=ax)
    map.drawmapboundary()
    map.drawcountries()
    map.drawcoastlines()

    x, y = map(array(cities['longitude']),array(cities['latitude']))
    cs = map.scatter(x, y, s=sizes, c=sizes, marker="o", cmap=cm.get_cmap('Dark2'),alpha=0.8)

    #setting up a colorbar which acts as a legend
    cbar = map.colorbar(cs, location='right',pad="2%")
    cbar.ax.set_yticklabels(colorbarValue)


# Let us now use the created function to plot the population density of India

# In[ ]:


population_sizes=cities["population_total"].apply(lambda x:int(x/5000))

#setting the lowest and highest value on the colorbar
colorbarValue=np.linspace(cities['population_total'].min(),cities['population_total'].max())

#converting the value on the legend from float to int
colorbarValue=colorbarValue.astype(int)

plot_map(population_sizes,colorbarValue)


# Let us now analyze the literacy rate of various states

# In[ ]:


#specifying the size of the figure
fig=plt.figure(figsize=(20,20))

#grouping states according to the total number of literates in that state
states=cities.groupby('state_name')['literates_total'].sum().sort_values(ascending=True)

#specifying the type of graph
states.plot(kind="barh", fontsize = 20)

#b = to show the grid lines
plt.grid(b=True, which='both', color='Black',linestyle='-')

#label the x-axis
plt.xlabel('Total literacy rate of states', fontsize = 20)


# We see Maharashtra, UP, and West Bengal have the highest literacy rate, which corresponds with the population. But Bihar, on the other hand, has quite a low literacy rate, corresponding to its population.

# Let us now find and plot the top 10 most literate cities in India

# In[ ]:


#ranking the cities according to the total literates in the city
top_literate_cities = cities.sort_values(by='literates_total',ascending=False)

#separating the top 10 cities with the highest number of literates
top10_literate_cities=top_literate_cities.head(10)
top10_literate_cities


# Incidentally, this is the same as the top 10 most populous cities in India

# If we want to plot this, we can call the plot function again or just code the map again. 

# Let us now plot the top 10 cities with the highest number of literates

# In[ ]:



plt.subplots(figsize=(20, 15))
map = Basemap(width=1200000,height=900000,projection='lcc',resolution='l',
                    llcrnrlon=67,llcrnrlat=5,urcrnrlon=99,urcrnrlat=37,lat_0=28,lon_0=77)

map.drawmapboundary ()
map.drawcountries ()
map.drawcoastlines ()

lg=array(top10_literate_cities['longitude'])
lt=array(top10_literate_cities['latitude'])
pt=array(top10_literate_cities['literates_total'])
nc=array(top10_literate_cities['name_of_city'])

x, y = map(lg, lt)
population_sizes_literates_total = top10_literate_cities["literates_total"].apply(lambda x: int(x / 5000))
plt.scatter(x, y, s=population_sizes_literates_total, marker="o", c=population_sizes_literates_total, cmap=cm.Dark2, alpha=0.7)


for ncs, xpt, ypt in zip(nc, x, y):
    plt.text(xpt+60000, ypt+30000, ncs, fontsize=10, fontweight='bold')

plt.title('Top 10 Cities with the most literates in India',fontsize=20)


# Let us plot the data for literacy across the country

# In[ ]:


population_sizes = cities["literates_total"].apply(lambda x: int(x / 5000))
colorbarValue = np.linspace(cities["literates_total"].min(), cities["literates_total"].max())
colorbarValue = colorbarValue.astype(int)

plot_map(population_sizes, colorbarValue)


# As we see, major metropolitan cities again show high population of literate people

# Let us now find the effective literacy amongst states

# In[ ]:


#grouping the states according to the aggregate of the average effective literacy rate, average male literacy rate, average female literacy rate
state_literacy_effective = cities[["state_name","effective_literacy_rate_total","effective_literacy_rate_male","effective_literacy_rate_female"]].groupby("state_name").agg({"effective_literacy_rate_total":np.average,
                                                                                                "effective_literacy_rate_male":np.average,
                                                                                                "effective_literacy_rate_female":np.average})

#specifying the type and details of the graph
state_literacy_effective.sort_values("effective_literacy_rate_total", ascending=True).plot(kind="barh",
                      grid=True,
                      figsize=(16,15),
                      alpha = 0.6,
                      width=0.6,
                      stacked = False,
                      edgecolor="g",
                      fontsize = 20)
plt.grid(b=True, which='both', color='black',linestyle='-')


# It is fascinating to see how the states that are often considered as rural have a higher effective literacy rate than states like UP or West Bengal.
