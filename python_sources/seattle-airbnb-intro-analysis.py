#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ****

# **Business Understanding**

# **Introduction**
# I recently visited Seattle with a number of good friends and had nothing but positive comments about the city. I grew up in the San Francisco/Bay Area all my life and when I stepped foot in Seattle I saw it as a fresh start with exciting new opportunities. In this notebook I will be taking my first step into data science so what better data to explore than with the city that started my curiosity into tech.
# 
# I have three questions I want to answer with this dataset. 
# 1. Where are the most expensive neighbourhoods to rent an AirBnB?
# 2. How are property types dispersed throughout Seattle?
# 3. Are there areas of potential opportunities depending on neighbourhoods?
# 
# I will try to keep this as informative as possible for other beginner data scientists and programers like myself and will include YouTube tutorials I looked at to aid in this project.

# First off let's import the libraries we will need.

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from decimal import Decimal
import datetime
import folium #Longitude and Lattitude mapping.
from sklearn.preprocessing import LabelEncoder, OneHotEncoder #We don't this in the analysis I am currently showing.
from sklearn.linear_model import LinearRegression #We don't this in the analysis I am currently showing.
import seaborn as sns
from itertools import *
import os
import folium
from folium import plugins
from folium.plugins import MarkerCluster #To be able to cluster our individual data points on folium.
from IPython.display import HTML, display


# Note: When working with kaggle datasets that are already in the system. Double click the csv files to get the file path of kaggle for the import below. In this case the file path is input and seattle (see below).

# **Data Understanding**

# In[ ]:


calendar = pd.read_csv('../input/seattle/calendar.csv')
listing = pd.read_csv('../input/seattle/listings.csv')
reviews = pd.read_csv('../input/seattle/reviews.csv')


# In[ ]:


reviews.head()


# In[ ]:


calendar.head()


# There are a number of NaN values in the calendar and reviews dataset. I will just drop them using the drop method. Usually I would only drop columns that are irrelevant to the analysis and replace NaN values with the mean or mode of the dataset. However, I won't be using the calendar and review datasets much in my analysis but will do a basic cleanup just incase.
# 
# The listing dataset looks very useful so I will take a deeper look into it rather than just dropping all NaN values. 
# 
# 

# In[ ]:


listing.head()


# **Data Preparation**

# I cannot work with this data yet because I cannot convert the data objects anywhere with dollar signs and commas associated with pricing. I will use a function to replace all dollar signs and commas with blanks so I can convert it to a numeric dataset. 

# I used this YouTube tutorial, StackOverFlow, and Kaggle for the code below regarding lambda expressions. [https://www.youtube.com/watch?v=25ovCm9jKfA&t=111s](http://) 
# 
# I am just going through each row in the columns that have a dollar sign and comma and removing them.

# In[ ]:


calendar = calendar.dropna(axis = 0, subset = ['price'], how = 'any')
reviews = reviews.dropna(axis = 0, subset = ['comments'], how = 'any')
calendar['date'] = pd.to_datetime(calendar['date'])
calendar['month'] = calendar.date.dt.month
calendar['year'] = calendar.date.dt.year
calendar['day'] = calendar.date.dt.day
calendar['price'] = pd.to_numeric(calendar['price'].apply(lambda x: str(x).replace('$', '').replace(',', '')), errors='coerce')

listing['monthly_price'] = pd.to_numeric(listing['monthly_price'].apply(lambda x: str(x).replace('$', '').replace(',', '')), errors='coerce')
listing['weekly_price'] = pd.to_numeric(listing['weekly_price'].apply(lambda x: str(x).replace('$', '').replace(',', '')), errors='coerce')
listing['price'] = pd.to_numeric(listing['price'].apply(lambda x: str(x).replace('$', '').replace(',', '')), errors='coerce')
listing['cleaning_fee'] = pd.to_numeric(listing['cleaning_fee'].apply(lambda x: str(x).replace('$', '').replace(',', '')), errors='coerce')
listing['security_deposit'] = pd.to_numeric(listing['security_deposit'].apply(lambda x: str(x).replace('$', '').replace(',', '')), errors='coerce')
listing['extra_people'] = pd.to_numeric(listing['extra_people'].apply(lambda x: str(x).replace('$', '').replace(',', '')), errors='coerce')
listing = listing.rename(columns = {'id':'listing_id'})


# Now that all the data is decently cleaned I wanted to take a deeper look at the columns I am dealing with. I will use the code below that goes through each column and gives me the sum of NaN values. I just divide the sum by the length of the dataset to give me a percentage of NaN values.
# 
# Another method is just use the describe function and divide by the length of the listing data (below). You just need to make sure you read it inversely (100% means all values are accounted for and 80% means 20% are NaN.
# 
# listing.describe()/len(listing)

# In[ ]:


(listing.isnull().sum()[listing.isnull().sum().nonzero()[0]])/len(listing) 


# I can see there are a lot of featues that have a high NaN percentage within the listing dataset. However, there are a few
# columns that can be used to create a robust analysis. 
# 
# 1. property_type
# 2. neighbourhood
# 2. review_scores_value
# 3. bathrooms
# 4. bedrooms
# 5. price #You can't find this column in the above list. I had to use listing.columns to find it.
# 6. longitude
# 7. latitude
# 
# I will create a new dataset that includes these features and call it new_list.
# 

# In[ ]:


default_list = listing[['property_type', 'neighbourhood', 'review_scores_value', 
                        'bathrooms', 'bedrooms', 'price', 'longitude', 'latitude']]

new_list = default_list.dropna(axis = 0, how = 'any')
new_list


# Great! We now have a numeric dataset that we can run some basic analysis on. Let's use seaborn to plot out some basic info regarding this new dataset. I will look at the count of each property type.

# In[ ]:


def Plot(cur, data_list):
    """Description: This function can be used to plot a graph by reading the data_list and grouping by the cursor obejct.
    
    Arguments: 
    cur: the cursor object.
    data_list: list of data.
    
    Returns:
    A graphical reprementation of the cur items in the data_list."""


    plt.figure(figsize=(20,20))
    plt.xticks(rotation=90)
    sns.countplot((data_list)[(cur)],
                 order = data_list[cur].value_counts().index)
    plt.show()


# In[ ]:


(new_list.property_type.value_counts())/(new_list.property_type.count())


# In[ ]:


Plot('property_type', new_list)


# A majority of the properties listed are Apartments and Houses which I will focus on later in the analysis.
# 
# I will do the same thing for neighbourhood.

# In[ ]:


(new_list.neighbourhood.value_counts())/(new_list.neighbourhood.count())


# In[ ]:


Plot('neighbourhood', new_list)


# Immediately I see there are a few property types that hold a majority of the dataset and a number of them that represent a very small percentage of the dataset. The same can be said for neighbourhoods. 
# 
# new_list.groupby('neighbourhood').nunique() - This gives me 79 unique values for the neighbourhood column.
# 
# Therefore, I want to simplify this list to limit the number of features I have while still maintaining the integrity of the data. 

# Let's start with grouping the new_list dataset by neighbourhood and counting them up. 
# 
# I will then use the nlargest function to give me the top 35 neighbourhoods.

# In[ ]:


new_list_neighbourhood = new_list.groupby('neighbourhood').count()
new_list_top_15_neighbourhood = new_list_neighbourhood.nlargest(35,'property_type')
new_list_top_15_neighbourhood


# I now have a list of the top 35 neighbourhoods that represents about 86.72% of the dataset. I am going to put this in a list that I can use to filter our original dataset and remove the neighbourhoods that hold very little data.

# In[ ]:


neighbourhood_list = ['Capitol Hill',
'Ballard',
'Belltown',
'Minor',
'Queen Anne',
'Fremont',
'Wallingford',
'First Hill',
'North Beacon Hill',
'University District',
'Stevens',
'Central Business District',
'Lower Queen Anne',
'Greenwood',
'Columbia City',
'Ravenna',
'Magnolia',
'Atlantic',
'North Admiral',
'Phinney Ridge',
'Green Lake',
'Leschi',
'Mount Baker',
'Eastlake',
'Maple Leaf',
'Madrona',
'Pike Place Market',
'The Junction',
'Seward Park',
'Bryant',
'Genesee',
'North Delridge',
'Roosevelt',
'Crown Hill',
'Montlake']


# I will use our neighbourhood_list we created (above) and see if our new_list dataset has a value within that list. This will filter out 44 (55.70%) of the unique neighbourhood categories while only removing 13.27% of the data.

# I used this youtube tutorial to help on this part. [https://www.youtube.com/watch?v=2AFGPdNn4FM](http://)

# In[ ]:


#This gives me a list of True/False statements for each row if the value in the neighbourhood columns is in 
#neighbourhood_list above.
true_false_by_neighbourhood = new_list.neighbourhood.isin(neighbourhood_list) 

#I can then put this new list of True/False statements into our origional new_list. This filters new_list down
# to 35 categories of the neighbourhood column while still containing 86.72% of the origional data.
filtered_neighborhood = new_list[true_false_by_neighbourhood]
filtered_neighborhood


# Let's do the same thing for property_type.

# In[ ]:


new_list_property_type = new_list.groupby('property_type').count()
new_list_top_16_property_type = new_list_property_type.nlargest(16, 'neighbourhood')
new_list_top_16_property_type



# Since about 98.55% of property_types are in the top 7. I will just take the top seven of property types for my list:
# 

# Now let's clean up the list by property_type.

# In[ ]:


property_type_list = ['House', 'Apartment', 'Townhouse', 'Condominium', 'Loft', 'Bed & Breakfast', 'Cabin']


# In[ ]:


#This gives me a list of True/False statements for each row if the value in the property_type column is in 
#the property_type_list above.
true_false_by_property = filtered_neighborhood.property_type.isin(property_type_list) 

#I can then put this new list of True/False statements into our filtered_neighbourhood list. 
#This filters new_list down seven property types while still containing 98.30% of the data of filtered_neighbourhood.
filtered_data = filtered_neighborhood[true_false_by_property]
filtered_data


# Our list filtered_data took our new_list and removed the property_types and neighbourhoods that contained very little data. As you can see, we still have 2428 data points from our 2463 points in new_list.

# I want to be able to effective measure how expensive each neighbourhood is. I suspect there is a correlation with number of bedrooms listed and price of the property. I will use the below code to plot that. 

# In[ ]:


sns.lmplot(data=filtered_data, x='bedrooms', y='price', hue='review_scores_value')


# The chart above indicates that there is a correlation with the number of rooms and price. If I want to view how expensive each neighbourhood I will add another column of price/number of rooms. This removes the potential for a neighbourhood to be more "expensive" just because it has a higher number of rooms available than average.

# In[ ]:


room_premium = (filtered_data.price)/(filtered_data.bedrooms)
filtered_data['Cost Per Bedroom'] = filtered_data['latitude'].add(room_premium)
filtered_data


# We now have a column at the end that gives the price for a single room.

# I'm now going to create a function that allows me to filter out whichever property_type I want while preserving the original data for further analysis. 

# In[ ]:


def Filterlist(cur, filepath):
    """Description: This function can be used to read the file and filter based on the input.
    
    Arguments: 
    cur: the cursor object.
    filepath: data file
    
    Returns: 
    The data file that is filtered by the cur object."""
    
    property_type = [cur]
    true_false_by_property = filtered_data.property_type.isin(property_type) 
    List = filtered_data[true_false_by_property]
    return List


# I am going to use this Filterlist function to filter by 'House' and run some analysis on that section of the data. 

# In[ ]:


House_list = Filterlist('House', filtered_data)
House_list


# **Data Modeling**

# Now that we have a filtered, clean data list. Let's start plotting our data and see what information we can glean.

# In[ ]:


def catplot(x_data, y_data, data_list):
    """Description: This function can be used to read the file in the data_list and create a catplot based on the
    x_data and y_data.
    
    Arguments: 
    x_data: Column in data_list that you want to plot on the x axis (Put as string). 
    y_ data: Column in the data_list that you want to plot on the y axis (Put as string).
    data_list: The datalist that contains the x_data and y_data as columns."""
    sns.catplot(x=x_data, y=y_data, data=data_list, height=8)


# In[ ]:


catplot('Cost Per Bedroom', 'neighbourhood', House_list)


# In[ ]:


Apartment_list = Filterlist('Apartment', filtered_data)
Apartment_list


# In[ ]:


catplot('Cost Per Bedroom', 'neighbourhood', Apartment_list)


# It seems that there is much more varability in terms of price for houses depending on the neighbourhood.
# 
# Queen Anne, Ballard, Minor, Capitol Hill, and Stevens looked among the pricer neighbourhoods. However, that is debatable since the varability is so high.
# 
# For Apartments there is less varability in the price per room. 
# 
# Queen Anne, Ballard, Lower Queen Anne, Minor, Capitol Hill, Pike Place Market, Central Business District, Belltown, and Firsthill ranked among the most expensive. 

# I noticed some districts appeared in the Apartment_list that were not amoung the House_list. I am going to plot the points on a map using each AirBnB latitiude and longitude coordinates and compare the two lists.

# I used this YouTube tutorial to learn about folium. I tried basemap and geopandas first but ran into issues with my system. [https://www.youtube.com/watch?v=4RnU5qKTfYY&t=796s](http://)

# **Map of Houses**

# In[ ]:


def folium_plot(x, y, data_list):
    """Find the latitude and longitude columns in data_list and plots them. 
    Then it custers the data into groups to be viewed on different levels.
    
    Arguements: 
        latitude: columns containing latitude coordinates and labeled as latitude. Write as string.
        longitude: columns containing longitude coordinates and labeled as longitude. Write as string.
        data_list: Data list where columns are held.
    
    Returns:
    Clustered map of all data points."""
    
    #Creates a map of Seattle.
    m = folium.Map(location=[47.60, -122.24], zoom_start = 11)
    m.save('index.html')


    #Takes the latitude and longitude coordinates and zips them into a form to be plotted.
    lat = pd.to_numeric(data_list[x], errors = 'coerce')
    lon = pd.to_numeric(data_list[y], errors = 'coerce')

    #Zip togethers each list of latitude and longitude coordinates. 
    result = zip(lat,lon)
    lat_lon = list(result)


    mc = MarkerCluster().add_to(m)
    for i in range(0,len(data_list)):
        folium.Marker(location=lat_lon[i]).add_to(mc)

    m.save('index.html')
    display(m)


# In[ ]:


folium_plot('latitude', 'longitude', House_list)


# **Map of Apartments**

# In[ ]:


folium_plot('latitude', 'longitude', Apartment_list)


# **Results Exaluation**

# Comparing the two maps it looks to be that Apartments are clustered around central Seattle (Neighbourhood: Belltown) while Houses are more dispersed up north (Neighbourhoods: Ballard, Fremont, and Madrona). This can explain the variability in terms of price per room since Apartments are clustered in one area and houses are dispersed throughout Seattle. 
# 
# In addition, the maps above indicate areas of high AirBnB listings which imply high AirBnB demand. However there are many areas close to high density AirBnB clusters that have little AirBnB's avaliable. My next step would be to create a model to predict prices depending on the features listed in this analysis and compare that to the areas that have few AirBnB rentals but have a high predicted listing price based on their proxity to current listings. We can then run further analysis to see if there are opportunities to promote more AirBnB listings there and updating our models to obtain live data from the AirBnB API's. 

# I wanted to thank the Kaggle community and my mentors for providing me this opportunity to learn. I just started my Data Science journey a couple of weeks ago and everything I used in this analysis I had to scourge StackOverFlow and YouTube tutorials (while repeatedly banging my head over my keyboard). I would greatly appreciate any suggestions or helpful tips on what to learn next and improve upon. Thank you.
# 

# In[ ]:





# ****
