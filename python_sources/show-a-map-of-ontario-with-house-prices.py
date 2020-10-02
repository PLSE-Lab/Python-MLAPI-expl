#!/usr/bin/env python
# coding: utf-8

# **Show a map of Ontario with house prices**
# 
# My goal was to summarize the house prices for each city and show the data on a map of Ontario. The summarized city house prices shows up as a coloured label with the label text showing the  city name and average house price. The color of the label will reflect the city's house prices as compared to all of the Ontario house prices.
# 
# **Step 1**
# 
# Create a city column and get averages for all cities

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('../input/properties.csv', index_col=0)

# Only consider houses worth more than $50K
df = df[df['Price ($)'] > 49000]


# Parse the address and create a city column
df['City'] = df['Address'].str.replace(', ON','')
df['City'] = df['City'].str.split(' ').str.get(-1)

# Remove areas that only have a few houses
df1 = df.groupby(['City']).filter(lambda x: len(x) > 10)

# Get the Average Price for a city
AvePrice = df1.groupby(['City']).mean()

print(AvePrice.head(10))


# **Step 2**
# 
# Put grouped data on to an Ontario map

# In[ ]:


import folium

map_hooray = folium.Map(location= [45.65, -83],
                    zoom_start = 7) # Uses lat then lon. The bigger the zoom number, the closer in you get

# Get the highest average house price
maxave = int(AvePrice['Price ($)'].max())
print("Highest City House Price is: ", maxave)

# Create a color map to match house prices. White - low price, Black - high price
colormap = ['white','lightgray','pink','lightred','orange','darkred','red','purple','darkpurple','black']

# Add marker info 
for index, row in AvePrice.iterrows(): 
    # Set icon color based on price
    theCol = colormap[ int((len(colormap) - 1 ) *  float( row['Price ($)']) / maxave) ]
    # Create a marker text with City name and average price
    markerText =  str(index) + ' ${:,.0f}'.format(row['Price ($)'])
    
    folium.Marker([row['lat'],row['lng']], popup = markerText, 
                  icon=folium.Icon(color= theCol)).add_to(map_hooray)

map_hooray


# **Final Comments**
# 
# I found that *folium* library to be very easy to get started on, unfortunately the marker icon colors appears to only accept color names, and not RGB or HEX colors, so this makes this features rather limiting.
# 
# 
