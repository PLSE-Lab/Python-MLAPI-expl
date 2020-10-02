#!/usr/bin/env python
# coding: utf-8

# # BIKE THEFT AND BIKE SHARE RIDERSHIP ANALYSIS
# 
# 
# I will be exploring two data sets:
# 
#     1. Toronto Bike Theft data :
#     
#             From: The Toronto Police Services
#             Content: The dataset contains information about individual
#                      Bicycle Theft incidents as reported to the Toronto 
#                      Police. Each row contains information of where the
#                      incidents have happened, when they have happened, the
#                      type of location, crime,status,division and more info. 
#                    
#             
#     2. Toronot Bikeshare data :
#     
#             From: Toronto Parking Authority
#             Content: Bikesharing information from 2017 and 2018. It includes
#                      coloumns that that the time/date that the trips started 
#                      and ended as well as the start and end stations. It also
#                      includes the user type of the rider.
#             
#  
# My goal is to find insights that can be used in replicating a bikeshare busines model.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import geopandas as gpd
import matplotlib.pyplot as plt
from numpy import arange

import datetime as dt
import plotly.express as px

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # PART 1: Toronto Bike Theft data

# (Thanks to Jaydeep for his fantastic analysis and visualization for the Toronto Bike Theft data)
# 
# https://www.kaggle.com/jrmistry/plotly-mapbox-tps-data-analyst-interview-notes

# In[ ]:



df_bicycle_thefts = gpd.read_file("../input/tps-toronto-bicycle-thefts/TPS_Toronto_Bicycle_Thefts.geojson")


# # *DATA ANALYSIS & VISUALIZATION*
# 
# I will explore the locations where bike thefts occurred the most.

# In[ ]:


df_bicycle_thefts.info()
df_bicycle_thefts.head()


# In[ ]:


df_bicycle_thefts.describe()


# By exploring the statistical description of the data above, we can see that from 2014 to 2018, at least one bike was stolen per month. The maximum number of bikes stolen a month were 12.
# 
# > #  *An average of 7 bikes were stolen every month.*
# 
# Bike theft is in inevitable and it also a big loss to a bikeshare business. One way to reduce bike theft is by building the bike stations in locations where they are least likely to be stolen. To find this, I will create a bar graph to see the number of thefts that occur in different locations types.

# In[ ]:


theft_location_value = df_bicycle_thefts["Location_Type"].value_counts().iloc[0:15]
theft_location = df_bicycle_thefts["Location_Type"].value_counts().iloc[0:15].index

fig = px.bar(df_bicycle_thefts, x= theft_location , y=theft_location_value)


# In[ ]:


fig.update_layout(
    title='Types of locations with highest number of thefts',
    xaxis_title = 'Location types',
    yaxis_title ='Number of thefts',
    xaxis_type = 'category'
)


fig.update_traces(marker_color='purple')

fig.show()


# There are some interesting insights we can see from the bar chart above.
# 
# TTC Subways and Go Stations have lower number of thefts than expected eventhough they are popular locations for locking bikes.This maybe because people are less likely to steal bikes in crowded areas. Another reason could be because there are existing bikeshare stations near TTC & GO Stations that people actually use. This is also another good sign that shows shared bikes are less likely to be stolen. 
# 
# A competetive edge is required to break into this market and obtain customers.

# # PART 2: Toronot Bikeshare data 
# 
# *Now I will explore Toronto Bikeshare data to find how the trip duration has changed over time by analysing dataset  of 2017-2018 quarters.*

# # *DATA CLEANING*
# 
# 1. Checking if any null/missing values exist int the trip_duration_seconds coloumn in each dataset within the toronto-bikeshare-data.

# In[ ]:


list = [
    '../input/toronto-bikeshare-data/bikeshare2018/bikeshare2018/Bike Share Toronto Ridership_Q1 2018.csv',
    '../input/toronto-bikeshare-data/bikeshare2018/bikeshare2018/Bike Share Toronto Ridership_Q2 2018.csv',
    '../input/toronto-bikeshare-data/bikeshare2018/bikeshare2018/Bike Share Toronto Ridership_Q3 2018.csv',
    '../input/toronto-bikeshare-data/bikeshare2018/bikeshare2018/Bike Share Toronto Ridership_Q4 2018.csv'
       ]



for i in list:  #looped through the list
    #print(i)
    
    ridership = pd.read_csv(i) #read the filepath
    #print(ridership['trip_duration_seconds'].isnull().values.any()) #to check if any nun/missing values exist
    
    #print(ridership.head()) #to explore the first five rows
    #print(ridership.tail()) #to explore the last five rows
    #print(ridership.describe()) #to explore the summary statistics


#  2. Next, I created a function to convert the coloumn of trip duration which was in seconds into minutes.

# In[ ]:


def convert_seconds_to_minutes(seconds):
  minutes = int(seconds / 60);
  seconds = seconds % 60;
  return f"{minutes}"


#return f"{minutes}{seconds}" to also show the seconds


# 3. I created a loop to convert all the data sets to minutes and then convert it back to a series so that we are able to create histograms to visualize it.

# In[ ]:



list = [    
    '../input/toronto-bikeshare-data/bikeshare2018/bikeshare2018/Bike Share Toronto Ridership_Q1 2018.csv',
    '../input/toronto-bikeshare-data/bikeshare2018/bikeshare2018/Bike Share Toronto Ridership_Q2 2018.csv',
    '../input/toronto-bikeshare-data/bikeshare2018/bikeshare2018/Bike Share Toronto Ridership_Q3 2018.csv',
    '../input/toronto-bikeshare-data/bikeshare2018/bikeshare2018/Bike Share Toronto Ridership_Q4 2018.csv'
       ]

trip_duration_item = [] #created a list of data sets


for i in list:  #looped through the list
    #print(i)
    
    ridership = pd.read_csv(i) #read the filepath
    trip_duration_string = [] #create a list for strings
    
    for seconds in ridership['trip_duration_seconds']: #gets the seconds in trip duration
       trip_duration_string.append(convert_seconds_to_minutes(seconds)) #converts it to minutes
    
    
    p = pd.Series(trip_duration_string) #take our new list and make it list
    #print (p)

    trip_duration_item.append(p) #appended it back to our first list trip_duration_item
    #print(trip_duration_item[0])


# # *DATA ANALYSIS & VISUALIZATION*
# 
# I created a function to draw the histograms for the trip durations in minutes. 
# 
# This will allow us to see how long people rode their bikes in different quarters in 2017-2018.
# 
# We can answer questions such as the following:
# 
# * Did the number of people who ride rental bikes increase or decrease throughout the years?
# * Did people ride their bikes for longer in different months of the year?

# Now, I will create a line graph to show the nuber of people that used bike share bikes in the different periods
# 
# (You can use this to find the count: trip_duration_item[0].describe(). You need to include the filepath of the other dataset in List.)

# In[ ]:


year = ['  2017 Q1', '   2017 Q2', '   2017 Q3', '   2017 Q4', '   2018 Q1', '   2018 Q2', '   2018 Q3', '   2018 Q4']
people_num = [132123, 333353, 663488, 363405, 178559, 558370, 822536, 363490]
#ax = plt.plot

plt.plot(year, people_num, c='DarkGoldenRod')

plt.xticks(rotation = 60) 
plt.tick_params(color = 'DarkGoldenRod', labelsize = '11', pad=20)


#plt.spines["right"].set_visible(False)
    
font = {'fontsize': 20,
 'color' : 'Black',
 }

plt.title('Number of people that use bike-share in different times/quarters', fontdict= font)
plt.xlabel('PERIOD')
plt.ylabel('RIDERSHIP')

plt.show()


#  Q1 : Jan-Mar
#  Q2 : Apr-Jun
#  Q3 : Jul-Sep
#  Q4 : Oct-Dec 
# 
# 2017
# Q1 = 132123
# Q2 = 333353
# Q3 = 663488
# Q4 = 363405
# 
# 2018
# Q1 = 178559
# Q2 = 558370
# Q3 = 822536
# Q4 = 363490
# 
# 
# 
# As expected, bikeshare ridership peaked drastically during the warmer months. However, we can also see that the minimum and maximum also increased as time passed. The increase in the two maximimum is a much greater increase and this shows that during the peak seasons, bikeshare gained popularity from one year to the next. 
# 
# # *This show that ridership continious to increase over the years and peaks in the warmer seasons.*

# Now I will draw histogram for each period to further explore them.

# In[ ]:


def draw_histogram(trip_duration_item, color, range_max, bin_num, hist_title): #creating a function to create histograms

    col = [color]
    x_label = {'x':'Duration(in minutes)'}
    x_range = [0, range_max]

    fig = px.histogram(x=trip_duration_item,
                       labels = x_label,
                       color_discrete_sequence= col,
                       marginal= 'box',
                       barmode = 'overlay',
                       range_x= x_range,
                       nbins=bin_num,
                       title= hist_title,
                       width=800, 
                       height=500)

    fig.show()


# I will now further analyze the datasets in each quarter of 2018 to see if there are any insights about the ridership.

# In[ ]:


#draw_histogram(trip_duration_item[0],'SkyBlue', 30, 700, 'Trip duration of people who used bike-share(Jan-Mar 2017)')
#draw_histogram(trip_duration_item[1],'Pink', 30, 700,'Trip duration of people who used bike-share(Apr-Jun 2017)')
#draw_histogram(trip_duration_item[2],'Plum', 30, 160000,'Trip duration of people who used bike-share(Jul-Sep 2017)')
#draw_histogram(trip_duration_item[3],'PaleVioletRed', 30, 40000, 'Trip duration of people who used bike-share(Oct-Dec 2017)')


# In[ ]:


draw_histogram(trip_duration_item[0],'Teal', 30, 5000, 'Trip duration of people who used bike-share(Jan-Mar 2018)')


# In[ ]:


draw_histogram(trip_duration_item[1],'MediumVioletRed', 30, 5000,'Trip duration of people who used bike-share(Apr-Jun 2018)')


# In[ ]:


draw_histogram(trip_duration_item[2],'DarkOrange', 30, 5000,'Trip duration of people who used bike-share(Jul-Sep 2018)')


# In[ ]:


draw_histogram(trip_duration_item[3],'SteelBlue', 30, 6400, 'Trip duration of people who used bike-share(Oct-Dec 2018)')


# The first thing that stands out is the majority of people only ride their bikes for 10 minutes or less. The peak minute that people rode the bikes for varies between 6 to 7 minutes. 
# 
# * Q1 peak - 6mins : 14.60k people
# * Q2 peak - 7mins : 34.404k people
# * Q3 peak - 7mins : 48.802k people
# * Q4 peak - 6mins : 28.441 people
# 
# You can see that a lot more people ride their bikes a little longer during the warmer months (Q2 & Q3). To see this even better. Lets see the number of people that ride their bikes for 20mins. 
# 
# * Q1 20mins : 2,593 people
# * Q2 20mins : 11.87k people
# * Q3 20mins : 19.15k people
# * Q4 20mins : 5,817 people
# 
# Once again, we can see that more people ride their bikes for much longer in the summer. The number of people who only ride their bikes for 7minutes is almost triple the number of people ride their bikes for 20mins.This shows that there is a gap that could be filled by giving more riders an incentive to ride for longer. These things could include a points program, a discount or even competitions for the users.
# 

# In conclusion, by analysing the Toronto Bike theft data, I found that building bike stations near the Subway and Go station is the most profitable location since it is least likely to be stolen. Next, by analyzing the  Toronto BikeShare data, I found that ridership drastically increased from 2017 to 2018 although there was seasonal fluctuations in both years. Last but not least, ecouraging more people to ride thier bikes during the summer for longer can help fill a gap in the market and inspire people to stay healthier.
