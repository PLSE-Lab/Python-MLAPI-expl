#!/usr/bin/env python
# coding: utf-8

# # NYC taxi trip Exploratory Data Analysis(EDA)
# 
# 
# 
#  File descriptions
#  
#  - train.csv - the dataset (contains 1458644 trip records in New York city in the year 2016)
#  
#  Data fields
# - id - a unique identifier for each trip
# - vendor_id - a code indicating the provider associated with the trip record
# - pickup_datetime - date and time when the meter was engaged
# - dropoff_datetime - date and time when the meter was disengaged
# - passenger_count - the number of passengers in the vehicle (driver entered value)
# - pickup_longitude - the longitude where the meter was engaged
# - pickup_latitude - the latitude where the meter was engaged
# - dropoff_longitude - the longitude where the meter was disengaged
# - dropoff_latitude - the latitude where the meter was disengaged
# - store_and_fwd_flag - This flag indicates whether the trip record was held in vehicle memory before sending to the vendor because the vehicle did not have a connection to the server - Y=store and forward; N=not a store and forward trip
# - trip_duration - duration of the trip in seconds
# 
# 
# Objective:
# ----------
# - Explore the data and find out which features/variables explains the trip duration using visualization tools
# - Explore the independent features in the dataset and visualise its behaviours 
# 
# 
# Business questions identified:
# -------------------------------
# - 1) What Number of passengers are taking most number of taxi trips in New York ?
# - 2) Which Vendor has the highest market share ?
# - 3) Which vendor has the better infrastructure in terms of storing the trip records or connectivity with the server ?
# - 4) What is the count of number of trips taken across all:
#       - 31 days in the month ?
#       - 24 hours in a day ?
#       - 7 days in a week ?
#       - 12 months in that year ?
# - 5) How is the Distance of a Trip affecting the Trip duration ?
# - 6) What is the Average duration of trips across all:
#       - 31 days in the month ?
#       - 24 hours in a day ?
#       - 7 days in a week ?
#       - 12 months in that year ?
# - 7) What is the average distance travelled across all:
#       - 31 days in the month ?
#       - 24 hours in a day ?
#       - 7 days in a week ?
#       - 12 months in that year ?
# - 8) What is the average SPEED of the trips driven across all:
#       - 31 days in the month ?
#       - 24 hours in a day ?
#       - 7 days in a week ?
#       - 12 months in that year ?

# In[ ]:


#importing necessary python packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Read the CSV file

#my_local_path = "B:/UPX docs/Machine Learning/Project_datasets/Project datasets modified/NYC Taxi Trip/NYC Taxi Trip/"
taxi_data = pd.read_csv('../input/train.csv')
taxi_data.head(5)


# In[ ]:


# Seperating the datetime stamp into two seperate columns for pickup_datetime and pickup_datetime

#taxi_data['dropoff_date']=pd.to_datetime(taxi_data['dropoff_datetime']).dt.date
#taxi_data['dropoff_time']=pd.to_datetime(taxi_data['dropoff_datetime']).dt.time
#taxi_data['pickup_date']=pd.to_datetime(taxi_data['pickup_datetime']).dt.date
#taxi_data['pickup_time']=pd.to_datetime(taxi_data['pickup_datetime']).dt.time
#taxi_data['pickup_datetime'] = pd.to_datetime(taxi_data.pickup_datetime) 


# # Tranforming the given datetime stamp into actual datetime format

# In[ ]:


taxi_data['pickup_datetime'] = pd.to_datetime(taxi_data.pickup_datetime) 


# In[ ]:


taxi_data.head()


# ![image.png](attachment:image.png)

# In[ ]:


taxi_mod = taxi_data
taxi_mod.info()


# ![image.png](attachment:image.png)

# #  Finding the distance between pick_up longitude and Latitude & drop_off longitude and Latitude using haversine

# In[ ]:


from haversine import haversine


# In[ ]:


def calc_distance(df):
    pickup = (df['pickup_latitude'], df['pickup_longitude'])
    drop = (df['dropoff_latitude'], df['dropoff_longitude'])
    return haversine(pickup, drop) 


# In[ ]:


taxi_mod['distance'] = taxi_mod.apply(lambda x: calc_distance(x), axis = 1)


# In[ ]:


#Calculate the Speed using the trip_duration and distance data

taxi_mod['trip_duration_hrs']=taxi_mod['trip_duration']/3600


# In[ ]:


taxi_mod['SPEED']=taxi_mod['distance']/taxi_mod['trip_duration_hrs']


# In[ ]:


taxi_mod.head()


# ![image.png](attachment:image.png)

# In[ ]:


#taxi_mod1=taxi_mod


# In[ ]:


#taxi_mod1 = taxi_mod1.drop(columns=['id','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude'])
#taxi_mod1.head()


# In[ ]:


import pandas_profiling
profile = pandas_profiling.ProfileReport(taxi_mod)
profile.to_file(outputfile="taxi_mod.html")


# In[ ]:


corr = taxi_mod.corr()
corr


# ![image.png](attachment:image.png)

# In[ ]:


sns.heatmap(corr,annot=True)
plt.show()


# ![image.png](attachment:image.png)

# From above Heat Map it is observed that there is positive correlation between:
# - Trip_duration and Distance
# - SPEED and Distance

# # Univariate analysis for: 
# - passenger_count
# - vendor_id
# - store_and_fwd_flag

# In[ ]:


plt.figure(figsize=(5,8))
total = float(len(taxi_mod))
plt.subplot(2,1,1)
ax=sns.countplot(x='passenger_count', data=taxi_mod)
plt.ylabel('number of trips')
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format(height/total),
            ha="center")
    
plt.figure(figsize=(5,8))
plt.subplot(2,1,2)
bx=sns.countplot(x='vendor_id', data=taxi_mod)
plt.ylabel('number of trips')
for p in bx.patches:
    height = p.get_height()
    bx.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format(height/total),
            ha="center")
    
plt.show()


# ![image.png](attachment:image.png)

# Observation from countplots of number of passengers 
# - Most number of trips are found to be taken by single passenger (greater than 700,000 out of 1,458,644 total trips)
# - 71% of the trips were travelled by single passenger
# - 14% of the trips were travelled by two passenger
# - 4% of the trips were travelled by three passenger
# - 11% of the trips were travelled by four and more passenger

# Observation from countplots of vendor_id 1 and 2
# - 53% of the trips are served by vendor 2 and 47% by vendor 1
# - Hence highest marketshare is owned by Vendor 2

# In[ ]:


plt.figure(figsize=(5,4))
dx=sns.countplot(x='store_and_fwd_flag', data=taxi_mod)
for p in dx.patches:
    height = p.get_height()
    dx.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format(height/total),
            ha="center")
plt.show()


# ![image.png](attachment:image.png)

# Above countplot shows that:
# - 99% of the trips records were not stored in the vehicle
# - 1% of the trips records were stored in the vehicle due to lack of connection to server at the time of trip

# In[ ]:


plt.figure(figsize=(10,8))
cx=sns.factorplot(x='store_and_fwd_flag', col='vendor_id', kind='count', data=taxi_mod);
plt.show()


# ![image.png](attachment:image.png)

# From above factorplot it is observed that:
# - Vendor 1 had the BETTER infrastructure for the offline storage of trip records
# OR
# - Vendor 1 has POOR connectivity with server
# 
# AND 
# 
# - Vendor 2 had the POOR infrastructure for the offline storage of trip records
# OR
# - Vendor 2 has BETTER connectivity with server

# # Univariate analysis for:
# - Hour_of day
# - month_of_date
# - day_of_week
# - day_of month

# Adding the features which separates the pickup_datetime stamp into hour_of_day, month_of_date, day_of_week, day_of_month, day_of_week_num
# 

# In[ ]:


#Adding the features which separates the pickup_datetime stamp into hour_of_day, month_of_date, day_of_week, day_of_month, day_of_week_num

taxi_mod['hour_of_day']=taxi_mod.pickup_datetime.dt.hour
taxi_mod['month_of_date'] = taxi_mod['pickup_datetime'].dt.month
taxi_mod['day_of_week'] = taxi_mod['pickup_datetime'].dt.weekday_name
taxi_mod['day_of_month'] = taxi_mod['pickup_datetime'].dt.day
taxi_mod['day_of_week_num'] = taxi_mod['pickup_datetime'].dt.dayofweek


# In[ ]:


taxi_mod.head()


# ![image.png](attachment:image.png)

# In[ ]:


plt.figure(figsize=(10,20))
plt.subplot(4,1,1)
sns.countplot(x='day_of_month', data=taxi_mod)
plt.ylabel('number of trips')
plt.subplot(4,1,2)
sns.countplot(x='hour_of_day', data=taxi_mod)
plt.ylabel('number of trips')
plt.subplot(4,1,3)
sns.countplot(x='day_of_week_num', data=taxi_mod)
plt.ylabel('number of trips')
plt.subplot(4,1,4)
sns.countplot(x='month_of_date', data=taxi_mod)
plt.ylabel('number of trips')
plt.show()


# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# From above shown countplots we can conclude that:
# - First 6 days of any given month has highest number of trips
# - Between 18:00 and 20:00 local time has the highest number of pick ups
#     - We can see there is a raise in the number of trips at the start of Business hours  (i.e, from 6am to 9am) which is in par with everyday experience
#     - And a peek in the number of trips can be observed during the end of Business hours (i.e, from 5pm to 8pm)
#     - Post 9pm, there is a decrease in  the number of trips till 5am of next day 
# - Almost all days in a week has the same number of trips(slightly less on Sundays can be sundays)
# - First 6 months of 2016 had the high number of pick-up(i.e from January to June)
# - From July to December the count of number of trips are almost the same

# In[ ]:


taxi_mod1=taxi_mod


# Filtering the duration value greater than 1 hour for trip distance less than 1km
# - non linear data points
# - inconsistent data

# REMOVING THE INCONSISTENT DATA FROM THE ORIGINAL DATA (1048574  - 337 = 1048238)

# In[ ]:


#taxi_mod2=pd.concat([taxi_mod, taxi_mod1]).loc[taxi_mod.index.symmetric_difference(taxi_mod1.index)]
taxi_mod2=taxi_mod1.loc[(taxi_mod1['trip_duration'] >=3600 ) & (taxi_mod1['distance'] <= 1),['trip_duration','distance'] ].reset_index(drop=True)
sns.regplot(taxi_mod2['distance'], taxi_mod2.trip_duration)
taxi_mod2.info()
plt.show()


# ![image.png](attachment:image.png)

# Observation:
# - For the trips which has distance less than 1KM and duration greater than 1Hour are non-linear
# - It is practically not possible to travel less than 1 KM for more than 1 Hour
# - Hence these 337 rows of inconcitent data is removed from dataset

# In[ ]:


taxi_mod3=pd.concat([taxi_mod2, taxi_mod1]).loc[taxi_mod1.index.symmetric_difference(taxi_mod2.index)]
taxi_mod3.info()


# ![image.png](attachment:image.png)

# In[ ]:


taxi_mod4=taxi_mod3.loc[(taxi_mod3['trip_duration'] <= 18000) & (taxi_mod3['distance'] <= 100),['trip_duration','distance'] ].reset_index(drop=True)


# In[ ]:


taxi_mod4.info()


# ![image.png](attachment:image.png)

# # Bi-variate Analysis between:
# 
# - Distance V/S Trip_duration
# - hour_of_day V/S Trip_duration
# - month_of_date V/S Trip_duration
# - day_of_week_num V/S Trip_duration
# - day_of_month V/S Trip_duration
# - hour_of_day V/S Distance
# - month_of_date V/S Distance
# - day_of_week_num V/S Distance
# - day_of_month V/S Distance
# - hour_of_day V/S SPEED(Km/Hr)
# - month_of_date V/S SPEED(Km/Hr)
# - day_of_week_num V/S SPEED(Km/Hr)
# - day_of_month V/S SPEED(Km/Hr)

# In[ ]:


plt.figure(figsize=(20,10))
plt.scatter(taxi_mod4['distance'],taxi_mod4['trip_duration'],s=1, alpha=0.5)
plt.xlabel('Distance in Km/hr')
plt.ylabel('Trip Duation in seconds')
plt.show()


# ![image.png](attachment:image.png)

# In[ ]:


plt.figure(figsize=(20,10))
sns.lmplot(x='distance', y='trip_duration', data=taxi_mod4, aspect=2.5, scatter_kws={'alpha':0.2})
plt.xlabel('Distance in Km/hr')
plt.ylabel('Trip Duation in seconds')
plt.show()


# ![image.png](attachment:image.png)

# Above plot shows that there is a linearity between distance of a trip and the trip druation
# - Trip duration increases linearly as the distance increases which is true and is inline with the real life experience
# - Hence distance of a trip is an important feature for predicting the duration of a trip

# In[ ]:


group1 = taxi_mod3.groupby('hour_of_day').trip_duration.mean()
plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.pointplot(group1.index, group1.values,color="#3fbb3f")
plt.ylabel('trip_duration')

group2 = taxi_mod3.groupby('month_of_date').trip_duration.mean()
plt.subplot(2,2,2)
sns.pointplot(group2.index, group2.values,color="#3fbb3f")
plt.ylabel('trip_duration')


group3 = taxi_mod3.groupby('day_of_week_num').trip_duration.mean()
plt.subplot(2,2,3)
sns.pointplot(group3.index, group3.values,color="#3fbb3f")
plt.ylabel('trip_duration')

group4 = taxi_mod3.groupby('day_of_month').trip_duration.mean()
plt.subplot(2,2,4)
sns.pointplot(group4.index, group4.values,color="#3fbb3f")
plt.ylabel('trip_duration')


plt.show()


# Hour of the day VS Trip duration
# ----------------------------------
# - Average duration of trips is observed to be increasing from 8am to 4pm and decreasing post 4pm
# - Trip duration is the least between 5am and 7am
# 
# Highest variation in Trip duration(800 - 1150 seconds) is found when the average Trip duration is plotted against the time in a day the trip was taken. Hence we can conclude that Duration of a Trip is highly dependant on Trip time of the day which in par with the experience in real life scenario.
# 
# Month of the year VS Trip duration
# ------------------------------------
# - Average trip duration is highest in the month of May
# - Duration is least in the month of July
# - Rest of the month in the yaer had almost the same Trip Duration
# 
# Variation in the Trip duration is found when Trip duration is plotted against month of the year which is between 850 seconds and 1025 seconds. Hence Trip duration travelled is fairly dependent on which month the year the trip was taken.
# 
# Day of the week VS Trip duration
# ----------------------------------
# - Wednesday is found to have highest trip duration compared with all other days in a week
# - Monday has least trip duration
# - Remaining days in a week including saturday and sunday has the same Trip duration
# 
# Least variation in travelled is found when Trip duration is plotted against day of the week which is between 925 seconds and 990 seconds. Hence we can conclude that distance travelled is less dependent on which day in a week a trip was taken.
# 
# Day of the month VS Trip duration
# -----------------------------------
# - 7th of all months in the year 2016 has the highest Trip duration
# - 2nd has least duration
# - Rest of the days has trip duration between 925 seconds to 1025 seconds
# 
# 

# In[ ]:


group5 = taxi_mod3.groupby('hour_of_day').distance.mean()
plt.figure(figsize=(25,15))
plt.subplot(2,2,1)
sns.pointplot(group5.index, group5.values,color="#bb7d3f")
plt.ylabel('distance')

group6 = taxi_mod3.groupby('month_of_date').distance.mean()
plt.subplot(2,2,2)
sns.pointplot(group6.index, group6.values,color="#bb7d3f")
plt.ylabel('distance')


group7 = taxi_mod3.groupby('day_of_week_num').distance.mean()
plt.subplot(2,2,3)
sns.pointplot(group7.index, group7.values,color="#bb7d3f")
plt.ylabel('distance')

group8 = taxi_mod3.groupby('day_of_month').distance.mean()
plt.subplot(2,2,4)
sns.pointplot(group8.index, group8.values,color="#bb7d3f")
plt.ylabel('distance')
plt.show()


plt.show()


# Hour of the day VS Distance Travelled
# ---------------------------------------
# - Average Trip distance traveleld between 5am and 6am is found to be at the peek for nearly 5.5 Kilometer
# - Post 6am the distance travelled is decreasing 
# - From 7pm onwards the distance travelled is found increasing till 5am 
# 
# The Average Distance travelled across all 24 hours in a day is between 3 KM and 5.5 KM, which is significantly higher variation in the distance convered. We can conclude that Distance Travelled is highly dependent on a particular hour in a day in which trip was made.
# 
# Month of the year VS Distance Travelled
# -----------------------------------------
# - Distance travelled in the month of May is highest
# - Distance travelled in the month of September is least
# 
# As the average distance travelled across all 12 months in the year 2016 is between 3.35 KM and 3.525 KM,which is very less variation, we can conclude that Distance travelled is least dependent on the Month of the Year
# 
# Day of the week VS Distance Travelled
# ---------------------------------------
# - On Sunday and Monday(day no 6 and 0) has the highest distance travelled which is close to 3.55 Kilometer
# - Distance travelled on Wednesday is least close to 3.35 Kilometer
# 
# As the average distance travelled across all 7 days in a week is between 3.35 KM and 3.6 KM,which is very less variation, we can conclude that Distance travelled is least dependent on the day of the week
# 
# Day of the month VS Distance Travelled
# ----------------------------------------
# As the average distance travelled across all 31 days in a month is between 3.25 KM and 3.55 KM,which is very less variation, we can conclude that Distance travelled is least dependent on the day of the month

# In[ ]:


group9 = taxi_mod3.groupby('hour_of_day').SPEED.mean()
plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.pointplot(group9.index, group9.values,color="#bb3f3f")
plt.ylabel('SPEED(Km/Hr)')

group10 = taxi_mod3.groupby('month_of_date').SPEED.mean()
plt.subplot(2,2,2)
sns.pointplot(group10.index, group10.values,color="#bb3f3f")
plt.ylabel('SPEED(Km/Hr)')


group11 = taxi_mod3.groupby('day_of_week_num').SPEED.mean()
plt.subplot(2,2,3)
sns.pointplot(group11.index, group11.values,color="#bb3f3f")
plt.ylabel('SPEED(Km/Hr)')

group12 = taxi_mod3.groupby('day_of_month').SPEED.mean()
plt.subplot(2,2,4)
sns.pointplot(group12.index, group12.values,color="#bb3f3f")
plt.ylabel('SPEED(Km/Hr)')
plt.show()


plt.show()


# Hour of the day VS SPEED
# -------------------------
# - During the Business working hours i.e, from 8am to 9pm Speed is almost constant at the lower end which is around 12 Km/Hr. - - - This observation is true and inline with daily experience and lower vehicle speed is observed mostly due to the traffic congestion during these hours of the day.
# - Post 9pm speed tends to increase above 13 Km/Hr due to lower traffic till 5 o'clock in the morning.
# - Speed of the vehicle is found to be at its peak around 5am to 6am
# - Gradual decrease in Speed is observed between 6am and 8 am as it is the onset of business hours
# - Large variation in Speed is observed when Speed is plotted agaist Hour of the day which is between(12Km/Hr and 24Km/Hr).Hence the Average speed of a taxi is highly dependent on which time of the day the trip was made
# 
# Month of the year VS SPEED
# ---------------------------
# - Average speed is found to be the least in the month of November around 14Km/Hr
# - Although not much of variation is observed in the Average speed, we can conclude that the Speed is less dependent on which month of year the trip was made
# 
# Day of the week VS SPEED
# -------------------------
# - Speed is found to be increasing after Friday till Sunday which is agreeable as it is working days of the week
# - Post Sunday .i,e from Monday the average Speed starts to decrease and will be between 13.75Km/Hr and 14.75Km/Hr until Friday
# - As variation in the speed is fairly higher, we can say that Speed is slightly dependent on the Day of the week
# 
# Day of the Month VS SPEED
# --------------------------
# - Average speed is found to be at the peak only on 1st of every month and remains 14Km/Hr and 14.75 Km/Hr on all other days
# - Although the variation the speed is higher, we observe the speed to be between 14Km/Hr and 14.85 on most of the days in a month. Hence we can conclude that Speed is not much dependent on which day og the Month trip was made.

# In[ ]:


taxi_mod.info()


# In[ ]:


taxi_mod9 = taxi_mod1.drop(taxi_mod1[(taxi_mod1.distance < 1)&(taxi_mod1.trip_duration > 3600)].index)
taxi_mod9 = taxi_mod1.drop(taxi_mod1[(taxi_mod3['trip_duration'] >= 18000) | (taxi_mod1['distance'] >= 200)].index)


# In[ ]:


taxi_mod9.info()


# In[ ]:


#taxi_mod9.to_csv('B:/UPX docs/Machine Learning/Project_datasets/Project datasets modified/NYC Taxi Trip/NYC Taxi Trip/Taxi_new.csv')

