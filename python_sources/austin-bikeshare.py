#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn as sk # SkLearn ML library
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# ### Load Packages

# In[ ]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import requests
from datetime import datetime


# ### Load files to pandas dataframes

# In[ ]:


trips = pd.read_csv('../input/austin-bike/austin_bikeshare_trips.csv')
stations = pd.read_csv('../input/austin-bike/austin_bikeshare_stations.csv')


# In[ ]:


trips.head(5)


# In[ ]:


trips.dtypes


# In[ ]:


stations.head()


# ### Ridership totals during different months
# 
# 

# In[ ]:


tripsByMonth = trips.groupby('month').month.count()
tripsByMonth.index = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

ax = sns.barplot(x='index', y='month', data=tripsByMonth.reset_index(), color='red')
ax.figure.set_size_inches(14,8)
sns.set_style(style='white')
ax.axes.set_title('Total Rides in Each Month', fontsize=24)
ax.set_xlabel('Month', size=20)
ax.set_ylabel('Rides', size=20)
ax.tick_params(labelsize=16)


# There seems to be an unusually high number of riders in the month of march. One reason for this is that Austinites are eager to enjoy the sunshine once their relatively week winter comes to an end. They then sit inside enjoying the AC during the heat of summer, and then venture out once more in october once winter is near. Another reason could be that data collection started at the beginning of march on the first year and ended at the end of march on the last year. Essentialy giving us an extra month of data for the month of march

# In[ ]:


tripsByYearMonth = trips
tripsByYearMonth = tripsByYearMonth.groupby(['month','year']).month.count()
tripsByYearMonth


# It definitly seems that we are missing data from certain years. With a quick glance i can see that December contains only 2013, 2014, and 2015 while March contains 2014, 2015, 2016, and 2017. Lets instead look at the same bar plot using data only from 2014 and 2015.

# In[ ]:


tripsFullYears = trips[trips['year'].isin(['2014','2015'])]
tripsByMonth = tripsFullYears.groupby(['month', 'year']).trip_id.count()


ax = sns.barplot(x='month', y='trip_id', hue='year', data=tripsByMonth.reset_index(), color='red')
ax.figure.set_size_inches(14,8)
sns.set_style(style='white')
ax.axes.set_title('Total Rides in Each Month', fontsize=24)
ax.set_xlabel('Month', size=20)
ax.set_ylabel('Rides', size=20)
ax.tick_params(labelsize=16)


# Even filtering on full years there is defiinitly a large jump in riders when winter ends and just before winter begins. And it seemst that ridership has increased from the year 2014 to year 2015 across all months.

# In[ ]:


#Create a binary column for trips that start and end at the same station
def round_trip(row):
    if row['end_station_id'] == row['start_station_id']:
        return 1
    return 0

trips['round_trip'] = trips.apply(lambda row: round_trip(row), axis=1)


# In[ ]:


aggregate = {'trip_id':'count', 'round_trip':'sum'}
roundTripsByMonth = trips.groupby('month').agg(aggregate)
roundTripsByMonth['round_trip_ratio'] = roundTripsByMonth['round_trip'] / roundTripsByMonth['trip_id'] * 100

#Replace float monthes with string months
roundTripsByMonth.index = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

ax = sns.barplot(x='index', y='round_trip_ratio', data=roundTripsByMonth.reset_index(), color='red')
ax.figure.set_size_inches(14,8)
ax.set_ylim(0,20)
ax.axes.set_title('Percent of Rides That Are Return Trips', fontsize=24)
ax.set_xlabel('Month', size=20)
ax.set_ylabel('Percent', size=20)
ax.tick_params(labelsize=16)


# It actually turns out that people are less likely to take return trips during the peak months, october and March. Perhaps the more consistent riders are using the bikes to run errands starting from one station going to the grocery store and then returning to the same station.

# ### Subscriber type ratios per month

# In[ ]:


def short_term_subscriber(row):
    if (
            row['subscriber_type'].lower().find('walk') > -1 or
            row['subscriber_type'].lower().find('weekend') > -1 or
            row['subscriber_type'].lower().find('24') > -1 or
            row['subscriber_type'].lower().find('single') > -1
        ):
        return 1
    return 0

trips['subscriber_type'] = trips['subscriber_type'].replace(np.nan, '', regex=True)
trips['short_term_membership'] = trips.apply(lambda row: short_term_subscriber(row), axis=1)


# In[ ]:


trips.head()


# In[ ]:


aggregate = {'trip_id': 'count', 'short_term_membership': 'sum'}
membershipTypeTripsPerMonth = trips.groupby('month').agg(aggregate)
membershipTypeTripsPerMonth['short_term_membership_percentage'] = membershipTypeTripsPerMonth['short_term_membership'] / membershipTypeTripsPerMonth['trip_id'] * 100 

membershipTypeTripsPerMonth.index = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

ax = sns.barplot(x='index', y='short_term_membership_percentage', data=membershipTypeTripsPerMonth.reset_index(), color='red')
ax.figure.set_size_inches(14,8)
ax.axes.set_title('Percent of Rides Using a Short Term Membership', fontsize=24)
ax.set_xlabel('Months', fontsize=20)
ax.set_ylabel('Percentage', fontsize=20)
ax.tick_params(labelsize=16)


# It seems that people who ride during the peak months of March and October are more frequently using a short term membership. While the riders in the heat of summer, August, or the relatively pleasant 'winter', average high of 60 F, are more frequently using a long term membership..
# 

# ## Can we predict daily ridership totals based off of the weather?

# ### First we need to clean up the weather data

# In[ ]:


weather = pd.read_csv('../input/austin-weather/austin_weather.csv')
trips = pd.read_csv('../input/austin-bike/austin_bikeshare_trips.csv')
stations = pd.read_csv('../input/austin-bike/austin_bikeshare_stations.csv')


# In[ ]:


weather.head()


# In[ ]:


weather.Events.unique()


# Create four columns that will be populated with boolean values for the four different weather events

# In[ ]:


weather['Rain'] = np.where(weather['Events'].str.contains('Rain'), 1, 0)
weather['Thunderstorm'] = np.where(weather['Events'].str.contains('Thunderstorm'), 1, 0)
weather['Fog'] = np.where(weather['Events'].str.contains('Fog'), 1, 0)
weather['Snow'] = np.where(weather['Events'].str.contains('Snow'), 1, 0)

weather = weather.drop('Events', 1)


# Convert traces of rain to .001 inches of rain to recognize that there was perciptation but it was a value less than what could be measured.

# In[ ]:


weather['PrecipitationSumInches'] = np.where(weather['PrecipitationSumInches'] == 'T', 0.001, weather['PrecipitationSumInches'])


# I will also assign a day of the week column to help control for weekends.

# In[ ]:


weather['Date'] = pd.to_datetime(weather['Date'])
weather['DayOfWeek'] = weather['Date'].dt.weekday


# In[ ]:


weather = weather[(weather['Date'] >= '2014-01-01') & (weather['Date'] <= '2015-12-31')]


# In[ ]:


weather = weather.set_index('Date', drop=True)


# In[ ]:


weather.head()


# In[ ]:


weather = weather.convert_objects(convert_numeric=True)
weather = weather.fillna(weather.mean())


# In[ ]:


weather.dtypes


# ### Next lets clean group the ridership data by dates and create our targets

# In[ ]:


trips.head()


# In[ ]:


trips = trips[trips['year'].isin(['2014','2015'])]
trips['Date'] = pd.to_datetime(trips['start_time']).dt.date
trips = trips.groupby(['Date']).trip_id.count()


# In[ ]:


trips.name = 'TripCount'


# In[ ]:


trips.shape


# ### Inner join ridership and weather data together
# This will remove the dates which i dont have ridership data for

# In[ ]:


tripWeather = trips.to_frame().join(weather, lsuffix='Date', rsuffix='Date', how='inner')


# In[ ]:





# ## Prepare data for learning

# In[ ]:


rideCounts = tripWeather['TripCount']
rideWeather = tripWeather.drop('TripCount', axis=1)


# In[ ]:


rideWeather_train, rideWeather_test, rideCounts_train, rideCounts_test = train_test_split( 
    rideWeather, rideCounts, test_size = .3, random_state = 13, shuffle=True)


# In[ ]:


rideWeather_train.shape


# In[ ]:


rideCounts_train.shape


# In[ ]:


rideWeather_test.shape


# In[ ]:


rideCounts_test.shape


# ## Lets Learn!

# In[ ]:


reg = linear_model.Ridge (alpha = .5)


# In[ ]:


reg.fit(rideWeather_train, rideCounts_train)


# In[ ]:


ridgeScore = reg.score(rideWeather_test, rideCounts_test)
print(ridgeScore)


# Wow this did terribly lets graph observed vs predicted to see visually how bad it looks.

# In[ ]:


rideCountsPredictions = reg.predict(rideWeather_test)
rideCountsActual = rideCounts_test.as_matrix()


# In[ ]:


ax = sns.regplot(x=rideCountsActual, y=rideCountsPredictions)
ax.figure.set_size_inches(10,6)
ax.axes.set_title('Predictions Vs. Actual', fontsize=24)
ax.set_xlabel('Actual', fontsize=20)
ax.set_ylabel('Predictions', fontsize=20)
ax.tick_params(labelsize=16)


# This does not look as bad as i expected. Perhaps whith some better feature selection i might be able to find a better model.

# ### Lets try some more feature engineering
# 
# Lets take another look at what we are working with

# In[ ]:


weather.head(10)


# In[ ]:


#Calculate correleation matrix
correlation = tripWeather.corr()

# plot the heatmap
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap( correlation, ax=ax)


# Lets first change day of the week to a boolean for it it is or is not a weekend
# 
# 0 = Monday  
# 1 = Tuesday  
# ....  
# 5 = Saturday  
# 6 = Sunday  
# 
# Swap to:  
# Monday, ... Friday = 0  
# Saturday, Sunday = 1  

# In[ ]:


def is_weekend(row):
    if row >= 5:
        return 1
    return 0

weather['Weekend'] = weather['DayOfWeek'].apply(is_weekend)
weather = weather.drop('DayOfWeek', axis=1)


# In[ ]:


weather.head()


# Temperatures: I am only going to retian the high and the low temperature since people rarely bike when the temperature is at its lowest ( 5AM ). I would think that the temperature during the hours of 8AM-11AM and 6PM-10PM may closely relate to the average temperature. While, the temperature from 11AM-6PM is best described by the high temperature.

# In[ ]:


weather = weather.drop('TempLowF', axis=1)


# Dew point: All three dewpoints, high, low and average, are all similarly correlated with the trip count. I am only going to retain the one that is most closely correlated to the trip count which is the average dew point.

# In[ ]:


weather = weather.drop(['DewPointHighF', 'DewPointLowF'], axis=1)


# Humidity: I am also going to retain only the avearge humidity for similar reasons

# In[ ]:


weather = weather.drop(['HumidityHighPercent', 'HumidityLowPercent'], axis=1)


# SeaLevelPreassure: Typically large changes in barometric pressure are related to severe weather. I am going to replace the sealevel values with a column that contains the difference between the high and low.

# In[ ]:


weather['PressureChange'] = weather['SeaLevelPressureHighInches'] - weather['SeaLevelPressureLowInches']

weather = weather.drop(['SeaLevelPressureHighInches', 'SeaLevelPressureAvgInches', 'SeaLevelPressureLowInches'], axis=1)


# Visibility: I am only going to retain the average visibility

# In[ ]:


weather = weather.drop(['VisibilityHighMiles', 'VisibilityLowMiles'], axis=1)


# Wind: I am only going to retain the average wind value

# In[ ]:


weather = weather.drop(['WindHighMPH', 'WindGustMPH'], axis=1)


# In[ ]:


weather.head()


# ### Lets rejoin the trip counts to the weather data and prepare to learn

# In[ ]:


tripWeather = trips.to_frame().join(weather, lsuffix='Date', rsuffix='Date', how='inner')


# In[ ]:


rideCounts = tripWeather['TripCount']
rideWeather = tripWeather.drop('TripCount', axis=1)

rideWeather_train, rideWeather_test, rideCounts_train, rideCounts_test = train_test_split( 
    rideWeather, rideCounts, test_size = .3, random_state = 13, shuffle=True)


# In[ ]:


print(rideWeather_train.shape)
print(rideWeather_test.shape)
print(rideCounts_train.shape)
print(rideCounts_test.shape)


# ### Lets learn Again!

# In[ ]:


reg = linear_model.Ridge (alpha = .5)


# In[ ]:


reg.fit(rideWeather_train, rideCounts_train)


# In[ ]:


ridgeScore = reg.score(rideWeather_test, rideCounts_test)
print(ridgeScore)


# Wow i did even worse.  
# Lets take a look at what this looks like when graphed

# In[ ]:


rideCountsPredictions = reg.predict(rideWeather_test)
rideCountsActual = rideCounts_test.as_matrix()


# In[ ]:


ax = sns.regplot(x=rideCountsActual, y=rideCountsPredictions)
ax.figure.set_size_inches(10,6)
ax.axes.set_title('Predictions Vs. Actual', fontsize=24)
ax.set_xlabel('Actual', fontsize=20)
ax.set_ylabel('Predictions', fontsize=20)
ax.tick_params(labelsize=16)


# ### Perhaps lets try a different learning algorithims
# 
# Lasso regression should work better if a small number of variables are able to more accurately predict the total rider count.

# In[ ]:


tripWeather = trips.to_frame().join(weather, lsuffix='Date', rsuffix='Date', how='inner')

rideCounts = tripWeather['TripCount']
rideWeather = tripWeather.drop('TripCount', axis=1)

rideWeather_train, rideWeather_test, rideCounts_train, rideCounts_test = train_test_split( 
    rideWeather, rideCounts, test_size = .3, random_state = 13, shuffle=True)

print(rideWeather_train.shape)
print(rideWeather_test.shape)
print(rideCounts_train.shape)
print(rideCounts_test.shape)


# In[ ]:


#initialize the model
reg = linear_model.Lasso(alpha=0.1)

#fit the training data to the model
reg.fit(rideWeather_train, rideCounts_train)

#find the R^2 score for the results
LassoScore = reg.score(rideWeather_test, rideCounts_test)
print(LassoScore)


# In[ ]:


rideCountsPredictions = reg.predict(rideWeather_test)
rideCountsActual = rideCounts_test.as_matrix()

ax = sns.regplot(x=rideCountsActual, y=rideCountsPredictions)
ax.figure.set_size_inches(10,6)
ax.axes.set_title('Predictions Vs. Actual', fontsize=24)
ax.set_xlabel('Actual', fontsize=20)
ax.set_ylabel('Predictions', fontsize=20)
ax.tick_params(labelsize=16)


# ### Outliers  
# lets take a look at what dates are the outliers. It looks like the core of the data is folowing a trend line of y=x. However, there are a significant amount of days that are extreme outliers, total rides > 1500

# In[ ]:


outlierTrips = trips[trips > 1500]
outlierTrips


# Other than Valentines day 2014 all of these days are from march and october. It actually turns out that the two periods in march correspond to UT Austins spring break. I will probably create another binary variable called SpringBreak that will equal 1 when school is on break. The days in october all correspond to weekends and they are not extreme outliers like march so they should already be captured by the weekend binary variable.

# ### Create binary variable for spring break woo

# In[ ]:


weather.head()


# In[ ]:


def spring_break_woo(date):
    if (date >= datetime(2015, 3, 14)) & (date <= datetime(2015, 3, 23)):
        return 1
    if (date >= datetime(2014, 3, 8)) & (date <= datetime(2014, 3, 17)):
        return 1
    return 0

weather['Date'] = weather.index
weather['SpringBreak'] = weather['Date'].apply(spring_break_woo)
weather = weather.drop('Date', axis=1)


# Quickly lets check to see if this made any improvements

# In[ ]:


tripWeather = trips.to_frame().join(weather, lsuffix='Date', rsuffix='Date', how='inner')

rideCounts = tripWeather['TripCount']
rideWeather = tripWeather.drop('TripCount', axis=1)

rideWeather_train, rideWeather_test, rideCounts_train, rideCounts_test = train_test_split( 
    rideWeather, rideCounts, test_size = .3, random_state = 13, shuffle=True)

print(rideWeather_train.shape)
print(rideWeather_test.shape)
print(rideCounts_train.shape)
print(rideCounts_test.shape)


# In[ ]:


#initialize the model
reg = linear_model.Lasso(alpha=0.1)

#fit the training data to the model
reg.fit(rideWeather_train, rideCounts_train)

#find the R^2 score for the results
LassoScore = reg.score(rideWeather_test, rideCounts_test)
print(LassoScore)


# In[ ]:


rideCountsPredictions = reg.predict(rideWeather_test)
rideCountsActual = rideCounts_test.as_matrix()

ax = sns.regplot(x=rideCountsActual, y=rideCountsPredictions)
ax.figure.set_size_inches(10,6)
ax.axes.set_title('Predictions Vs. Actual', fontsize=24)
ax.set_xlabel('Actual', fontsize=20)
ax.set_ylabel('Predictions', fontsize=20)
ax.tick_params(labelsize=16)


# ### WOW
# 
# what an improvement

# Lets also check ridge regression

# In[ ]:


reg = linear_model.Ridge(alpha = .5)

reg.fit(rideWeather_train, rideCounts_train)

ridgeScore = reg.score(rideWeather_test, rideCounts_test)
print(ridgeScore)


# In[ ]:


rideCountsPredictions = reg.predict(rideWeather_test)
rideCountsActual = rideCounts_test.as_matrix()

ax = sns.regplot(x=rideCountsActual, y=rideCountsPredictions)
ax.figure.set_size_inches(10,6)
ax.axes.set_title('Predictions Vs. Actual', fontsize=24)
ax.set_xlabel('Actual', fontsize=20)
ax.set_ylabel('Predictions', fontsize=20)
ax.tick_params(labelsize=16)


# The two graphs look nearly identical

# ### Future Work
# 
# Weather data along with basic binary variables derived from the date are able to explain approximately .6 of the variation in bike share data which to me seems reasonable. I could likely increase this value by .1-.2 by improving my feature engineering however, i think i will be limited by the data that i have. The other .2-.3 of varation would likely require utilizing other data sets. For example traffic data and austin gas prices could have an effect on a persons willingness to choose a car vs a bike share trip. Also, any changes in bike share prices or bike share advertising will influence a persons demand for using a bike share.
# 

# In[ ]:




