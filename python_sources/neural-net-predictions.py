#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# # Introduction
# The purpose of this kernel is to see how the neural network from the [first project](https://github.com/udacity/deep-learning/tree/master/first-neural-network) in the [Deep Learning Nanodegree](https://www.udacity.com/course/deep-learning-nanodegree-foundation--nd101) performs.
# 
# The purpose of that neural network was also to predict bike-sharing demand, but for the [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset).
# 
# # Data Preparation
# ## Importing the Trips Data

# In[2]:


trips_df = pd.read_csv("../input/austin-bike/austin_bikeshare_trips.csv")
stations_df = pd.read_csv("../input/austin-bike/austin_bikeshare_stations.csv")
weather_df = pd.read_csv("../input/austin-weather/austin_weather.csv")


# In[3]:


weather_df.columns


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


trips_df.head()


# In[ ]:


stations_df.head()


# ## Importing the Weather Data
# Next, we'd like to reshape this data so that it looks more like the UCI dataset.  
# The head() of that data looks like:
# ![UCI Bike Sharing Dataset](http://i.imgur.com/qtVLSZe.png)

# In[ ]:


trips_df.start_time.min(), trips_df.start_time.max()


# Notably, this will include weaving in historical weather data, since that is not included in the Austing Bikeshare dataset.
# 
# We have some choices on which weather datasets to use, but for ease of downloading, and instructional value, let's work with the data that WeatherUnderground provides for the [Austin KATT](https://www.wunderground.com/history/airport/KATT/) station.

# In[ ]:


weather_df.head()


# Unfortunately, we're not going to have the temperature resolution from the WeatherUnderground dataset (daily) that we have from the UCI Bikesharing Dataset (hourly), but hopefully it should still be useful to us.

# ## Preparing the trips data
# Let's see if we can get an idea of how the trip are distributed

# In[ ]:


trips_df.start_time.head()


# In[ ]:


type(trips_df.start_time[0])


# To better make use of this data, we'll want to convert it into a Python `datetime` object.

# In[ ]:


import datetime


# In[ ]:


def toDatetime(s):
    return datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


# In[ ]:


trips_df["start_time"] = trips_df.start_time.apply(toDatetime, input)


# In[ ]:


trips_df.start_time.head()


# Now, let's add a column that will allow us to easily sum the trips by the hour during which they started.

# In[ ]:


trips_df["start_hour"] = trips_df.start_time.apply(lambda x: datetime.datetime(x.year, x.month, x.day, x.hour))


# In[ ]:


trips_df.start_hour.head()


# ### Adding Zeroes
# We'll want to plot a distribution of the trip start times and frequencies, but we're missing some hours in our dataset (those hours where no one checked out a bike).
# 
# So, let's create a `date_range` to allow us to reindex the `start_hour` Series as a separate Series.

# In[ ]:


all_trip_hours = pd.date_range(start='2013-12-21', end='2017-07-31 23:00:00', freq='H')


# In[ ]:


all_trip_hours[:5]


# In[ ]:


trips_starts_grouped = trips_df.start_hour.value_counts()


# In[ ]:


trips_starts_grouped.sort_index(inplace=True)


# In[ ]:


trips_starts_grouped.head()


# In[ ]:


trips_starts_grouped = trips_starts_grouped.reindex(all_trip_hours, fill_value=0)


# In[ ]:


trips_starts_grouped[:24*7].plot(figsize = (15, 10), ylim=(0,120))


# The above is Christmas week, so perhaps that's not very representative, but it's enough to at least show that there is a cyclical nature to the data.
# 
# What would some week in March look like?

# In[ ]:


offset = 24*30*3
trips_starts_grouped[offset : offset + 24*7].plot(figsize = (15, 10), ylim=(0,120))


# That makes a lot more sense, but this is still a holiday week.  Let's check out a week in late September.

# In[ ]:


offset = 24*(30*9 - 3) 
trips_starts_grouped[offset : offset + 24*7].plot(figsize = (15, 10), ylim=(0,120))


# Let's assume that the above is a pretty typical week for ridership.  
# It looks like there's some after-midnight rentals, and typically higher weekend ridership than weekday ridership.  
# ### Cleaning the Data
# For the purposes of this analysis, we're after a dataset that approximates the UCI dataset:
# ![UCI Bike Sharing Dataset](http://i.imgur.com/qtVLSZe.png)

# ### Choosing Variables
# Accordingly, from our dataset, let's define the following **dummy variables**:
# * season (implicit)
# * month
# * hour
# * day of the week (implicit)
# * which weather events occurred
# 
# And the following **quantitative variables**:
# * temperature
# * humidity
# * windspeed

# ### Creating Seasons
# First, let's define a function that will allow us to map `datetime` objects to seasons, so that we can add that as a column to our rides dataframe.

# In[ ]:


def timestampToSeason(ts):
    """
    Given a pandas Timestamp object, return which 'season' the date is in.
    
    Arbitrarily, to match the UCI dataset:
    1 -> spring
    2 -> summer
    3 -> fall
    4 -> winter
    
    returns:
    integer
    """
    dt = ts.to_pydatetime().date()
    
    # Our only dates in 2013 are from Dec 21 onward
    year = dt.year
    if year == 2013:
        return 4
    else:
        # Check out the wonderful [wikipedia Seasons page]
        # (https://en.wikipedia.org/wiki/Season)
        eq_sol_by_year = {2014:
                          [datetime.date(2014, 3, 20),
                           datetime.date(2014, 6, 21),
                           datetime.date(2014, 9, 23),
                           datetime.date(2014, 12, 21)],
                         2015:
                          [datetime.date(2015, 3, 20),
                           datetime.date(2015, 6, 21),
                           datetime.date(2015, 9, 23),
                           datetime.date(2015, 12, 22)],
                         2016:
                          [datetime.date(2016, 3, 20),
                           datetime.date(2016, 6, 20),
                           datetime.date(2016, 9, 22),
                           datetime.date(2016, 12, 21)],
                         2017:
                          [datetime.date(2017, 3, 20),
                           datetime.date(2017, 6, 21),
                           datetime.date(2017, 9, 22),
                           datetime.date(2017, 12, 21)]}
        
        if dt < eq_sol_by_year[year][0] or dt >= eq_sol_by_year[year][3]:
            return 4
        elif dt < eq_sol_by_year[year][1]:
            return 1
        elif dt < eq_sol_by_year[year][2]:
            return 2
        else: # dt < eq_sol_by_year[year][3]
            return 3


# In[ ]:


print(trips_df.start_time[0])
timestampToSeason(trips_df.start_time[0])


# In[ ]:


trips_df["season"] = trips_df.start_time.apply(timestampToSeason)


# In[ ]:


trips_df.head()


# ### Creating Weekdays
# Next, let's use a similar approach from above to add a weekday column to our data.

# In[ ]:


def timestampToWeekday(ts):
    """
    Given a pandas Timestamp object, return which weekday that corresponds to,
    according to the python datetime library.
    
    https://docs.python.org/3.0/library/datetime.html
        
    returns:
    integer
    """
    return ts.to_pydatetime().weekday()


# In[ ]:


trips_df["weekday"] = trips_df.start_time.apply(timestampToWeekday)


# In[ ]:


trips_df.head()


# ### Creating Categorical Data from Weather Events

# In[ ]:


weather_df.Events.unique()


# In[ ]:


# major props to https://datascience.stackexchange.com/a/14851
cleaned = weather_df.Events.str.split(' , ', expand = True).stack()


# In[ ]:


cleaned.tail(15)


# In[ ]:


res = pd.get_dummies(cleaned).groupby(level=0).sum()


# In[ ]:


res.columns


# In[ ]:


del res[' ']


# In[ ]:


res.tail(15)


# In[ ]:


res1 = weather_df.join(res)


# In[ ]:


weather_df.


# In[ ]:


weather_df.head()


# In[ ]:


weather_df.iloc[1]


# ### Dropping columns
# To clean up our 

# ### Normalizing Quantitative Data
# Now, our last data preparation step will be to normalize our quantitative data [from above](#Choosing-Variables).
# 
# We'll do what the [UCI bikesharing dataset](https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset) did here.
# 
# That is:  
# * For temperature, we'll subtract t_min (19) from each temp, then divide by (t_max - t_min), where t_max here is 107.  
#   * We'll use `Temp-high-f` and `Temp-low-f`.
# * For humidity, we'll divide by 100, the max possible and in our dataset.
#   * We'll use `Humidity-high-percent` and `Humidity-low-percent`.
# * For wind speed, we'll divide by the max in our dataset (29).
#   * We'll use `Wind-high-mph` and `Wind-avg-mph`.

# In[ ]:




