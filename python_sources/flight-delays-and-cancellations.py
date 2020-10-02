#!/usr/bin/env python
# coding: utf-8

# In[329]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy as sp
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
# Any results you write to the current directory are saved as output.


# From the dataset, we want to see which factors affect the flight delays and cancellations, especially whethere there are particular airports where delays and cancellations happen more frequently. 

# In[330]:


airlines = pd.read_csv("../input/airlines.csv")
airports = pd.read_csv("../input/airports.csv", nrows = 1000)
flights = pd.read_csv("../input/flights.csv", dtype={'DESTINATION_AIRPORT': object, 'ORIGIN_AIRPORT': object})


# First, clean up the data. Here, we focus on the airline, day of the weeks, and the delayed arrival time. 

# In[331]:


flights_clean = flights.dropna(subset=['DAY_OF_WEEK', 'ARRIVAL_DELAY', 'CANCELLED', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT'])
flights_delay = flights_clean[flights_clean['ARRIVAL_DELAY'] > 0]
keep_labels = ['DAY_OF_WEEK', 'AIRLINE', 'ARRIVAL_DELAY', 'CANCELLED', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT']
reason_labels = ['AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY']
flights_delay = flights_delay[keep_labels]


# In[332]:


get_ipython().run_line_magic('matplotlib', 'inline')
mean_delay = pd.DataFrame()
mean = flights_delay.groupby('AIRLINE', as_index = False)['ARRIVAL_DELAY'].mean()
mean.plot.bar(x = 'AIRLINE', title = 'Average Arrival Delay')
plt.show()


# In[333]:


highest_time = mean['ARRIVAL_DELAY'].max()
def airline_name_search(code):
    return  airlines[airlines['IATA_CODE'] == code].iloc[0]['AIRLINE']
highest_time_code = mean[mean['ARRIVAL_DELAY'] == highest_time].iloc[0]['AIRLINE']
highest_time_airline = airline_name_search(highest_time_code)
print('The airline with the highest average delay time is ' + highest_time_airline)

xticks = [x*20 for x in range(10)]
delay_plot = flights_delay[flights_delay['AIRLINE'] == highest_time_code]['ARRIVAL_DELAY'].plot(kind = 'hist', bins = xticks, xticks = xticks, xlim = (0, 200), title = ("Frequency of delay arrival time of " + highest_time_airline))


# In[334]:


lowest_time = mean['ARRIVAL_DELAY'].min()
lowest_time_code = mean[mean['ARRIVAL_DELAY'] == lowest_time].iloc[0]['AIRLINE']
lowest_time_airline = airline_name_search(lowest_time_code)
print('The airline with the lowest average delay time is ' + lowest_time_airline)

xticks = [x*20 for x in range(10)]
delay_plot = flights_delay[flights_delay['AIRLINE'] == lowest_time_code]['ARRIVAL_DELAY'].plot(kind = 'hist', bins = xticks, xticks = xticks, xlim = (0, 200), title = ("Frequency of delay arrival time of " + lowest_time_airline))


# We can see that Southwest Airlines has the highest average delay time (not average delay frequency). On the other hand, American Airlines has the lowest average delay time. Now, let's inspect which airlines have the highest and lowest average delay frequency out of all flights taken off. 

# In[335]:


delay_count = flights_delay.groupby('AIRLINE', as_index = False).count()
total_count = flights.groupby('AIRLINE', as_index = False)['ARRIVAL_DELAY'].count()
delay_frequency = pd.DataFrame()
delay_frequency['AIRLINE'] = delay_count['AIRLINE']
delay_frequency['FREQUENCY'] = (delay_count['ARRIVAL_DELAY']/total_count['ARRIVAL_DELAY']*100).astype(float)
delay_frequency.plot.bar(x = 'AIRLINE', title = 'Frequency of delayed flights')


# In[ ]:


max_frequency = delay_frequency['FREQUENCY'].max()
min_frequency = delay_frequency['FREQUENCY'].min()
max_frequency_code = delay_frequency[delay_frequency['FREQUENCY'] == max_frequency].iloc[0]['AIRLINE']
max_frequency_airline = airline_name_search(max_frequency_code)
print('The airline with the highest delay frequency is ' + max_frequency_airline)

min_frequency_code = delay_frequency[delay_frequency['FREQUENCY'] == min_frequency].iloc[0]['AIRLINE']
min_frequency_airline = airline_name_search(min_frequency_code)
print('The airline with the lowest delay frequency is ' + min_frequency_airline)


# In[ ]:




