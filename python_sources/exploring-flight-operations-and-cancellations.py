#!/usr/bin/env python
# coding: utf-8

# # Flight operations overview #
# 
# On this Notebook I will try to get some conclusions regarding to the flights operation using the dataset provided.
# 
# *Note: This is my first analysis ever. I would really appreciate any suggestion or comment that could help me improve the code/analysis quality. Thanks in advance!*

# ## Setting up ##

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Loading dataframes ##

# In[ ]:


airlines = pd.read_csv('../input/airlines.csv')
airports = pd.read_csv('../input/airports.csv')
flights = pd.read_csv('../input/flights.csv', low_memory=False)


# ##Exploring Data frames ##

# ### Airlines ###

# In[ ]:


airlines.head(2)


# The airlines dataframe provides us the IATA code for each airline. We can use this data to create a dictionary to use it later.

# In[ ]:


airlines_dict = dict(zip(airlines['IATA_CODE'],airlines['AIRLINE']))
airlines_dict


# ###Airports ###

# In[ ]:


airports.head(2)


# In[ ]:


airports.shape[0] # Number of airports


# The airports data frame provides geographical information of 322 airport associated to its IATA code.

# ### Flights ###

# In[ ]:


flights.head(2)


# In[ ]:


rows,cols = flights.shape
print("Number of rows: ", rows)
print("Number of columns: ", cols)


# In[ ]:


flights.info()


# ## Adding some useful columns ##
# 
#  1. I add a column with the date in datetime format to make easier plotting the data.
#  2. I add a columns with the month name
#  3. I add a column with the day of the week name
#  4. I add a column with the name of the airline that operates each flight

# In[ ]:


# 1. Date
flights['DATE'] = pd.to_datetime(flights[['YEAR','MONTH','DAY']], yearfirst=True)

# 2. Month name
month_dict={
    1:  '01- January',
    2:  '02- February',
    3:  '03- March',
    4:  '04- April',
    5:  '05- May',
    6:  '06- June',
    7:  '07- July',
    8:  '08- August',
    9:  '09- September',
    10: '10- October',
    11: '11- November',
    12: '12- December'
}
flights['MONTH_desc'] = flights['MONTH'].apply(lambda m: month_dict[m])

# 3. Day of the week name
dow_dict = {
    1: 'Tuesday',
    2: 'Wednesday',
    3: 'Thursday',
    4: 'Friday',
    5: 'Saturday',
    6: 'Sunday',
    7: 'Monday'
}
flights['DOW_desc'] = flights['DAY_OF_WEEK'].apply(lambda d: dow_dict[d])

# 4. Airline name
flights['AIRLINE_desc'] = flights['AIRLINE'].apply(lambda a: airlines_dict[a])
flights.head()


# In[ ]:


Seems like there were no errrors. Time to clean some data.


# ## Cleaning ##
# 
# Let's clean the dataframe to get the columns we will use for the analysis.
# By now, I will not need the reasons for cancellations or delays.

# In[ ]:


flights.drop(['CANCELLATION_REASON','AIR_SYSTEM_DELAY','SECURITY_DELAY','AIRLINE_DELAY','LATE_AIRCRAFT_DELAY','WEATHER_DELAY'], axis=1, inplace=True)


# For this analysis I don't need the Tail number nor the flight number

# In[ ]:


flights.drop(['FLIGHT_NUMBER','TAIL_NUMBER'], axis=1, inplace=True)


# In[ ]:


flights.head(2)


# I still haven't decided if I will explore details about time, distances and locations. I'm not dropping more columns by now.

# ## Flights Operation ##
# First of all, I want to get a first idea of the scheduled flights on 2015 and how many of them were cancelled.

# In[ ]:


cancelled = flights[flights['CANCELLED'] == 1].count()['CANCELLED']
scheduled = flights.count()['SCHEDULED_DEPARTURE']
operated = scheduled - cancelled
ratio_oper = operated / scheduled * 100
ratio_cancel = 100 - ratio_oper

print("Scheduled flights: ", scheduled)
print("Cancelled flights: ", cancelled)
print("Operated flights: ", operated)
print("\n")
print("Ratio operated flights over scheduled flights: %s" % ratio_oper)
print("Ratio operated flights over scheduled flights: %s" % ratio_cancel)


# 98.45% of scheduled flights were operated.
# Let's plot it!

# In[ ]:


fig = plt.figure(figsize=(10,4));

ax = fig.add_axes([0,0,1,1]);

flights.groupby('DATE').count()['SCHEDULED_DEPARTURE'].plot.line(c='b', label="scheduled");
flights[flights['CANCELLED'] == 0].groupby('DATE').count()['SCHEDULED_DEPARTURE'].plot.line(c='g', label="operated");
flights[flights['CANCELLED'] == 1].groupby('DATE').count()['SCHEDULED_DEPARTURE'].plot.line(c='r', label="cancelled");

ax.legend();


# From the plot we can observe that:
# 
#  1. Total scheduled flights varies depending on the season. There are more flights on Summer months 
#  and we can deduce a little increment of flights during Christmas hollidays.
#  2. On the first half of September there's a remarkable decrease of scheduled flights. Maybe because of the 9/11 psychologic impact? 
#  3. In the end of January and in the begining of February there were a lot of cancellations. During all february there are a large amount of cancelled flights periodically. I firstly tought those cancellations could be caused by bad weather, but it's (almost) constant periodicity make me think about a possibility of Air controllers strike. I will search it later.

# What if I group the scheduled flights by airlines?

# In[ ]:


fig = plt.figure(figsize=(10,15))

ax = fig.add_axes([0,0,1,1])

for airline in airlines['AIRLINE']:
    flights[(flights['AIRLINE_desc'] == airline) & (flights['CANCELLED'] == 0) ].groupby('DATE').count()['SCHEDULED_DEPARTURE'].plot.line(x='DATE', label=airline)

ax.legend();


# Mmmh... Too difficult to read... Let's try a heat map.
# 
# First, I need to create a matrix to visualize it. I create one with airlines on columns, months on rows and the total of scheduled flights as a value.

# In[ ]:


pvt_scheduled_airline_date = flights.pivot_table(index="MONTH_desc",columns="AIRLINE_desc",values="SCHEDULED_DEPARTURE",aggfunc=lambda x: x.count())
pvt_scheduled_airline_date.head()


# In[ ]:


fig = plt.figure(figsize=(10,10))
sns.heatmap(pvt_scheduled_airline_date, linecolor="w", linewidths=1);


# Now it's easier to get an idea about the volume of operated flights by each company.
# 
# Notice that from July, US Airways Inc haven't scheduled any flight, while from the same data American Airlines has increased its flights. That suggests me that American Airlines merged and they are operating as American Airlines Inc.
# 
# We can also see that Southwest Airlines Co. schedules much more flights than other companies. It could be because it has a very large fleet, it operates low range flights or both. Both are indicators of a Low Cost company strategy, I think Southwest Airlines is a low-cost operator, I will look for it later.
# 
# As we can see in the heatmap and the previous plot, Southwest Airlines is the largest Airline in volume of flights. Until July, the second largest company was Delata Air Lines, but from July its flight operations number is quite similar to American Airlines, which reinforces my hipothesis of its merging with US Airways.

# ## Validating hipothesis ##

# ### American Airlines merger with US Airways ###

# *"April 8, 2015: FAA granted American Airlines and US Airways the authority to operate
# as a single carrier. The decision allowed the two airlines to combine work forces,
# websites, and reservations systems, starting the fall of 2015. (See October 20, 2014;
# October 16, 2015.)"* 
# 
# Source: [Federal Aviation Association 1997-2015 Chronology][1] 
# 
#   [1]: https://www.faa.gov/about/history/media/final_1997-2015_chronology.pdf

# ###February cancellations ###

# After some Googling, I am sure that high ratio of flight cancellations occurred between January and February were due to extreme weather. There were at least 2 strong blizzards that caused a lot of flight cancellations.
# 
# According to CBS News, on January 26th there [were more than 6500 flight cancellations in 2 days because of extreme weather][1].
# 
# According to USA Today, on [February 16th there were more than 3000 flights cancelled because of a blizzard][2].
# 
# 
#   [1]: http://www.cbsnews.com/news/blizzard-2015-flight-delays-and-cancellations-pile-up/
#   [2]: http://www.usatoday.com/story/todayinthesky/2015/02/16/flight-cancellations-at-600-and-counting-from-new-storm/23488003/
