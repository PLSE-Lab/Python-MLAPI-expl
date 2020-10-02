#!/usr/bin/env python
# coding: utf-8

# # Analysis: Airline delays in the U.S.

# In[ ]:


# data processing
import pandas as pd


# In[ ]:


# Load file (this machine can't handle more)
df = pd.read_csv("/kaggle/input/airline-delay-and-cancellation-data-2009-2018/2018.csv")


# In[ ]:


# Let's get familiar with the dataset
df.info()


# In[ ]:


# 7.2M records and 28 columns
# We have (technical) data on airlines, airport, flight number, etc
# Pretty much all other data is time-related (in minutes)


# In[ ]:


# Set to see all columns
pd.set_option('display.max_columns', None)


# In[ ]:


df.head()


# In[ ]:


# Check unique values in OP_CARRIER (airline) column
df.OP_CARRIER.unique()


# In[ ]:


# Renaming airline codes to company names
# Source: https://en.wikipedia.org/wiki/List_of_airlines_of_the_United_States

df['OP_CARRIER'].replace({
    'UA':'United Airlines',
    'AS':'Alaska Airlines',
    '9E':'Endeavor Air',
    'B6':'JetBlue Airways',
    'EV':'ExpressJet',
    'F9':'Frontier Airlines',
    'G4':'Allegiant Air',
    'HA':'Hawaiian Airlines',
    'MQ':'Envoy Air',
    'NK':'Spirit Airlines',
    'OH':'PSA Airlines',
    'OO':'SkyWest Airlines',
    'VX':'Virgin America',
    'WN':'Southwest Airlines',
    'YV':'Mesa Airline',
    'YX':'Republic Airways',
    'AA':'American Airlines',
    'DL':'Delta Airlines'
},inplace=True)


# In[ ]:


# Quality check
df.OP_CARRIER.unique()


# # Canceled flights exploration

# In[ ]:


# Total number of canceled flights
df.CANCELLED.sum()


# In[ ]:


# Let's explore column CANCELLED
df.CANCELLED.unique()


# In[ ]:


# From above we see it's binary: 0 or 1, let's see how it looks like
canceled = df[(df['CANCELLED'] > 0)]


# In[ ]:


canceled.head(3)


# # Conclusion
# - Canceled flights are not delayed flights
# - If canceled, the flight didn't happen, and values are NaN
# - We can filter out Canceled Flights for out analysis
# - DEP_DELAY Actual Departure Time
# - ARR_DELAY Total Delay on Arrival in minutes
# - If both of these numbers are negative =>> there was no delay

# In[ ]:


# OPTIONAL: Leaving only non-canceled flights
# df = df[(df['CANCELLED'] == 0)]


# # Departure delay and Arrival delay exploration

# In[ ]:


# Departure delay data (in minutes)
df.DEP_DELAY.head()


# In[ ]:


# Arrival delay data (in minutes)
df.ARR_DELAY.head()


# - If a number is positive = flight delayed
# - Since that we're exploring only delayed flights, non-delayed should be disregarded
# - The danger of keeping those is if we summarise for plotting for example, we'll get false data
# - Sum of Delayed minutes will be less because of the negative numbers that will subtract the real delays
# - BUT
# - Let's first define what a delayed flight is:
# - A Delayed flight is a flight that arrives late at its destination
# - Flight can be delayed on departure but still, arrive on time = not a delayed flight

# In[ ]:


# To do this analysis right, let's filter all negative numbers in ARR_DELAY column
# Number of delayed flights 
df[df.ARR_DELAY > 0 ].count()


# In[ ]:


# Filter out non-delayed flights < 0 DEP_DELAY
df = df[(df['ARR_DELAY'] > 0)]


# In[ ]:


# Minutes to hours 
df['ARR_DELAY'] = df['ARR_DELAY'] / 60

# Minutes to hours 
df['DEP_DELAY'] = df['DEP_DELAY'] / 60


# In[ ]:


# Down from 7.2 to 2.5 million (relevant) records
df.info()


# # DateTime data manipulation

# In[ ]:


# Check if FL_DATE is DateTime type
type(df['FL_DATE'])


# In[ ]:


# Convert string to DateTime
pd.to_datetime(df.FL_DATE)


# In[ ]:


# Month variable
df['FL_DATE_month'] = pd.to_datetime(df['FL_DATE']).dt.month
# Weekday variable
df['FL_DATE_weekday'] = pd.to_datetime(df['FL_DATE']).dt.weekday_name


# # Data Visualisation

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # Arrival and departure delays by month of the year

# In[ ]:


# Arrival and departure delays by month of the year
plt.figure(figsize=(25, 12)).subplots_adjust(hspace = 0.5)

plt.subplot(2, 2 ,1)
df.groupby('FL_DATE_month').ARR_DELAY.sum().plot.bar().set_title('ARRIVAL delays by month')
plt.title('ARRIVAL delays by month', fontsize=16)
plt.ylabel('Hours', fontsize=14)
plt.xlabel('Month of the year', fontsize=14)

plt.subplot(2, 2 ,2)
df.groupby('FL_DATE_month').DEP_DELAY.sum().plot.bar()
plt.title('DEPARTURE delays by month', fontsize=16)
plt.ylabel('Hours', fontsize=14)
plt.xlabel('Month of the year', fontsize=14)

plt.show()


# # Delays by airlines

# In[ ]:


# Delays by airlines
plt.figure(figsize=(20, 6))
df.groupby('OP_CARRIER').ARR_DELAY.sum().sort_values(ascending=False).plot.bar()
plt.title('Delays by AIRLINES', fontsize=16)
plt.xlabel('Airline', fontsize=14)
plt.ylabel('Hours', fontsize=14)
plt.show()


# # Delays by City

# In[ ]:


# Delays by City
city_by_delay = df.groupby('ORIGIN').ARR_DELAY.sum().sort_values(ascending=False)
plt.figure(figsize=(20, 6))
city_by_delay[:15].plot.bar()
plt.title('Delays by City', fontsize=16)
plt.xlabel('City', fontsize=14)
plt.ylabel('Hours', fontsize=14)
plt.show()

