#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


df_flights = pd.read_csv('../input/flight-delays/flights.csv')


# In[ ]:


df_flights.head()


# In[ ]:


# Checking the datatypes of each column
df_flights.dtypes


# In[ ]:


# Dropping unnecessary columns
df_flights = df_flights.drop(["DEPARTURE_TIME","SCHEDULED_DEPARTURE","SCHEDULED_ARRIVAL","ARRIVAL_TIME","TAIL_NUMBER","WHEELS_OFF","WHEELS_ON","TAXI_IN","TAXI_OUT","ELAPSED_TIME"],axis=1)


# In[ ]:


import matplotlib.pyplot as plt
# Counting the missing values in each variable
df_flights.isnull().mean().sort_values(ascending=False).plot.bar(figsize=(12,6))
plt.ylabel('Percentage of missing values')
plt.xlabel('Variables')
plt.title('Quantifying missing data')


# Handling null values in the Delay features, i.e when there were no Airline, Security or Weather delays the value was left blank, now filling 0's as values will not bias the data.

# In[ ]:


df_flights['AIRLINE_DELAY'] = df_flights['AIRLINE_DELAY'].fillna(0)
df_flights['AIR_SYSTEM_DELAY'] = df_flights['AIR_SYSTEM_DELAY'].fillna(0)
df_flights['SECURITY_DELAY'] = df_flights['SECURITY_DELAY'].fillna(0)
df_flights['LATE_AIRCRAFT_DELAY'] = df_flights['LATE_AIRCRAFT_DELAY'].fillna(0)
df_flights['WEATHER_DELAY'] = df_flights['WEATHER_DELAY'].fillna(0)


# In[ ]:


# Counting the missing values in each variable
df_flights.isnull().mean().sort_values(ascending=False).plot.bar(figsize=(12,6))
plt.ylabel('Percentage of missing values')
plt.xlabel('Variables')
plt.title('Quantifying missing data')


# Examinig the CANCELLATION_REASON column to fill the NaN with letter 'NC' for our understanding that there was No Cancellation

# In[ ]:


df_flights['CANCELLATION_REASON'].value_counts()


# In[ ]:


df_flights['CANCELLATION_REASON'].value_counts().plot.bar(figsize=(12,6))
plt.ylabel('Number of Reasons')
plt.xlabel('Reasons')
plt.title('Listing the Missing reasons')


# In[ ]:


# Converting NaN labels to NC
df_flights['CANCELLATION_REASON'] = df_flights['CANCELLATION_REASON'].fillna('NC')
# Verifying the change
df_flights['CANCELLATION_REASON'].value_counts()


# In[ ]:


# Plotting the missing values in each variable
df_flights.isnull().mean().sort_values(ascending=False).plot.bar(figsize=(12,6))
plt.ylabel('Percentage of missing values')
plt.xlabel('Variables')
plt.axhline(y=0.02, color='red') #highlight the 2% mark with a red line:
plt.title('Quantifying missing data')


# The reason for missing values in the above 3 variables is due to the varibales DIVERTED or CANCELLED having value 1 (true), removing these null values will remove all the 1(true) records from DIVERTED or CANCELLED variables. We have to filter and ignore Cancelled and Diverted flights while performing calculations on these columns. 

# In[ ]:


# Visualize the variable distribution with histograms
df_flights.hist(bins=30, figsize=(12,12), density=True)
plt.show()


# Most of the numerical variables in the dataset are skewed.

# In[ ]:


# Determine the number of unique categories in each variable:
df_flights.nunique()


# Creating a date column by combining the 'YEAR','MONTH', 'DAY' columns

# In[ ]:


df_flights['DATE'] = pd.to_datetime(df_flights[['YEAR','MONTH', 'DAY']])


# In[ ]:


# Verifying the change
df_flights["DATE"].head()


# Loading the Airlines dataset and merging airline name with df_flights

# In[ ]:


df_airlines = pd.read_csv('../input/flight-delays/airlines.csv')
df_airlines


# In[ ]:


df_flights = df_flights.rename(columns={"AIRLINE":"IATA_CODE"})
df_merge = pd.merge(df_flights,df_airlines,on="IATA_CODE")
df_merge.head()


# Loading the Airports dataset and merging on origin and destination airport to get city, origin and destination state details

# In[ ]:


df_airports = pd.read_csv('../input/flight-delays/airports.csv')
df_airports = df_airports.rename(columns={"IATA_CODE":"CODE"})
df_airports


# In[ ]:


# Merging the origin details
df = df_merge.merge(df_airports[['STATE','AIRPORT','CODE']], how = 'left',
                left_on = 'ORIGIN_AIRPORT', right_on = 'CODE').drop('CODE',axis=1)
df = df.rename(columns={"STATE":"ORIGIN_STATE","AIRPORT":"ORG_AIRPORT"})
df.head()


# In[ ]:


# Merging the destination details
df = df.merge(df_airports[['STATE','AIRPORT','CODE']], how = 'left',
                left_on = 'DESTINATION_AIRPORT', right_on = 'CODE').drop('CODE',axis=1)
df = df.rename(columns={"STATE":"DESTINATION_STATE","AIRPORT":"DES_AIRPORT"})
df.head()


# In[ ]:


df.to_csv("flightdata.csv",index=False)


# In[ ]:


df["YEAR"].count()


# In[ ]:




