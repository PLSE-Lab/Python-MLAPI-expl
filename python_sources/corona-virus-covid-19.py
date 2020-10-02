#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


# ### Load Data

# In[ ]:


# load data
def load_data(data, headers):
    df = pd.read_csv(data, names=headers, header=0)
    return df

headers = ['ID', 'ObservationDate', 'State', 'Country', 'LastUpdate', 'Confirmed', 'Deaths', 'Recovered']
df_corona = load_data(data='/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv', headers=headers)
df_corona.head() # First five rows in the dataset


# ### Preprocessing

# In[ ]:


df_corona.shape


# In[ ]:


# Summary of the data
df_corona.describe()


# In[ ]:


# Check for null values
df_corona.isnull().sum()


# In[ ]:


# Checking the first five rows of state column with null values
df_corona[df_corona['State'].isnull()].head()


# In[ ]:


# We could replace the states that were not there with the country
df_corona['State'].fillna(df_corona['Country'], inplace=True)


# In[ ]:


# Checking if they have been filled with Nigeria as a test
df_corona[df_corona['Country'] == 'Nigeria']


# In[ ]:


# Checking the data types of the Columns
df_corona.info()


# In[ ]:


# ObservationDate and LastUpdate should be datetime, State and Country as strings, Confirmed, Deaths and Recovered as integers
# Converting ObservationDate and LastUpdate columns to datetime
df_corona[['ObservationDate', 'LastUpdate']] = df_corona[['ObservationDate', 'LastUpdate']].apply(pd.to_datetime)
# Converting State and Country to Strings
df_corona[['State', 'Country']] = df_corona[['State', 'Country']].astype('str')
# Converting confirmed, deaths and recovered to Integers
df_corona[['Confirmed', 'Deaths', 'Recovered']] = df_corona[['Confirmed', 'Deaths', 'Recovered']].astype('int')
df_corona.info()


# In[ ]:


# Checking the first five rows of the dataset to see what we have
df_corona.head()


# In[ ]:


# Last five rows of the dataset
df_corona.tail()


# In[ ]:


# Sort the dataset alphabetically by Country Name, State, then the ObservationDate
df_corona.sort_values(by=['Country', 'State', 'ObservationDate'], inplace=True)


# In[ ]:


# Setting the country as the index column
df_corona.set_index('Country', inplace=True)
df_corona.head()


# In[ ]:


df_corona.head(20)


# In[ ]:


# Diamond Princess Cruise Ship
df_corona.loc['Others'].head()


# <p>I noticed that the cruise ship had a different naming convention for the earlier dates which affects the cummulative sum.</p>

# In[ ]:


df_corona.loc['US'].head()


# In[ ]:


# Replace Cruise Ship with Diamond Princess Cruise Ship
df_corona['State'] = df_corona['State'].str.replace('Cruise Ship', 'Diamond Princess cruise ship')
df_corona.loc['Others']


# In[ ]:


# Cummulative Count of the patients by Country and State
c = df_corona.groupby(['Country', 'State'])[['Confirmed', 'Recovered', 'Deaths', 'State', 'ObservationDate']].tail(1)
c['Active'] = c['Confirmed'] - (c['Recovered'] + c['Deaths'])
corona_crd_country = c.groupby('Country')[['Confirmed', 'Recovered', 'Deaths', 'Active']].sum().sort_values(by="Confirmed", ascending=False)
corona_crd_country.head()


# In[ ]:


corona_crd_country[['Confirmed', 'Recovered', 'Deaths']].head(5).plot(kind="bar", figsize=(12, 8))
plt.title('Top 10 total Confirmed, Recovered and Death Cases by Country')


# In[ ]:


corona_crd_state = c.groupby(['Country', 'State'])[['Confirmed', 'Recovered', 'Deaths', 'Active']].sum().sort_values(by="Confirmed", ascending=False)
corona_crd_state.head()


# In[ ]:


corona_crd_state[['Confirmed', 'Recovered', 'Deaths']].head().plot(kind="bar", figsize=(12, 8))
plt.title("Top 10 total Confirmed, Recovered and Death Cases by Country and State")


# In[ ]:


# Total count of confirmed, Recovered and Deaths
total_count_crd = corona_crd_country.sum().sort_values(ascending=False)
total_count_crd


# In[ ]:


total_count_crd[['Confirmed', 'Recovered', 'Deaths']].plot(kind="barh", figsize=(12, 8))
total_count_crd.sort_values(ascending=False, inplace=True)
plt.title("Total count of the number of confirmed cases, recovered patients and deaths")


# In[ ]:


# Number of active cases
active_cases = total_count_crd['Active']
closed_cases = total_count_crd['Confirmed'] - active_cases
print(f"Active Cases: {total_count_crd['Active']}, Closed Cases: {closed_cases}")


# In[ ]:


perc_active_cases = (active_cases / total_count_crd['Confirmed']) * 100.0
perc_closed_cases = (closed_cases / total_count_crd['Confirmed']) * 100.0

pac = round(perc_active_cases)
pcc = round(perc_closed_cases)

labels = ['Active Cases', 'Closed Cases']
sizes = [pac, pcc]
explode = (0, 0.1)

plt.figure(figsize=(12, 8))
plt.pie(sizes, labels=labels, explode=explode, autopct='%1.1f%%', startangle=90)
plt.axis('equal')


# In[ ]:


china = df_corona.loc['Mainland China'].groupby(['ObservationDate'])[['Confirmed', 'Recovered', 'Deaths', 'ObservationDate']].sum()
print(china[0:5])
rest_of_world = df_corona.loc[df_corona.index != 'Mainland China'].groupby(['ObservationDate'])[['Confirmed', 'Recovered', 'Deaths', 'ObservationDate']].sum()
print(rest_of_world[0:5])
china[['Confirmed', 'Recovered', 'Deaths']].plot(kind="line",  figsize=(12, 8), label="China")
plt.title('Comparison of Confirmed, Recovered, Deaths between Mainland China and the rest of the world')
rest_of_world[['Confirmed', 'Recovered', 'Deaths']].plot(kind="line", figsize=(12, 8), label="Rest of the world")
plt.title('Rest of the world')
plt.legend()


# In[ ]:


total_china_count = china['Confirmed'].tail(1)[0]
rest_of_world_count = rest_of_world['Confirmed'].tail(1)[0]

china_perc = (total_china_count / total_count_crd['Confirmed']) * 100.0
rest_of_world_perc = (rest_of_world_count / total_count_crd['Confirmed']) * 100.0

china_perc = round(china_perc)
rest_of_world_perc = round(rest_of_world_perc)

china_perc, rest_of_world_perc


# In[ ]:


# Pie Chart of the confirmed cases in mainland china vs the rest of the world
labels = ['Mainland China', 'Rest of the world']
sizes = [china_perc, rest_of_world_perc]
explode = (0, 0.1)

plt.figure(figsize=(12, 8))
plt.pie(sizes, labels=labels, explode=explode, autopct='%1.1f%%', startangle=90)


# In[ ]:


# There is no country called 'others'. Very curious about what information its associated with
c.loc['Others']


# <p>This is the Diamond Princess is a cruise ship that housed about 3700 passengers and crew. There was a 14-day vacation for adventure seekers to explore South-East Asia until a passenger carrying COVID-19 entered the ship. There are about 696 passengers confirmed to the infected now and 6 deaths. The ship only found out about the virus on its way back to Yokohama, Japan</p>

# In[ ]:


recent_time = df_corona['ObservationDate'].max()
recent_time


# In[ ]:


df_corona[df_corona['ObservationDate'] == recent_time]


# In[ ]:




