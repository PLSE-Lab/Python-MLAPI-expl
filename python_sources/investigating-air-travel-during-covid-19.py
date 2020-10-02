#!/usr/bin/env python
# coding: utf-8

# I started this investigation to examine the number of international flights arriving into Sydney Airport during COVID-19. I used a fantastic open source API https://opensky-network.org/ to collect this data and have made this helper notebook for anyone else that wants to explore air travel data using their API.
# 
# This notebook will help you build a pandas dataframe like the following:
# 
#                           Canada  Germany  Hong Kong  Japan  Malaysia  New Zealand 
#        time_stamp_day                                                             
#            2020-04-01       0        0          2      0         1            2   
# 
# For my particular inputs, this was telling me the daily counts of flights arrving into Sydney airport and their origin country, over my duration of interest.

# In[ ]:


import requests
import pandas as pd
import pprint
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
pd.set_option('display.float_format', lambda x: '%.3f' % x)


# In[ ]:


#You will have to get a (free) account on opensky-network to obtain a username and password before continuing.
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
username = user_secrets.get_secret("username")
password = user_secrets.get_secret("password")


# Here I just do a basic plot of confirmed cases in the state of New South Wales.

# In[ ]:


airports=pd.read_csv("../input/airports.csv", header=0)
confirmed_cases=pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv")

cases=confirmed_cases.loc[confirmed_cases['Province/State'] == 'New South Wales']
cases=cases.drop(['Lat','Long','Province/State','Country/Region'],axis=1)
cases = cases.stack() 
df = cases.to_frame()
df.columns = ['count']


# In[ ]:


plt.rcParams['figure.figsize'] = [15, 10]
df.plot.line(rot=0,legend=False)
plt.xticks(rotation=90) 


# In[ ]:


def epoch_to_time(x):
    return pd.Timestamp(x+10*60*60, unit='s')

def flights_week(start,end, airport_code, country):
    '''
    description: Returns the daily counts and origin countries of flights into your airport of interest, over the given time period, note that 
    this duration cannot exceed 1 week.
    
    input: start = starting time of investigation in ISO 8601 format, e.g. 2020-04-01T00:00:00.007Z
           end = ending time of investigation in ISO 8601 format, e.g. 2020-04-07T23:59:59.597Z
           airport_code = ICAO airport code of airport of interest. These can be found in the airports dataset within this notebook.
           country = This will be used in the function to exclude data from this country. For example, setting this to Australia will exclude
                     flights with an origin country of Australia so this excludes domestic flights in my example.

    
    output: A pandas data frame with column headers as countries of origin flights and row index as date of arrival.     
    
    example: 
    
                      Country           Canada  Germany  Hong Kong  Japan  Malaysia  New Zealand 
       time_stamp_day                                                             
           2020-04-01                     0        0          2      0         1            2   
    
    '''
    utc_time_begin = datetime.strptime(start, "%Y-%m-%dT%H:%M:%S.%fZ")
    epoch_time_begin = int((utc_time_begin - datetime(1970, 1, 1)).total_seconds())

    utc_time_end = datetime.strptime(end, "%Y-%m-%dT%H:%M:%S.%fZ")
    epoch_time_end = int((utc_time_end - datetime(1970, 1, 1)).total_seconds())

    call="https://{}:{}@opensky-network.org/api/flights/arrival?airport={}&begin={}&end={}".format(username,password,airport_code,epoch_time_begin,epoch_time_end)
    res = requests.get(call)
    
    if res.status_code != 200:
        print("Error")
        print(res.status_code)

    con = res.json()
    flight_list = []
    for i in range(len(con)):
        flight_list.append([]*3)
        flight_list.append([con[i]['estDepartureAirport'],con[i]['callsign'],con[i]['lastSeen']])

    df = pd.DataFrame(flight_list, columns = ['departure_airport', 'call_sign','epoch_arrival'])
    df = df.dropna()
    df.isnull().values.any()

    df['time_stamp']=df.apply(lambda row: epoch_to_time(int(row['epoch_arrival'])),axis=1)
    df['time_stamp_day']=df["time_stamp"].values.astype('datetime64[D]')

    xx = df.merge(airports[['ICAO','Country']],left_on='departure_airport',right_on='ICAO', how='left')
    xv = xx.dropna()

    test1=xv.loc[xv['Country'] != country]
    test1 = test1.drop_duplicates()

    country_count = pd.crosstab(test1.time_stamp_day,test1.Country)
    return(country_count)


# In[ ]:


#Let us look at a week during COVID-19
#flights_week(start,end,airport_code,country)
covid_week = flights_week("2020-04-01T00:00:00.007Z","2020-04-07T23:59:59.597Z","YSSY","Australia")


# In[ ]:


plt.rcParams['figure.figsize'] = [15, 10]
prev_week.plot.line(rot=0)
plt.xticks(rotation=90)


# In[ ]:


#Try a week not impacted by COVID-19
#flights_week(start,end,airport_code,country)
non_covid_week = flights_week("2019-04-01T00:00:00.007Z","2019-04-07T23:59:59.597Z","YSSY","Australia")


# In[ ]:


plt.rcParams['figure.figsize'] = [15, 10]
non_covid_week.plot.line(rot=0)
plt.xticks(rotation=90)


# In[ ]:




