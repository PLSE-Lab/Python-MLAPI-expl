#!/usr/bin/env python
# coding: utf-8

#  # 2015 Flight Delays and Cancellations
# 
# ![Airport departures](https://images.unsplash.com/photo-1421789497144-f50500b5fcf0?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=900&q=60 "Flights")
# 
#  Photo by Matthew Smith on Unsplash.
# 
#  ### Which airline should you fly on to avoid significant delays?
# 
# 
#  The U.S. Department of Transportation's (DOT) Bureau of Transportation
#  Statistics tracks the on-time performance of domestic flights operated
#  by large air carriers. Summary information on the number of on-time, delayed,
#  canceled, and diverted flights is published in DOT's monthly Air Travel Consumer
#  Report and in this dataset of 2015 flight delays and cancellations.
# 
#  The flight delay and cancellation data was collected and published by the DOT's Bureau of Transportation Statistics.

#  ## Datasets
#  ### We will work with 3 dataset collected by the DOT's Bureau of Transportation Statistics.
# - airlines
# - airports
# - flights

#  ## Library Imports

# In[ ]:


import pandas as pd 
import numpy as np 
import plotly
from plotly.offline import iplot, init_notebook_mode
import plotly.graph_objs as go
import matplotlib.pyplot as plt 
import seaborn as sns 
from datetime import datetime

sns.set(style="whitegrid")
# warnings.filterwarnings("ignore")


#  ## Load Datasets

# In[ ]:


def load_datasets():
    airlines = pd.read_csv('../input/flight-delays/airlines.csv')
    airports = pd.read_csv('../input/flight-delays/airports.csv')
    flights = pd.read_csv('../input/flight-delays/flights.csv')
    return (airlines, airports, flights)

datasets = load_datasets()


# In[ ]:


airlines_df = datasets[0]
airports_df = datasets[1]
flights_df = datasets[2]


#  Lets take a look at the first few lines of each dataset
# 

#  #### Airlines

# In[ ]:


airlines_df.head()


# In[ ]:


print(f'Dataframe has {airlines_df.shape[0]} rows, and {airlines_df.shape[1]} columns.')


#  #### Airports

# In[ ]:


airports_df.head()


# In[ ]:


print(f'Dataframe has {airports_df.shape[0]} rows, and {airports_df.shape[1]} columns.')


#  #### Flights

# In[ ]:


weekday_dict = {
    1 : 'Monday',
    2 : 'Tuesday',
    3 : 'Wednesday',
    4 : 'Thursday',
    5 : 'Friday',
    6 : 'Saturday',
    7 : 'Sunday',
}

month_dict = {
    1 : 'Jan',
    2 : 'Feb',
    3 : 'Mar', 
    4 : 'Apr',
    5 : 'May',
    6 : 'Jun', 
    7 : 'Jul', 
    8 : 'Aug',
    9 : 'Sep',
    10 : 'Oct',
    11 : 'Nov',
    12 : 'Dec'
}

flights_df['DAY_OF_WEEK'] = flights_df['DAY_OF_WEEK'].map(weekday_dict)
flights_df['flight_date'] = [datetime(year, month, day) for year, month, day in zip(flights_df.YEAR, flights_df.MONTH, flights_df.DAY)]
flights_df['MONTH'] = flights_df['MONTH'].map(month_dict)
flights_df.head()


# In[ ]:


print(f'Dataframe has {flights_df.shape[0]} rows, and {flights_df.shape[1]} columns.')


#  ### Lets combine these dataframes in to one.
# 

# In[ ]:


# Rename airline code column.
airlines_df.rename(columns={'IATA_CODE':'AIRLINE_CODE'}, inplace=True)
# Rename airport code column.
airports_df.rename(columns={'IATA_CODE':'AIRPORT_CODE'}, inplace=True)
# Rename flights airline code column.
flights_df.rename(columns={'AIRLINE':'AIRLINE_CODE'}, inplace=True)
# Rename flights origin code column.
flights_df.rename(columns={'ORIGIN_AIRPORT':'ORIGIN_AIRPORT_CODE'}, inplace=True)
# Rename flights destination code column.
flights_df.rename(columns={'DESTINATION_AIRPORT':'DESTINATION_AIRPORT_CODE'}, inplace=True)



#  #### We merge flights_df and airlines_df on 'AIRLINE' column.

# In[ ]:


combined_df = pd.merge(flights_df, airlines_df, on='AIRLINE_CODE', how='left')


#  #### We merge flights_df and airports_df on 'ORIGIN_AIRPORT_CODE' column.

# In[ ]:


combined_df = pd.merge(combined_df, airports_df, left_on='ORIGIN_AIRPORT_CODE', right_on='AIRPORT_CODE', how='left')


#  #### We merge flights_df and airports_df on 'DESTINATION_AIRPORT_CODE' column.

# In[ ]:


combined_df = pd.merge(combined_df, airports_df, left_on='DESTINATION_AIRPORT_CODE', right_on='AIRPORT_CODE', how='left')

# Caculating flight airtime
combined_df['elapsed_time'] = combined_df['WHEELS_ON'] - combined_df['WHEELS_OFF']


#  #### There are null values throughout the CANCELLATION_REASON, AIR_SYSTEM_DELAY, SECURITY_DELAY, AIRLINE_DELAY, LATE_AIRCRAFT_DELAY, and WEATHER_DELAY columns.
#  #### I decide to fill the null values with 0.0.

# In[ ]:


combined_df.fillna(value=0.0, inplace=True)


# In[ ]:


combined_df.head()


# In[ ]:


# Rename origin airport meta columns.
combined_df.rename(columns={'AIRPORT_x':'ORIGIN_AIRPORT', 
                            'CITY_x':'ORIGIN_CITY', 
                            'STATE_x':'ORIGIN_STATE',
                            'COUNTRY_x':'ORIGIN_COUNTRY',
                            'LATITUDE_x':'ORIGIN_LATITUDE',
                            'LONGITUDE_x':'ORIGIN_LONGITUDE'}, inplace=True)


# In[ ]:


# Rename destination airport meta columns.
combined_df.rename(columns={'AIRPORT_y':'DESTINATION_AIRPORT', 
                            'CITY_y':'DESTINATION_CITY', 
                            'STATE_y':'DESTINATION_STATE',
                            'COUNTRY_y':'DESTINATION_COUNTRY',
                            'LATITUDE_y':'DESTINATION_LATITUDE',
                            'LONGITUDE_y':'DESTINATION_LONGITUDE'}, inplace=True)


#  ## Origin Airport Distribution

# In[ ]:


number_of_flights = combined_df.shape[0]


# In[ ]:


origin_airport_group = combined_df.groupby('ORIGIN_AIRPORT')['FLIGHT_NUMBER'].count().sort_values(ascending=False)


#  ## Destination Airport Distribution

# In[ ]:


destination_airport_group = combined_df.groupby('DESTINATION_AIRPORT')['FLIGHT_NUMBER'].count().sort_values(ascending=False)


#  ## Airline Distribution

# In[ ]:


airline_group = combined_df.groupby('AIRLINE')['FLIGHT_NUMBER'].count().sort_values(ascending=False)


#  ### Top 10 Origin Airport Distribution # of flights

# In[ ]:


labels = list(origin_airport_group[1:11].index)
values = list(origin_airport_group[1:11].values)

trace = go.Pie(labels=labels, values=values)
layout = go.Layout(title='Origin Airport Flight Distribution (Top 10)',
                    autosize=False,
                    width=800,
                    height=500,
                    margin=go.layout.Margin(
                        l=50,
                        r=50,
                        b=100,
                        t=100,
                        pad=4
                    ))

fig = go.Figure(data=[trace], layout=layout)
iplot(fig, filename='origin_distribution')


#  No suprise that flight distributions for both Origin and Destinaton airports are very similar.
# 

#  ### Top 10 Destination Airport Distribution # of flights

# In[ ]:


labels = list(destination_airport_group[1:11].index)
values = list(destination_airport_group[1:11].values)

trace = go.Pie(labels=labels, values=values)
layout = go.Layout(title='Destination Airport Flight Distribution',
                    autosize=False,
                    width=800,
                    height=500,
                    margin=go.layout.Margin(
                        l=50,
                        r=50,
                        b=100,
                        t=100,
                        pad=4
                    ))

fig = go.Figure(data=[trace], layout=layout)
iplot(fig, filename='destination_distribution')


# In[ ]:





#  ### Top 10 Airlines Distribution # of flights

# In[ ]:


labels = list(airline_group[:10].index)
values = list(airline_group[:10].values)

trace = go.Pie(labels=labels, values=values)
layout = go.Layout(title='Flight Distribution by Airline',
                    autosize=False,
                    width=800,
                    height=500,
                    margin=go.layout.Margin(
                        l=50,
                        r=50,
                        b=100,
                        t=100,
                        pad=4
                    ))

fig = go.Figure(data=[trace], layout=layout)
iplot(fig, filename='airline_distribution')




#  ### Top 20 Origin City Distribution # of flights

# In[ ]:


city_group = combined_df.groupby(['ORIGIN_CITY'])['FLIGHT_NUMBER'].count().sort_values(ascending=False)
city_group[1:21]

trace = go.Bar(x=city_group[1:21].index, 
                y=city_group[1:21].values, 
                name='city',
                marker={
                    'color':city_group[1:21].values,
                    'colorscale':'Reds',
                    'showscale':True,
                    },
                )

layout = go.Layout(title='Number of Flights from Origin City',
                    xaxis={'title':'Origin City'},
                    yaxis={'title':'# of Flights'},
                    autosize=False,
                    width=800,
                    height=500,
                    margin=go.layout.Margin(
                        l=50,
                        r=50,
                        b=100,
                        t=100,
                        pad=4
                    ))

fig = go.Figure(data=[trace], layout=layout)
iplot(fig, filename='origin_city_bar')


#  ### Top 10 Airlines Distribution # of flights

# In[ ]:



trace = go.Bar(x=airline_group[:10].index, 
                y=airline_group[:10].values, 
                name='airlines',
                marker={
                    'color':airline_group[:10].values,
                    'colorscale':'Reds',
                    'showscale':True,
                    },
                )

layout = go.Layout(title='Number of Flights by Airline',
                    xaxis={'title':'Airline'},
                    yaxis={'title':'# of Flights'},
                    autosize=False,
                    width=800,
                    height=500,
                    margin=go.layout.Margin(
                        l=50,
                        r=50,
                        b=100,
                        t=100,
                        pad=4
                    ))

fig = go.Figure(data=[trace], layout=layout)
iplot(fig, filename='airline_bar')


#  ### Distribution # of flights per Month

# In[ ]:


month_group = combined_df.groupby(['MONTH'])['FLIGHT_NUMBER'].count()

trace = go.Bar(x=month_group.index, 
                y=month_group.values, 
                name='month',
                marker={
                    'color':month_group.values,
                    'colorscale':'Reds',
                    'showscale':True,
                    },
                )

layout = go.Layout(title='Number of Flights per Month',
                    xaxis={'title':'Month'},
                    yaxis={'title':'# of Flights'},
                    autosize=False,
                    width=500,
                    height=500,
                    margin=go.layout.Margin(
                        l=50,
                        r=50,
                        b=100,
                        t=100,
                        pad=4
                    )
                )

fig = go.Figure(data=[trace], layout=layout)
iplot(fig, filename='month_bar')


#  ### Distribution # of flights per Day of the Week

# In[ ]:


day_group = combined_df.groupby(['DAY_OF_WEEK'])['FLIGHT_NUMBER'].count().sort_values(ascending=False)

trace = go.Bar(x=day_group.index, 
                y=day_group.values, 
                name='day_of_week',
                marker={
                    'color':day_group.values,
                    'colorscale':'Reds',
                    'showscale':True,
                    },
                )

layout = go.Layout(title='Number of Flights per Day Of Week',
                    xaxis={'title':'Day Of Week'},
                    yaxis={'title':'# of Flights'},
                    autosize=False,
                    width=500,
                    height=500,
                    margin=go.layout.Margin(
                        l=50,
                        r=50,
                        b=100,
                        t=100,
                        pad=4
                    )
                )

fig = go.Figure(data=[trace], layout=layout)
iplot(fig, filename='day_bar')


# In[ ]:


# flights_df.head(30)


# In[ ]:




