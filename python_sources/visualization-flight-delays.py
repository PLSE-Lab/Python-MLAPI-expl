#!/usr/bin/env python
# coding: utf-8

# <p>
# <center>
# <font size="6">
# Visualization: Flight Delay
# </font>
# </center>
# </p>
# 
# <p>
# <center>
# <font size="4">
# By Gaofeng Huang
# </font>
# </center>
# </p>
# 
# <p>
# <center>
# <font size="3">
# <a href="https://www.kaggle.com/usdot/flight-delays#flights.csv">
#     Source: 2015 Flight Delays and Cancellations 
#     </a>
# </font>
# </center>
# </p>

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import zipfile
import seaborn as sns
sns.set(rc={'figure.figsize':(12,8)})
# set the precision of 2 decimal place.
pd.set_option('display.precision',2)
# pd.options.display.float_format = '{:,.2f}'.format
# zipfile.ZipFile('flight-delays.zip').extractall('.')


# In[ ]:


# plotly
import plotly.offline as py
init_notebook_mode(connected=True)


# # Data Overview

# In[ ]:


path = '../input/flight-delays/'
# read data of flight.csv
# a warning claims 'ORIGIN_AIRPORT' and 'DESTINATION_AIRPORT' have type of int data
flight_raw = pd.read_csv(path+'flights.csv', dtype={'ORIGIN_AIRPORT': str, 
                                               'DESTINATION_AIRPORT': str})


# In[ ]:


# over 5 million rows, over 30 columns, quite a big data.
flight_raw.shape


# ### Raw Variables (Column Names)
# **YEAR**: Year of the Flight Trip  
# **MONTH**: Month of the Flight Trip  
# **DAY**: Day of the Flight Trip  
# **DAY_OF_WEEK**: Day of week of the Flight Trip  
# **AIRLINE**: Airline Identifier  
# **FLIGHT_NUMBER**: Flight Identifier  
# **TAIL_NUMBER**: Aircraft Identifier  
# **ORIGIN_AIRPORT**: Starting Airport  
# **DESTINATION_AIRPORT**: Destination Airport  
# **SCHEDULED_DEPARTURE**: Planned Departure Time  
# **DEPARTURE_TIME**: WHEEL_OFF - TAXI_OUT  
# **DEPARTURE_DELAY**: Total Delay on Departure  
# **TAXI_OUT**: The time duration elapsed between departure from the origin airport gate and wheels off  
# **WHEELS_OFF**: The time point that the aircraft's wheels leave the ground  
# **SCHEDULED_TIME**: Planned time amount needed for the flight trip  
# **ELAPSED_TIME**: AIR_TIME + TAXI_IN + TAXI_OUT  
# **AIR_TIME**: The time duration between wheels_off and wheels_on time  
# **DISTANCE**: Distance between two airports  
# **WHEELS_ON**: The time point that the aircraft's wheels touch on the ground  
# **TAXI_IN**: The time duration elapsed between wheels-on and gate arrival at the destination airport  
# **SCHEDULED_ARRIVAL**: Planned arrival time  
# **ARRIVAL_TIME**: WHEELS_ON + TAXI_IN  
# **ARRIVAL_DELAY**: ARRIVAL_TIME - SCHEDULED_ARRIVAL  
# **DIVERTED**: Aircraft landed on airport that out of schedule  
# **CANCELLED**: Flight Cancelled (1 = cancelled)  
# **CANCELLATION_REASON**: Reason for Cancellation of flight: A - Airline/Carrier; B - Weather; C - National Air System; D - Security  
# **AIR_SYSTEM_DELAY**: Delay caused by air system  
# **SECURITY_DELAY**: Delay caused by security  
# **AIRLINE_DELAY**: Delay caused by the airline  
# **LATE_AIRCRAFT_DELAY**: Delay caused by aircraft  
# **WEATHER_DELAY**: Delay caused by weather  

# In[ ]:


# take a look at the basic statistics
flight_raw.loc[:,['DEPARTURE_TIME', 'DEPARTURE_DELAY',
                  'AIR_TIME', 'DISTANCE', 'ARRIVAL_TIME', 
                  'ARRIVAL_DELAY', 'DIVERTED', 'CANCELLED'
                 ]].describe()


# In[ ]:


# Count the number of NAN and different values in each column
def count_NA_levels(data):
    for i in data.columns:
        x = data[i].unique()
        y = data[i]
        count_na = data.shape[0] - (y.dropna(axis=0, how='any')).shape[0]
        if count_na > 0:
            print(i + '({} NaN): '.format(count_na) + str(len(x)))
        else:
            print(i + '(no NaN): ' + str(len(x)))


# In[ ]:


count_NA_levels(flight_raw)


# In[ ]:


flight_dropna = flight_raw.dropna(axis=0, how='any', 
                                  subset=['ARRIVAL_DELAY', 'DEPARTURE_DELAY'])


# In[ ]:


count_NA_levels(flight_dropna)


# In[ ]:


flight_dropna.shape


# In[ ]:


flight_clean = flight_dropna.loc[:,['MONTH','DAY','DAY_OF_WEEK',
                           'AIRLINE','ORIGIN_AIRPORT','DESTINATION_AIRPORT',
                           'DEPARTURE_DELAY','ARRIVAL_DELAY', 'DISTANCE',
                           'AIR_SYSTEM_DELAY','SECURITY_DELAY','AIRLINE_DELAY',
                           'LATE_AIRCRAFT_DELAY','WEATHER_DELAY'
                          ]]


# In[ ]:


flight_clean.shape


# In[ ]:


count_NA_levels(flight_clean)


# In[ ]:


delay_over15min = flight_clean.dropna(subset=['AIR_SYSTEM_DELAY','SECURITY_DELAY',
                                        'AIRLINE_DELAY','LATE_AIRCRAFT_DELAY',
                                        'WEATHER_DELAY'], how='all')


# In[ ]:


count_NA_levels(delay_over15min)


# In[ ]:


delay_over15min.shape


# In[ ]:


flight_clean[flight_clean.ARRIVAL_DELAY >= 15].shape


# In[ ]:


flight = flight_clean.drop(['AIR_SYSTEM_DELAY','SECURITY_DELAY','AIRLINE_DELAY',
                           'LATE_AIRCRAFT_DELAY','WEATHER_DELAY'], axis=1)


# In[ ]:


flight.shape


# In[ ]:


# create a column to measure delay or not
# DELAY_OR_NOT: True (ARRIVAL_DELAY > 0), False (ARRIVAL_DELAY <= 0)

flight['DELAY_OR_NOT'] = flight.loc[:, ['ARRIVAL_DELAY']] > 0
flight.head()


# ### Variables (Column Names)
# 
# **MONTH**: Month of the Flight Trip  
# **DAY**: Day of the Flight Trip  
# **DAY_OF_WEEK**: Day of week of the Flight Trip  
# **AIRLINE**: Airline Identifier  
# **ORIGIN_AIRPORT**: Starting Airport  
# **DESTINATION_AIRPORT**: Destination Airport  
# **DEPARTURE_DELAY**: Total Delay on Departure  
# **ARRIVAL_DELAY**: ARRIVAL_TIME - SCHEDULED_ARRIVAL  
# **DISTANCE**: Distance between two airports  
# 
# **DELAY_OR_NOT**: True (ARRIVAL_DELAY > 0), False (ARRIVAL_DELAY <= 0)
# 
# **AIR_SYSTEM_DELAY**: Delay caused by air system  
# **SECURITY_DELAY**: Delay caused by security  
# **AIRLINE_DELAY**: Delay caused by the airline  
# **LATE_AIRCRAFT_DELAY**: Delay caused by aircraft  
# **WEATHER_DELAY**: Delay caused by weather  

# ## Data Analysis and Visualization
# ### Departure Delay and Arrival Delay

# In[ ]:


# read data of airlines.csv for the full name of airlines
airline_name = pd.read_csv(path+'airlines.csv')
airline_name


# In[ ]:


# merge the fullname of airline companies into flight data
flight_fullname = flight.rename(columns={'AIRLINE': 'IATA_CODE'})
flight_fullname = flight_fullname.merge(airline_name, on='IATA_CODE')


# In[ ]:


# Make clear on DEPARTURE_DELAY and ARRIVAL_DELAY
# Delay caused before departure or after departure?

airline_deparr_plot = flight_fullname.loc[:, ['AIRLINE', 
                                     'DEPARTURE_DELAY',
                                     'ARRIVAL_DELAY']].groupby('AIRLINE').mean()


# In[ ]:


airline_deparr_plot


# In[ ]:


airline_deparr_plot.plot.barh(figsize=(12,8), stacked=False)
plt.show()


# In[ ]:


# ARRIVAL_DELAY is the total delay, i.e. delay result of this flight
# extract the delay in the airtime and landing
airline_deparr_plot['ARRIVAL_DELAY'] = (- airline_deparr_plot['DEPARTURE_DELAY'] 
                                        + airline_deparr_plot['ARRIVAL_DELAY'])
ax = airline_deparr_plot.plot.barh(figsize=(12,8), stacked=True)
ax.legend(['DEPARTURE_DELAY', 'AIRTIME_LANDING_DELAY\n(ARRIVAL_DELAY-DEPARTURE_DELAY)'])
# ax.title('')
plt.show()


# ## Airlines

# In[ ]:


# use this function to select which airline we are interested
def flight_airline(airline):
    return flight.loc[flight['AIRLINE']==airline]
# draw the number of delays for [what, e.g. MONTH] we want to groupby
def draw_count_delay(data, select, kind='bar'):
    data_select = data.loc[:, [select, 'DELAY_OR_NOT']].groupby(select).sum()
    ax = data_select.plot(kind=kind, figsize=(10,6))
    ax.legend(['Number of Delays'])
    plt.show()


# In[ ]:


draw_count_delay(flight_fullname, 'AIRLINE')


# In[ ]:


# split by month
flight_month = flight.loc[:, ['MONTH']].groupby('MONTH').sum()
get_airline = flight.AIRLINE.unique()

for a in get_airline:
    flight_month[a] = flight_airline(a).loc[:, ['MONTH', 'DELAY_OR_NOT']]                                        .groupby('MONTH').sum()
#set full name
flight_month.columns = flight_fullname.AIRLINE.unique()


# In[ ]:


ax = flight_month.T.plot(kind='bar', stacked=True, figsize=(12,8), colormap='rainbow')
ax.set(ylabel='Number of Delays', xlabel='AIRLINE')
plt.show()


# In[ ]:


# view the market share of these airlines

def airline_marketshare(data=flight_fullname, by='AIRLINE', titlehere='Market Share of Airlines in 2015'):
    df = data.loc[:, [by]]
    df['Share %'] = 1
    top = df.groupby(by).sum().sort_values(by='Share %',ascending=False)
    top = top.reset_index()
    
    sharePlot = top['Share %'].plot.pie(subplots=True,
                                         autopct='%0.2f%%',
                                         fontsize=12,
                                         figsize=(10,10),
                                         legend=False,
                                         labels=top[by],
                                         shadow=False,
                                         explode=(0.01,0.02,0.03,0.04,0.05,0.06,
                                                  0.07,0.08,0.1,0.15,
                                                  0.2,0.25,
                                                  0.3,0.35)[:len(data[by].unique())],
                                         startangle=90,
                                         colormap='summer',
                                         title=titlehere
                                       )
    
    plt.show()


# In[ ]:


airline_marketshare()


# In[ ]:


def draw_pct_delay(data, select, kind='bar'):
    data_select = (100*(data.loc[:, [select, 'DELAY_OR_NOT']].groupby(select).sum())
                   /(data.loc[:, [select, 'DELAY_OR_NOT']].groupby(select).count()))
    data_select.plot(kind=kind)
    plt.show()


# In[ ]:


draw_pct_delay(flight, 'AIRLINE')


# In[ ]:


def draw_pct_delay_month(data, select, kind='bar'):
    for m in range(12):
        data_select['{}'.format(m)] = (100*(data.loc[:, [select, 'DELAY_OR_NOT']].groupby(select).sum())
                       /(data.loc[:, [select, 'DELAY_OR_NOT']].groupby(select).count()))
    data_select.plot(kind=kind, stacked=True)
    plt.show()


# In[ ]:


flight_month_pct = flight.loc[:, ['MONTH']].groupby('MONTH').sum()
airline = flight.AIRLINE.unique()

for a in airline:
    flight_a = flight_airline(a)
    flight_month_pct[a] = 100*(flight_a.loc[:, ['MONTH', 'DELAY_OR_NOT']].groupby('MONTH').sum()
                          /flight_a.shape[0])
flight_month_pct.columns = flight_fullname.AIRLINE.unique()


# In[ ]:


flight_month_pct.T


# In[ ]:


ax = flight_month_pct.T.plot.bar(figsize=(12,8), stacked=True, colormap='rainbow')
ax.set(ylabel='Percentage %',xlabel='AIRLINE')
plt.show()


# In[ ]:


# boxplot of delay time (min) and airlines

a = flight_fullname.loc[:, ['ARRIVAL_DELAY','AIRLINE']]
ax = sns.boxplot(y='AIRLINE', x='ARRIVAL_DELAY', data=a, linewidth=1, fliersize=2)
ax.set(xscale="log", xlabel='Delay Time (min)') #will ignore the negative value (no delays)
plt.show()


# ## Airports

# In[ ]:


# read airport data for longitude and latitude
# so we can draw it in the map
airport = pd.read_csv(path+'airports.csv')


# In[ ]:


# get the delay rate of (origin or destination) airports
def get_airport_plot(select):
    data_select = airport.rename(columns={'IATA_CODE': select})
    data_select_plot = pd.merge(data_select,
                               flight.loc[:, [select, 'DELAY_OR_NOT']]\
                               .groupby(select).mean().reset_index())
    
    data_select_plot['text_plot'] = ('Airport: ' + data_select_plot['AIRPORT'] + '<br>' 
                                    + 'City: ' + data_select_plot['CITY'] + '<br>'
                                    + 'State: ' + data_select_plot['STATE'] + '<br>'
                                    + 'Percentage of Delay: '
                                    + ((data_select_plot['DELAY_OR_NOT']*10000)\
                                       .astype(int)/100).astype(str) + '%<br>')
    
    return data_select_plot


# In[ ]:


airport_origin_plot = get_airport_plot('ORIGIN_AIRPORT')
# airport_origin_plot.head()


# In[ ]:


#colorscale made for plotting
scale = [[0.0, 'rgb(0,100,0)'],[0.2, 'rgb(34,139,34)'],
         [0.4, 'rgb(60,179,60)'],[0.6, 'rgb(173,255,47)'],
         [0.8, 'rgb(255,215,0)'],[1.0, 'rgb(255,99,71)']]
# draw the (origin or destination) airports in the map w.r.t delay rate
def delay_pct(dataplot, titlehere, filename):
    #data
    data = [dict(type='scattergeo',
                 lat=dataplot['LATITUDE'],
                 lon=dataplot['LONGITUDE'],
                 marker=dict(
                     autocolorscale=False, 
                     cmax=50, 
                     cmin=0, 
                     color= dataplot['DELAY_OR_NOT']*100,
                     colorbar=dict(title="Percentage of Delay (%)"), 
                     colorscale=scale, #'Viridis' 
                     line=dict(
                         color="rgba(102,102,102)", 
                         width=1
                     ), 
                     opacity=0.8, 
                     size=8
                 ),

                 text=dataplot['text_plot'],
                 mode='markers',
                )]
    
    #layout
    layout = dict(title= titlehere + '<br> Hover for value',
                 geo=dict(scope='USA',
                          projection=dict(type='albers usa'),
                          showlakes=True,
                          showland=True,
                          lakecolor='rgb(95,145,237)',
                          landcolor='rgb(250,250,250)',
                         )
                 )
    
    fig = dict(data=data, layout=layout)
    return py.iplot(fig, validate=False, filename=filename)


# In[ ]:


delay_pct(airport_origin_plot, 'Flight Delay Rate of Origin Airports in 2015', 'Delay_2015_Origin')


# In[ ]:


airport_destination_plot = get_airport_plot('DESTINATION_AIRPORT')
# airport_destination_plot.head()


# In[ ]:


delay_pct(airport_destination_plot, 'Flight Delay Rate of Destination Airports in 2015', 'Delay_2015_Destination')


# In[ ]:


# get the delay time info w.r.t (origin or destination) airports
def get_airport_delaytime_plot(select, val):
    #merge flight dataset and airport dataset
    data_select = airport.rename(columns={'IATA_CODE': select})
    data_select_plot = pd.merge(data_select,
                               flight.loc[:, [select, val]]\
                               .groupby(select).mean().reset_index())
    
    #for the hover text when plotting
    data_select_plot['text_plot'] = ('Airport: ' + data_select_plot['AIRPORT'] + '<br>' 
                                    + 'City: ' + data_select_plot['CITY'] + '<br>'
                                    + 'State: ' + data_select_plot['STATE'] + '<br>'
                                    + 'Average Delay Minutes: '
                                    + ((data_select_plot[val]*100)\
                                       .astype(int)/100).astype(str) + '<br>')
    
    return data_select_plot


# In[ ]:


airport_origin_delaytime_plot = get_airport_delaytime_plot('ORIGIN_AIRPORT', 'ARRIVAL_DELAY')
# airport_origin_delaytime_plot.head()


# In[ ]:


# delay as origin vs. delay as destination (delay rate)
a = airport_origin_plot.rename(columns={'ORIGIN_AIRPORT': 'APT',
                                        'DELAY_OR_NOT': 'DELAYasORIGIN'})
b = airport_destination_plot.rename(columns={'DESTINATION_AIRPORT': 'APT',
                                             'DELAY_OR_NOT': 'DELAYasDEST'})
c = a.loc[:,['APT','DELAYasORIGIN']].merge(b.loc[:,['APT','DELAYasDEST']], on='APT')
c.set_index('APT').corr()


# In[ ]:


# delay as origin vs. delay as destination (delay rate)
sns.lmplot(x='DELAYasORIGIN', y='DELAYasDEST',
           data=c, height=9)
plt.show()


# In[ ]:


#draw the (origin or destination) airports in the map visualized by delay time
def delay_time(dataplot, titlehere, filename):
    #data
    data = [dict(type='scattergeo',
                 lat=dataplot['LATITUDE'],
                 lon=dataplot['LONGITUDE'],
                 marker=dict(
                     autocolorscale=False, 
                     cmax=15, 
                     cmin=-20, 
                     color= dataplot[dataplot.columns[-2]],
                     colorbar=dict(title="Average Delay Minutes"), 
                     colorscale=scale, #'Viridis' 
                     line=dict(
                         color="rgba(102,102,102)", 
                         width=1
                     ), 
                     opacity=0.8, 
                     size=8
                 ),

                 text=dataplot['text_plot'],
                 mode='markers',
                )]
    
    #layout
    layout = dict(title=titlehere + '<br> Hover for value',
                 geo=dict(scope='USA',
                          projection=dict(type='albers usa'),
                          showlakes=True,
                          showland=True,
                          lakecolor='rgb(95,145,237)',
                          landcolor='rgb(250,250,250)',
                         )
                 )
    
    fig = dict(data=data, layout=layout)
    return py.iplot(fig, validate=False, filename=filename)


# In[ ]:


delay_time(airport_origin_delaytime_plot, 'Average Flight Delay Time of Origin Airports in 2015', 'Avg_DelayTime_Origin')


# In[ ]:


airport_dest_delaytime_plot = get_airport_delaytime_plot('DESTINATION_AIRPORT', 'ARRIVAL_DELAY')
# airport_dest_delaytime_plot.head()


# In[ ]:


delay_time(airport_dest_delaytime_plot, 'Average Flight Delay Time of Destination Airports in 2015', 'Avg_DelayTime_Destionation')


# In[ ]:


# delay as origin vs. delay as destination (delay time)
a = airport_origin_delaytime_plot.rename(columns={'ORIGIN_AIRPORT': 'APT',
                                        'ARRIVAL_DELAY': 'DELAYasORIGIN'})
b = airport_dest_delaytime_plot.rename(columns={'DESTINATION_AIRPORT': 'APT',
                                             'ARRIVAL_DELAY': 'DELAYasDEST'})
c = a.loc[:,['APT','DELAYasORIGIN']].merge(b.loc[:,['APT','DELAYasDEST']], on='APT')
c.set_index('APT').corr()


# In[ ]:


# delay as origin vs. delay as destination (delay time)
sns.lmplot(x='DELAYasORIGIN', y='DELAYasDEST',
           data=c, height=9)
plt.show()


# In[ ]:


# get the info (origin and destination airports) of each flight route
def route(flight):
    #for origin airports info
    a = flight.loc[:, ['ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'ARRIVAL_DELAY']]
    a['AtoB'] = a['ORIGIN_AIRPORT'] + a['DESTINATION_AIRPORT']
    b = a.loc[:,['AtoB','ARRIVAL_DELAY']].groupby('AtoB').mean()
    b['IATA_CODE'] = b.index.str[:3]
    c = pd.merge(airport, b.reset_index())
    c.rename(columns={'IATA_CODE': 'ORIGIN_AIRPORT'}, inplace=True)
    c_sorted = c.set_index('AtoB').sort_index()
    
    #for destination airports info
    b.drop('IATA_CODE', axis=1, inplace=True)
    b['IATA_CODE'] = b.index.str[3:]
    d = pd.merge(airport, b.reset_index())
    d.rename(columns={'IATA_CODE': 'DESTINATION_AIRPORT'}, inplace=True)
    d_sorted = d.set_index('AtoB').sort_index()
    
    return (c_sorted, d_sorted)


# In[ ]:


#draw the route map visualized by the delay time
def delay_time_route(airline='AA'):
    #data
    
    route_origin, route_destination = route(flight_airline(airline))

    #draw the airport points in the map
    data = [dict(type='scattergeo',
                 lat=route_origin['LATITUDE'],
                 lon=route_origin['LONGITUDE'],
                 marker=dict(
                     color='#FFD700',
                     line=dict(
                         color="rgba(102,102,102)", 
                         width=1
                     ), 
                     opacity=0.8, 
                     size=6
                 ),
                 mode='markers',
                )]   
    
    #draw the flight route in the map
    for i in range(route_origin.shape[0]):
        data += [dict(
            lat=[route_origin['LATITUDE'][i], route_destination['LATITUDE'][i]], 
            line=dict(  
                color='#4682B4',
                width=1
            ), 
            locationmode="USA-states", 
            lon=[route_origin['LONGITUDE'][i], route_destination['LONGITUDE'][i]], 
            mode="lines",
            text=('From: ' + (route_origin['AIRPORT'][i]) 
                  + '<br>To: ' + route_destination['AIRPORT'][i]
                  + '<br>Avg. Delay: ' 
                  + ((route_origin['ARRIVAL_DELAY'][i]*100).astype(int)/100).astype(str) 
                  + ' mins'
                 ),
            opacity=(route_origin['ARRIVAL_DELAY'][i])/30 if (route_origin['ARRIVAL_DELAY'][i])>0 else 0, 
            type="scattergeo"
        )]
    
    #layout
    layout = dict(title='Average Delay Time of Airline {} Route in 2015'.format(airline)
                  + '<br> Hover for value',
                 geo=dict(scope='north america',
                          projection=dict(type="azimuthal equal area"),
                          showlakes=True,
                          showland=True,
                          lakecolor='rgb(95,145,237)',
                          landcolor='rgb(250,250,250)'
                         ),
                  showlegend=False
                 )
    
    fig = dict(data=data, layout=layout)
    return py.iplot(fig, validate=False, filename='Delay_Route_2015_{}'.format(airline))


# In[ ]:


delay_time_route('DL')


# In[ ]:


delay_time_route('AA')


# In[ ]:


delay_time_route('HA')


# In[ ]:


# get the main reason of a delay

delay_over15min['DELAY_REASON'] = None
for reason in delay_over15min.columns[-6:-1]:
    delay_over15min['DELAY_REASON'][delay_over15min[reason]
                                    /delay_over15min['ARRIVAL_DELAY'] > 0.5] = reason


# In[ ]:


# find the main delay reason of each airline

# calculate the reason of delays w.r.t the delay time (min)
# i.e. if a delay occurs, how long is the delay time in this delay reason?
ax = sns.barplot(x='AIRLINE', y='ARRIVAL_DELAY',
                data=delay_over15min, 
                hue='DELAY_REASON')
ax.set(ylabel='Delay Time (min)')
plt.show()


# In[ ]:


# calculate the reason caused delays in percentage
# i.e. if a delay occurs, what is the probability of this delay reason?
count_delay_reason = delay_over15min.loc[:, ['AIRLINE', 'AIR_SYSTEM_DELAY',
                                             'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY',
                                             'SECURITY_DELAY', 'WEATHER_DELAY'
                                            ]]
count_delay_reason.set_index('AIRLINE', inplace=True)
count_delay_reason = 100*(((count_delay_reason>0).reset_index().groupby('AIRLINE').sum())
                      /count_delay_reason.groupby('AIRLINE').count())
count_delay_reason


# Comparing with the previous bar chart, weather delays are less probable occured. However, if a weather delay happens, it will cause a longer delay time in average.

# In[ ]:


ax = count_delay_reason.plot.bar(width=0.8)
ax.set(ylabel='Percentage %')
plt.show()


# In[ ]:




