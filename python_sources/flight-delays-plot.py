#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
pd.plotting.register_matplotlib_converters()
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

import plotly.offline as py
import plotly.figure_factory as ff
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
from plotly import tools

df = pd.read_csv('../input/flight-delays/flights.csv')#, low_memory=False)


# In[ ]:


df


# In[ ]:


df.head(10).T


# * wheels_off = taxi_out + departure_time
# * wheels_on = air_time + wheels_off
# * arrival_time = taxi_in + wheels_on
# * elapsed_time = air_time + taxi_in + taxi_out

# **ARRIVAL_DELAY = ARRIVAL_TIME-SCHEDULED_ARRIVAL**

# In[ ]:


df


# In[ ]:


#flight_data.isna().sum()
pd.concat([df.isnull().sum(), 100 * df.isnull().sum()/len(df)], 
              axis=1).rename(columns={0:'Missing Records', 1:'Percentage (%)'})


# In[ ]:


#df
dff = df.ORIGIN_AIRPORT.value_counts()[:10]
len(dff)
dff


# In[ ]:


dff = df.ORIGIN_AIRPORT.value_counts()[:10]

trace = go.Bar(
    x=dff.index,
    y=dff.values,
    marker=dict(
        color = dff.values,
        colorscale='Jet',
        showscale=True
    )
)

data = [trace]
layout = go.Layout(
    title='Origin City Distribution', 
    yaxis = dict(title = '# of Flights')
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# In[ ]:


dff = df.ORIGIN_AIRPORT.value_counts()
len(dff)


# In[ ]:


dff = df.ORIGIN_AIRPORT.value_counts()[:10]
label = dff.index
size = dff.values

colors = ['skyblue', '#FEBFB3', '#96D38C', '#D0F9B1', 'gold', 'orange', 'lightgrey', 
          'lightblue','lightgreen','aqua']
trace = go.Pie(labels=label, values=size, marker=dict(colors=colors),hole = .2)

data = [trace]
layout = go.Layout(
    title='Origin Airport Distribution'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# In[ ]:


airports_path = "../input/flight-delays/airports.csv"
airports = pd.read_csv(airports_path)
airports


# In[ ]:


dff = df.AIRLINE.value_counts()[:10]

trace = go.Bar(
    x=dff.index,
    y=dff.values,
    marker=dict(
        color = dff.values,
        colorscale='Jet',
        showscale=True)
)

data = [trace]
layout = go.Layout(xaxis=dict(tickangle=15),
    title='Airline distribution', 
                   yaxis = dict(title = '# of Flights'))

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# In[ ]:


df = pd.merge(df,airlines, left_on='AIRLINE', right_on = 'IATA_CODE')
df.insert(loc=5, column='AIRLINE', value=df.AIRLINE_y)
df = df.drop(['AIRLINE_y','IATA_CODE'], axis=1)


# In[ ]:


airport = pd.read_csv('../input/flight-delays/airports.csv')
df = pd.merge(df,airport[['IATA_CODE','AIRPORT','CITY']], left_on='ORIGIN_AIRPORT', right_on = 'IATA_CODE')
df = df.drop(['IATA_CODE'], axis=1)
df = pd.merge(df,airport[['IATA_CODE','AIRPORT','CITY']], left_on='DESTINATION_AIRPORT', right_on = 'IATA_CODE')
df = df.drop(['IATA_CODE'], axis=1)


# In[ ]:


dff = df['AIRPORT_x'].value_counts()[:10]
label = dff.index
size = dff.values

colors = ['skyblue', '#FEBFB3', '#96D38C', '#D0F9B1', 'gold', 'orange', 'lightgrey', 
          'lightblue','lightgreen','aqua']
trace = go.Pie(labels=label, values=size, marker=dict(colors=colors),hole = .2)

data = [trace]
layout = go.Layout(
    title='Origin Airport Distribution'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# In[ ]:


dff = df.CITY_x.value_counts()[:10]

trace = go.Bar(
    x=dff.index,
    y=dff.values,
    marker=dict(
        color = dff.values,
        colorscale='Jet',
        showscale=True
    )
)

data = [trace]
layout = go.Layout(
    title='Origin City Distribution', 
    yaxis = dict(title = '# of Flights')
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# In[ ]:


dff = df.AIRLINE.value_counts()[:10]

trace = go.Bar(
    x=dff.index,
    y=dff.values,
    marker=dict(
        color = dff.values,
        colorscale='Jet',
        showscale=True)
)

data = [trace]
layout = go.Layout(xaxis=dict(tickangle=15),
    title='Airline distribution', 
                   yaxis = dict(title = '# of Flights'))

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# In[ ]:


dff = df.MONTH.value_counts().to_frame().reset_index().sort_values(by='index')
dff.columns = ['month', 'flight_num']
month = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May',
            6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
dff.month = dff.month.map(month)

trace = go.Bar(
    x=dff.month,
    y=dff.flight_num,
    marker=dict(
        color = dff.flight_num,
        colorscale='Reds',
        showscale=True)
)

data = [trace]
layout = go.Layout(
    title='# of Flights (monthly)', 
    yaxis = dict(title = '# of Flights'
                                                )
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# In[ ]:


df['dep_delay'] = np.where(df.DEPARTURE_DELAY>0,1,0)
df['arr_delay'] = np.where(df.ARRIVAL_DELAY>0,1,0)
dff = df.groupby('MONTH').dep_delay.mean().round(2)

dff.index = dff.index.map(month)
trace1 = go.Bar(
    x=dff.index,
    y=dff.values,
    name = 'Departure_delay',
    marker = dict(
        color = 'aqua'
    )
)

dff = df.groupby('MONTH').arr_delay.mean().round(2)
dff.index = dff.index.map(month)

trace2 = go.Bar(
    x=dff.index,
    y=dff.values,
    name='Arrival_delay',
    marker=dict(
        color = 'red'
    )
)

data = [trace1,trace2]
layout = go.Layout(
    title='% Delay (Months)', 
    yaxis = dict(title = '%')
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# In[ ]:





# In[ ]:


plt.figure(figsize=(10,6))

plt.title("Average Arrival Delay for Spirit Airlines Flights, by Month")

# Bar chart showing average arrival delay for Spirit Airlines flights by month
sns.barplot(x=df.index, y=df['NK'])

# Add label for vertical axis
plt.ylabel("Arrival delay (in minutes)")


# In[ ]:


airlines_path = "../input/flight-delays/airlines.csv"
airlines = pd.read_csv(airlines_path)


# In[ ]:


df = pd.merge(df,airlines, left_on='AIRLINE', right_on = 'IATA_CODE')
df.insert(loc=5, column='AIRLINE', value=df.AIRLINE_y)
df = df.drop(['AIRLINE_y','IATA_CODE'], axis=1)


# In[ ]:


airports_path = "../input/flight-delays/airports.csv"
airport = pd.read_csv(airports_path)
df = pd.merge(df,airport[['IATA_CODE','AIRPORT','CITY']], left_on='ORIGIN_AIRPORT', right_on = 'IATA_CODE')
df = df.drop(['IATA_CODE'], axis=1)
df = pd.merge(df,airport[['IATA_CODE','AIRPORT','CITY']], left_on='DESTINATION_AIRPORT', right_on = 'IATA_CODE')
df = df.drop(['IATA_CODE'], axis=1)


# In[ ]:


df.head()


# In[ ]:


plt.figure(figsize=(10,6))

# Add title
plt.title("Average Arrival Delay for Spirit Airlines Flights, by Month")

# Bar chart showing average arrival delay for Spirit Airlines flights by month
sns.barplot(x=flight_data.index, y=flight_data['NK'])

# Add label for vertical axis
plt.ylabel("Arrival delay (in minutes)")


# In[ ]:


dff = df['AIRPORT_x'].value_counts()[:10]
label = dff.index
size = dff.values

colors = ['skyblue', '#FEBFB3', '#96D38C', '#D0F9B1', 'gold', 'orange', 'lightgrey', 
          'lightblue','lightgreen','aqua']
trace = go.Pie(labels=label, values=size, marker=dict(colors=colors),hole = .2)

data = [trace]
layout = go.Layout(
    title='Origin Airport Distribution'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# In[ ]:


dff = df.CITY_x.value_counts()[:10]

trace = go.Bar(
    x=dff.index,
    y=dff.values,
    marker=dict(
        color = dff.values,
        colorscale='Jet',
        showscale=True
    )
)

data = [trace]
layout = go.Layout(
    title='Origin City Distribution', 
    yaxis = dict(title = '# of Flights')
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# In[ ]:


flight_path = "../input/flight-delays/flights.csv"
flight_data = pd.read_csv(flight_path )#, index_col="Date", parse_dates=True)


# In[ ]:


flight_data


# In[ ]:


DEPARTURE_features = ['SCHEDULED_DEPARTURE', 'DEPARTURE_TIME', 'DEPARTURE_DELAY']
flight_data[DEPARTURE_features]


# In[ ]:


dayOfWeek={1:'Monday', 2:'Tuesday', 3:'Wednesday', 4:'Thursday', 5:'Friday', 
                                           6:'Saturday', 7:'Sunday'}
dff = df.DAY_OF_WEEK.value_counts()
dff = dff.to_frame().sort_index()
dff.index = dff.index.map(dayOfWeek)

trace1 = go.Bar(
    x=dff.index,
    y=dff.DAY_OF_WEEK,
    name = 'Weather',
    marker=dict(
        color = dff.DAY_OF_WEEK,
        colorscale='Jet',
        showscale=True
    )
)

data = [trace1]
layout = go.Layout(
    title='# of Flights (Day of Week)', 
    yaxis = dict(title = '# of Flights'
                                                    )
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# In[ ]:


TIME_features = ['AIR_TIME', 'TAXI_IN', 'TAXI_OUT', 'ELAPSED_TIME']
flight_data[TIME_features]


# In[ ]:





# In[ ]:


plt.figure(figsize=(12,6))
sns.lineplot(x= flight_data['DISTANCE'] , y = flight_data['ELAPSED_TIME'] )


# In[ ]:


df = pd.merge(df,airlines, left_on='AIRLINE', right_on = 'IATA_CODE')
df.insert(loc=5, column='AIRLINE', value=df.AIRLINE_y)
df = df.drop(['AIRLINE_y','IATA_CODE'], axis=1)

