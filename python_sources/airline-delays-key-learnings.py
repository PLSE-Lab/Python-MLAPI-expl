#!/usr/bin/env python
# coding: utf-8

# # Contents

# I published a post explaining a bit of data definitions and relationships in the following link:
# https://www.kaggle.com/usdot/flight-delays/discussion/29308
# For anyone interested to check out.
# 
# # Data Import & Initial Explore

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="whitegrid")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


airlines = pd.read_csv('../input/airlines.csv')
airlines.head()


# In[ ]:


airports= pd.read_csv('../input/airports.csv')
airports.head()


# In[ ]:


flights = pd.read_csv('../input/flights.csv', low_memory=False)
flights.head()


# In[ ]:


flights.tail()


# Looks like the flight data are across year 2015, let's explore a bit deep inside.

# In[ ]:


print (airlines.shape)
print (airports.shape)
print (flights.shape)


# In[ ]:


for col in airlines:
    print ("%d NULL values are found in column %s" % (airlines[col].isnull().sum().sum(), col))


# In[ ]:


for col in airports:
    print ("%d NULL values are found in column %s" % (airports[col].isnull().sum().sum(), col))


# In[ ]:


for col in flights:
    print ("%d NULL values are found in column %s" % (flights[col].isnull().sum().sum(), col))


# Looks like 3 airports are missing latitude and longitude information, which can be found and plugged-in easily. 
# The flight data is a bit mess, I believe the reasons for missing data can be various aspects (e.g. due to flight cancellation, data load/quality issue). But there are no missing information for the flight schedule information (origin & destination airports, scheduled departure, arrivals, distance, diverted, cancelled).
# 
# ** Let's explore a bit further on the last portion of the missing data - cancellation reason & various delays**

# In[ ]:


print (flights.shape[0]*flights.CANCELLED.mean())
print (flights.shape[0] - flights.CANCELLATION_REASON.isnull().sum().sum())


# The two numbers match up, which infers that only the cancelled flights have a cancellation reason. Intuitive!

# In[ ]:


print (flights['ARRIVAL_DELAY'][flights['ARRIVAL_DELAY'] >= 15].count())
print (flights.shape[0] - flights.AIR_SYSTEM_DELAY.isnull().sum().sum())


# The two numbers match up, which infers that only the flights with arrival delay >= 15 minutes having a detailed delay breakdown (e.g. air system, airline, weather).

# ** Next I will join the airline data to get started for the analysis **

# In[ ]:


flights_v1 = pd.merge(flights, airlines, left_on='AIRLINE', right_on='IATA_CODE', how='left')
flights_v1.drop('IATA_CODE', axis=1, inplace=True)
flights_v1.rename(columns={'AIRLINE_x': 'AIRLINE_CODE','AIRLINE_y': 'AIRLINE'}, inplace=True)


# # Flight Volume, Cancellation & Divertion Rate

# In[ ]:


airline_rank_v01 = pd.DataFrame({'flight_volume' : flights_v1.groupby(['AIRLINE'])['FLIGHT_NUMBER'].count()}).reset_index()
airline_rank_v01.sort_values("flight_volume", ascending=True, inplace=True)
flight_volume_total = airline_rank_v01['flight_volume'].sum()
airline_rank_v01['flight_pcnt'] = airline_rank_v01['flight_volume']/flight_volume_total


# In[ ]:


airline_rank_v02 = pd.DataFrame({'cancellation_rate' : flights_v1.groupby(['AIRLINE'])['CANCELLED'].mean()}).reset_index()
airline_rank_v02.sort_values("cancellation_rate", ascending=False, inplace=True)
airline_rank_v03 = pd.DataFrame({'divertion_rate' : flights_v1.groupby(['AIRLINE'])['DIVERTED'].mean()}).reset_index()
airline_rank_v03.sort_values("divertion_rate", ascending=False, inplace=True)
airline_rank_v1 = pd.merge(airline_rank_v01, airline_rank_v02, left_on='AIRLINE', right_on='AIRLINE', how='left')
airline_rank_v1 = pd.merge(airline_rank_v1, airline_rank_v03, left_on='AIRLINE', right_on='AIRLINE', how='left')


# In[ ]:


airline_rank_v1


# The following plots are based on the plotly package. The api_key below is tied to the username, which will be different for different users.

# __The following scripts work on my local Jupyter notebook, but not on Kaggle. Still figuring out why.. I attach the plot via html for now!__
# 
# import plotly.plotly as py
# import plotly.graph_objs as go
# from plotly import tools
# import numpy as np
# 
# y_flight_vol = list(airline_rank_v1.flight_volume)
# y_cancel_rate = list(airline_rank_v1.cancellation_rate)
# x_airline = list(airline_rank_v1.AIRLINE)
# 
# trace0 = go.Bar(x=y_flight_vol,y=x_airline,marker=dict(color='rgba(50, 171, 96, 0.6)',line=dict(color='rgba(50, 171, 96, 1.0)',
#             width=1),),name='Total Numbers of Flights by Airlines in 2015',orientation='h',)
# 
# trace1 = go.Scatter(x=y_cancel_rate,y=x_airline,mode='lines+markers',line=dict(color='rgb(128, 0, 128)'),name='Cancellation Rate by Airlines in 2015',)
# 
# layout = dict(title='Airline Rank Analysis - 1',yaxis1=dict(showgrid=False,showline=False,showticklabels=True,domain=[0, 0.85],),yaxis2=dict(
#         showgrid=False,showline=True,showticklabels=False,linecolor='rgba(102, 102, 102, 0.8)',linewidth=2,domain=[0, 0.85],),xaxis1=dict(
#         zeroline=False,showline=False,showticklabels=True,showgrid=True,domain=[0, 0.42],),xaxis2=dict(zeroline=False,showline=False,
#         showticklabels=True,showgrid=True,domain=[0.47, 1],side='top',dtick=25000,),
#     
# legend=dict(x=0.01,y=1.00,font=dict(size=10,),),margin=dict(l=160,r=20,t=70,b=70,),paper_bgcolor='rgb(248, 248, 255)',plot_bgcolor='rgb(248, 248, 255)',)
# 
# annotations = []
# 
# y_fv = np.round(y_flight_vol, decimals=2)
# y_cr = np.round(y_cancel_rate, decimals=4)
# 
# for ydn, yd, xd in zip(y_cr, y_fv, x_airline):
#     annotations.append(dict(xref='x2', yref='y2',y=xd, x=ydn,text='{:,}'.format(ydn),font=dict(family='Arial', size=12,color='rgb(128, 0, 128)'),showarrow=False))
#     annotations.append(dict(xref='x1', yref='y1',y=xd, x=yd,text=str(yd),font=dict(family='Arial', size=12,color='rgb(50, 171, 96)'),showarrow=False))
# 
# annotations.append(dict(xref='paper', yref='paper',font=dict(family='Arial', size=10,color='rgb(150,150,150)'),showarrow=False))
# 
# layout['annotations'] = annotations
# 
# fig = tools.make_subplots(rows=1, cols=2, specs=[[{}, {}]], shared_xaxes=True,
#                           shared_yaxes=False, vertical_spacing=0.001)
# 
# fig.append_trace(trace0, 1, 1)
# fig.append_trace(trace1, 1, 2)
# 
# fig['layout'].update(layout)
# py.iplot(fig, filename='airline_rank_analysis-1')

# %% HTML
# <iframe width="780" height="780" frameborder="0" scrolling="yes" src="https://plot.ly/~dongxu027/18.embed"></iframe>

# __The following scripts work on my local Jupyter notebook, but not on Kaggle. Still figuring out why.. I attach the plot at the end via html for now!__
# 
# import plotly.plotly as py
# import plotly.graph_objs as go
# from plotly import tools
# import numpy as np
# 
# y_flight_vol = list(airline_rank_v1.flight_volume)
# y_divert_rate = list(airline_rank_v1.divertion_rate)
# x_airline = list(airline_rank_v1.AIRLINE)
# 
# trace0 = go.Bar(x=y_flight_vol,y=x_airline,
#                 marker=dict(color='rgba(50, 171, 96, 0.6)',line=dict(color='rgba(50, 171, 96, 1.0)',width=1),),
#                 name='Total Numbers of Flights by Airlines in 2015', orientation='h',)
# 
# trace1 = go.Scatter(x=y_divert_rate,y=x_airline,mode='lines+markers',line=dict(color='rgb(128, 0, 128)'),
#                     name='Divertion Rate by Airlines in 2015',)
# 
# layout = dict(title='Airline Rank Analysis - 2',yaxis1=dict(showgrid=False,showline=False,showticklabels=True,
#         domain=[0, 0.85],),yaxis2=dict(showgrid=False,showline=True,showticklabels=False,linecolor='rgba(102, 102, 102, 0.8)',
#         linewidth=2,domain=[0, 0.85],),xaxis1=dict(zeroline=False,showline=False,showticklabels=True,showgrid=True,domain=[0, 0.42],),
#               xaxis2=dict(zeroline=False,showline=False,showticklabels=True,showgrid=True,domain=[0.47, 1],side='top',dtick=25000,),
#               legend=dict(x=0.01,y=1.00,font=dict(size=10,),),margin=dict(l=160,r=20,t=70,b=70,),paper_bgcolor='rgb(248, 248, 255)',
#     plot_bgcolor='rgb(248, 248, 255)',)
# 
# annotations = []
# 
# y_fv = np.round(y_flight_vol, decimals=2)
# y_dv = np.round(y_divert_rate, decimals=4)
# 
# for ydn, yd, xd in zip(y_dv, y_fv, x_airline):
#     annotations.append(dict(xref='x2', yref='y2', y=xd, x=ydn,text='{:,}'.format(ydn),font=dict(family='Arial', size=12,
#                                       color='rgb(128, 0, 128)'),showarrow=False))
#     annotations.append(dict(xref='x1', yref='y1',y=xd, x=yd,text=str(yd),font=dict(family='Arial', size=12,
#                                       color='rgb(50, 171, 96)'),showarrow=False))
# 
# annotations.append(dict(xref='paper', yref='paper',font=dict(family='Arial', size=10,color='rgb(150,150,150)'),showarrow=False))
# 
# layout['annotations'] = annotations
# 
# fig = tools.make_subplots(rows=1, cols=2, specs=[[{}, {}]], shared_xaxes=True,shared_yaxes=False, vertical_spacing=0.01)
# 
# fig.append_trace(trace0, 1, 1)
# fig.append_trace(trace1, 1, 2)
# 
# fig['layout'].update(layout)
# py.iplot(fig, filename='airline_rank_analysis-2')

# %% HTML
# <iframe width="790" height="790" frameborder="0" scrolling="yes" src="https://plot.ly/~dongxu027/20.embed"></iframe>

# Looks like the airlines with most number of flights in 2015 are Southwest (1.2M flights - ranked No.1), Delta, American and Skywest, which also have relatively low cancellation rate. American Eagle has the highest cancellation rate - around 5%. 
# 
# Also, the divertation rate seems positively correlated with the flight volume. Overall the divertion rate is less than 1%.

# # Taxi-in & Taxi-out Time 

# In[ ]:


airline_rank_v04 = pd.DataFrame({'taxi_out_time' : flights_v1.groupby(['AIRLINE'])['TAXI_OUT'].mean()}).reset_index()
airline_rank_v05 = pd.DataFrame({'taxi_in_time' : flights_v1.groupby(['AIRLINE'])['TAXI_IN'].mean()}).reset_index()


# In[ ]:


ax = plt.subplots()
sns.set_color_codes("pastel")
sns.set_context("notebook", font_scale=1.5)
ax = sns.barplot(x="taxi_out_time", y="AIRLINE", data=airline_rank_v04, color="g")
ax = sns.barplot(x="taxi_in_time", y="AIRLINE", data=airline_rank_v05, color="b")
ax.set(xlabel="taxi_time (taxi_in: blue, taxi_out: green)")


# Interestingly, we can see that overall taxi in time is less than taxi out time for all the airlines. All airlines have an average taxi_in time less than 10 minutes, while all taxi out time are greater than 10 minutes. Also, it seems Southwest has the shortest taxi-in and taxi-out time (at least among the shortest).

# # Airline Flight Speed (miles/hour)

# In[ ]:


flights_v1['fly_speed'] = 60*flights_v1['DISTANCE']/flights_v1['AIR_TIME']
sns.set_context("notebook", font_scale=2.5)
sns.set(style="ticks", palette="muted", color_codes=True)
ax = sns.violinplot(x="fly_speed", y="AIRLINE", data=flights_v1);
sns.despine(trim=True)


# Based on the violinplot, we can see that in average, the majority of flying speed accross airlines are close to 400~450 miles per hour; with the Hawaiian Airlines Inc. is the slowest airline and also large variation (by simply looking at the data shape distribution). 
# 
# It is intersting to see that in some rare cases, an aircraft can go as high as 800 miles per hour in average during a flight trip (recall how I calculate this value).
# 
# Let's further look at the average flying speed by different airlines:

# In[ ]:


airline_rank_v06 = pd.DataFrame({'fly_speed' : flights_v1.groupby(['AIRLINE'])['fly_speed'].mean()}).reset_index()
airline_rank_v06.sort_values("fly_speed", ascending=False)


# United Airlines is the fastest, while Hawaiian Airlines is at the bottom.

# # Arrival & Departure Delays

# In[ ]:


airline_rank_v07 = pd.DataFrame({'avg_arrival_delay' : flights_v1.groupby(['AIRLINE'])['ARRIVAL_DELAY'].mean()}).reset_index()
airline_rank_v08 = pd.DataFrame({'avg_departure_delay' : flights_v1.groupby(['AIRLINE'])['DEPARTURE_DELAY'].mean()}).reset_index()


# In[ ]:


airline_rank_v1 = pd.merge(airline_rank_v1, airline_rank_v07, left_on='AIRLINE', right_on='AIRLINE', how='left')
airline_rank_v1 = pd.merge(airline_rank_v1, airline_rank_v08, left_on='AIRLINE', right_on='AIRLINE', how='left')


# In[ ]:


ax = sns.set_color_codes("pastel")
sns.set_context("notebook", font_scale=1.5)
ax = sns.barplot(x="avg_departure_delay", y="AIRLINE", data=airline_rank_v08,
            label="accuracy", color="b")
ax = sns.barplot(x="avg_arrival_delay", y="AIRLINE", data=airline_rank_v07,
            label="accuracy", color="r")
ax.set(xlabel="delay_time (arrival: red, departure: blue)")


# Based on this analysis, looks like all the lines have longer departure delays than arrival delays, except for Hawwaiian Airlines. My intuition is that the flights can adjust speed to catch up time while departure delay sometimes are out of control.
# 
# Spirit Airlines and Frontier Airlines are among the longest arrival and departure delay airlines. It is worth noting that Alaska Airlines is the only airline among all to arrive the destination earlier than scheduled in average.

# # Look into the airline delay breakdown structure

# In[ ]:


airline_rank_v09 = pd.DataFrame(flights_v1.groupby(['AIRLINE'])['AIR_SYSTEM_DELAY', 'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY'].sum()).reset_index()
airline_rank_v09['total'] = airline_rank_v09['AIR_SYSTEM_DELAY'] + airline_rank_v09['AIRLINE_DELAY'] + airline_rank_v09['LATE_AIRCRAFT_DELAY'] + airline_rank_v09['WEATHER_DELAY']
airline_rank_v09['pcnt_LATE_AIRCRAFT_DELAY'] = (airline_rank_v09['LATE_AIRCRAFT_DELAY']/airline_rank_v09['total'])
airline_rank_v09['pcnt_AIRLINE_DELAY'] = (airline_rank_v09['AIRLINE_DELAY']/airline_rank_v09['total'])
airline_rank_v09['pcnt_AIR_SYSTEM_DELAY'] = (airline_rank_v09['AIR_SYSTEM_DELAY']/airline_rank_v09['total'])
airline_rank_v09['pcnt_WEATHER_DELAY'] = (airline_rank_v09['WEATHER_DELAY']/airline_rank_v09['total'])


# The security system delay constitute less than 0.5% accross all delay components, so I will remove it from analysis below - Just look at the other 4 types of delays. 

# __The following scripts work on my local Jupyter notebook, but not on Kaggle. Still figuring out why.. I attach the plot at the end via html.__
# 
# import plotly.plotly as py
# import plotly.graph_objs as go
# 
# top_labels = ['late aircraft delay', 'airline delay', 'air system delay','weather delay']
# 
# colors = ['rgba(38, 24, 74, 0.8)', 'rgba(71, 58, 131, 0.8)',
#           'rgba(122, 120, 168, 0.8)', 'rgba(164, 163, 204, 0.85)',
#           'rgba(190, 192, 213, 1)']
# 
# 
# x_data = [list(airline_rank_v09.loc[0,"pcnt_LATE_AIRCRAFT_DELAY":"pcnt_WEATHER_DELAY"]),
#          list(airline_rank_v09.loc[1,"pcnt_LATE_AIRCRAFT_DELAY":"pcnt_WEATHER_DELAY"]),
#          list(airline_rank_v09.loc[2,"pcnt_LATE_AIRCRAFT_DELAY":"pcnt_WEATHER_DELAY"]),
#          list(airline_rank_v09.loc[3,"pcnt_LATE_AIRCRAFT_DELAY":"pcnt_WEATHER_DELAY"]),
#          list(airline_rank_v09.loc[4,"pcnt_LATE_AIRCRAFT_DELAY":"pcnt_WEATHER_DELAY"]),
#          list(airline_rank_v09.loc[5,"pcnt_LATE_AIRCRAFT_DELAY":"pcnt_WEATHER_DELAY"]),
#          list(airline_rank_v09.loc[6,"pcnt_LATE_AIRCRAFT_DELAY":"pcnt_WEATHER_DELAY"]),
#          list(airline_rank_v09.loc[7,"pcnt_LATE_AIRCRAFT_DELAY":"pcnt_WEATHER_DELAY"]),
#          list(airline_rank_v09.loc[8,"pcnt_LATE_AIRCRAFT_DELAY":"pcnt_WEATHER_DELAY"]),
#          list(airline_rank_v09.loc[9,"pcnt_LATE_AIRCRAFT_DELAY":"pcnt_WEATHER_DELAY"]),
#          list(airline_rank_v09.loc[10,"pcnt_LATE_AIRCRAFT_DELAY":"pcnt_WEATHER_DELAY"]),
#          list(airline_rank_v09.loc[11,"pcnt_LATE_AIRCRAFT_DELAY":"pcnt_WEATHER_DELAY"]),
#          list(airline_rank_v09.loc[12,"pcnt_LATE_AIRCRAFT_DELAY":"pcnt_WEATHER_DELAY"]),
#          list(airline_rank_v09.loc[13,"pcnt_LATE_AIRCRAFT_DELAY":"pcnt_WEATHER_DELAY"])]
# 
# 
# y_data = list(airline_rank_v09['AIRLINE'])
# 
# traces = []
# 
# for i in range(0, len(x_data[0])):
#     for xd, yd in zip(x_data, y_data):
#         traces.append(go.Bar(x=xd[i],y=yd,orientation='h',marker=dict(color=colors[i],line=dict(color='rgb(248, 248, 249)',width=1))))
# 
# layout = go.Layout(xaxis=dict(showgrid=False,showline=False,showticklabels=False,zeroline=False,domain=[0.15, 1]
#     ),yaxis=dict(showgrid=False,showline=False,showticklabels=False,zeroline=False,
#     ),barmode='stack',paper_bgcolor='rgb(248, 248, 255)',plot_bgcolor='rgb(248, 248, 255)',margin=dict(l=120,r=10,t=140,b=80
#     ),showlegend=False,)
# 
# annotations = []
# 
# for yd, xd in zip(y_data, x_data):
#     annotations.append(dict(xref='paper', yref='y',x=0.14, y=yd,xanchor='right',text=str(yd),font=dict(family='Arial', size=14,
#                             color='rgb(67, 67, 67)'),showarrow=False, align='right'))
# 
#     annotations.append(dict(xref='x', yref='y',x=xd[0] / 2, y=yd,text=str((xd[0]*100).round(decimals=1)) + '%',
#                             font=dict(family='Arial', size=14,color='rgb(248, 248, 255)'),showarrow=False))
# 
#     if yd == y_data[-1]:
#         annotations.append(dict(xref='x', yref='paper',x=xd[0] / 2, y=1.1,text=top_labels[0],font=dict(family='Arial', size=14,
#                                 color='rgb(67, 67, 67)'),showarrow=False))
#     space = xd[0]
#     for i in range(1, len(xd)):
#             annotations.append(dict(xref='x', yref='y',x=space + (xd[i]/2), y=yd, text=str((xd[i]*100).round(decimals=1))+ '%',
#                                     font=dict(family='Arial', size=14,color='rgb(248, 248, 255)'),showarrow=False))
#             if yd == y_data[-1]:
#                 annotations.append(dict(xref='x', yref='paper',x=space + (xd[i]/2), y=1.1,text=top_labels[i],
#                                         font=dict(family='Arial', size=14,color='rgb(67, 67, 67)'),showarrow=False))
#             space += xd[i]
# 
# layout['annotations'] = annotations
# 
# fig = go.Figure(data=traces, layout=layout)
# py.iplot(fig)

# %% HTML
# <iframe width="790" height="790" frameborder="0" scrolling="yes" src="https://plot.ly/~dongxu027/36.embed"></iframe>

# Based on the above analysis, the late aircraft delay, the airline delay and air system delay consists of most of the arrival delays. 
# - The late aircraft delay consists of almost one third of the delays accross all airlines, with the Southwest Airlines being half of delays. 
# - The airline delay consists of 25~33% of all the delays, with Hawaiian Airlines the factor is the 58% of all the delays.
# - the air system delay is around 20~30% of all the delays, with Spirit Airlines being the most dominant factor and Hawaiian Airlines only 1.8%.
# - the weather delay factor varies among all the airlines and should be tied to weather conditions
