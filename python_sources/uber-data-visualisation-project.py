#!/usr/bin/env python
# coding: utf-8

# # Data cleasing and visualisation practice by using Uber dataset

# ## Description: 
# This exercise uses Uber dataset in New York for year 2014, from April to September. The main of this exercise are:
# 
# - Data cleasing including split datetime data into seperate columns in year, month, day and so on;
# - Data wrangling 
# - Data visualisation by using package Plotly, plotly.graphs_objs, ipywidgets to create interactive charts 

# ### 1. load package

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

import plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import ipywidgets as widgets
from IPython.display import display


# In[ ]:


init_notebook_mode(connected = False) # to set offline mode for interactive chart by plotly package 


# ### 2. load dataset 

# In[ ]:


# load six csv file and combine them all together into a big dataframe
filenames = []
pri_name = "uber-raw-data-"
last_name = "14.csv"
months = ['apr', 'may', 'jun', 'jul', 'aug', 'sep']

for month in months:
    filename = pri_name + month + last_name
    filenames.append(filename)

print(filenames)


# In[ ]:


# load all the data from all csv file 
big_df = []

for filename in filenames:
    uberdata = pd.read_csv('../input/uber-pickups-in-new-york-city/' + filename)
    big_df.append(uberdata)

uber_data = pd.concat(big_df)
print(uber_data.head())


# ### 3. Data preprocessing 

# In[ ]:


# define function for data cleansing 
def data_clean(data_frame):
    data_frame['Date/Time'] = pd.to_datetime(data_frame['Date/Time'])
    data_frame = data_frame.rename({'Date/Time': 'Date'}, axis = 1)
    
    data_frame['year'] = data_frame.Date.dt.year
    data_frame['month'] = data_frame.Date.dt.month
    data_frame['day'] = data_frame.Date.dt.day
    data_frame['weekday'] = data_frame.Date.dt.weekday
    data_frame['hour'] = data_frame.Date.dt.hour
    data_frame['minute'] = data_frame.Date.dt.minute
    data_frame['second'] = data_frame.Date.dt.second
    
    return data_frame

#### create new dataframe and check data shape
all_data = data_clean(uber_data)
print(all_data.shape)


# ### 4. Data visualisation 

# In[ ]:


# create function to plot bar chart
def plot_bar(xdata, ydata, chart_type, xlabel, colnum ):
    
    fig = go.Figure(data = [
        go.Bar(name = 'Count', 
               x = xdata, 
               y = ydata,
               marker_color = 'crimson',
              width = [0.5]*colnum)
       
    ])

    fig.update_layout(barmode = chart_type,
                     title = 'Trips by '+ xlabel,
                     xaxis = dict(title = xlabel),
                     yaxis = dict(title = 'Total Count'))

    fig.show()


# ### 4.1 Trips by Month

# In[ ]:


# group all the data based on differnt day in april
agg_month = all_data.groupby(['month'], as_index = False)['Base'].count()
# print(agg_month)

months = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']

plot_bar(months, agg_month['Base'], 'group', 'Month', 6)


# ### 4.2 Trips by Weekday and Month

# In[ ]:


# data preparation for weekday
agg_weekday = all_data.groupby(['month', 'weekday'], as_index = False)['Base'].count()
# print(agg_weekday[:7])
# print(len(agg_weekday))
def filter_weekday(dataset, num):
    data = []
    for index in range(0, len(dataset)):
        if dataset['weekday'][index] == num:
            target_data = dataset['Base'][index]
            data.append(target_data)
    
    return data


weekday_data = []
for i in range(0,7):
    data = filter_weekday(agg_weekday, i)
    weekday_data.append(data)

# print(weekday_data)


# In[ ]:


# create chart for visualization 

months = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']

fig = go.Figure(data = [
    go.Bar(name = 'Sunday', x = months, y = weekday_data[0]),
    go.Bar(name = 'Monday', x = months, y = weekday_data[1]),
    go.Bar(name = 'Tuesday', x = months, y = weekday_data[2]),
    go.Bar(name = 'Wednesday', x = months, y = weekday_data[3]),
    go.Bar(name = 'Thursday', x = months, y = weekday_data[4]),
    go.Bar(name = 'Friday', x = months, y = weekday_data[5]),
    go.Bar(name = 'Saturday', x = months, y = weekday_data[6])
    
           
])

fig.update_layout(barmode = 'group',
                 title = 'Trips by weekday and month',
                 xaxis = dict(title = 'Month'),
                 yaxis = dict(title = 'Total Count'),
                 bargap = 0.2,
                 bargroupgap = 0.1
)

fig.show()


# In[ ]:


months = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']

fig = go.Figure(data = [
    go.Bar(name = 'Sunday', x = months, y = weekday_data[0]),
    go.Bar(name = 'Monday', x = months, y = weekday_data[1]),
    go.Bar(name = 'Tuesday', x = months, y = weekday_data[2]),
    go.Bar(name = 'Wednesday', x = months, y = weekday_data[3]),
    go.Bar(name = 'Thursday', x = months, y = weekday_data[4]),
    go.Bar(name = 'Friday', x = months, y = weekday_data[5]),
    go.Bar(name = 'Saturday', x = months, y = weekday_data[6])
           
])

fig.update_layout(barmode = 'stack',
                 title = 'Trips by weekday and Month',
                 xaxis = dict(title = 'Month'),
                 yaxis = dict(title = 'Total Count'))

fig.show()


# #### 4.3 Trips by Day

# In[ ]:


# aggregate day count of each month and make a bar chart 

agg_day = all_data.groupby(['day'], as_index = False)['Base'].count()
# print(agg_day)

plot_bar(agg_day['day'], agg_day['Base'], 'group', 'Day',31)


# #### 4.4 Trips by Hour

# In[ ]:


hour = []
for x in range(0,24):
    hour.append(x)
    
agg_hour = all_data.groupby(['hour'], as_index = False)['Base'].count()

plot_bar(hour, agg_hour['Base'], 'group', 'Hour', 24)


# #### 4.5 Trips by Base and Month

# In[ ]:


# groupby base 

agg_base = all_data['Base'].value_counts()
# print(agg_base.index[0])
# print(agg_base.values)
title = agg_base.index

plot_bar(title, agg_base.values, 'group', 'Base', 5)


# In[ ]:


# trips by base and month data
all_data['counter'] = 1
basename = uber_data.Base.unique()

agg_BaseMonth = all_data.groupby(['month', 'Base'], as_index = False)['counter'].count()
# print(agg_BaseMonth)


# In[ ]:


months = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']

# create list for label of base 
option = []
for label in uber_data.Base.unique():
    option.append(label)
    

# dropdown list for bases
origin = widgets.Dropdown(
    options=option,
    value='B02512',
    description='Base Name:',
)

# function to build interactive bar chart 
def update_plot(origin):
    
    y_data = agg_BaseMonth[agg_BaseMonth.Base == origin].counter
    
    fig = go.Figure(data = [
        go.Bar(name = 'Base Count', 
               x = months, 
               y = y_data,
              marker_color = 'crimson',
        width = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),       
    ])

    fig.update_layout(barmode = 'group',
                     title = 'Trips by Base and Month',
                     xaxis = dict(title = 'Month'),
                     yaxis = dict(title = 'Total Count'))

    fig.show()
    

widgets.interactive(update_plot, origin = origin)


# In[ ]:




