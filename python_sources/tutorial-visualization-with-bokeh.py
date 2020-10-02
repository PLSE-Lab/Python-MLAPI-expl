#!/usr/bin/env python
# coding: utf-8

# > The purpose of this tutorial is to give an overview of how Bokeh can be used to generate reasonably good looking graphs. While most of how to go about this is available all over the world wide web in a distributed manner, hope all that research can be presented in a collective manner here.
# 

# > For the purpose of this tutorial, we will use the 2008 USA Flight Statistics data that has been used in the mlcourse.ai course.

# In[174]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import os
from bokeh.io import show, output_file
from bokeh.models import ColumnDataSource, Legend, LegendItem, Scatter
from bokeh.plotting import figure, output_file, show, output_notebook
from bokeh.models.tools import HoverTool
from bokeh.core.properties import value
from bokeh.palettes import Spectral10, Category20, Category20_17, inferno, magma, viridis
import matplotlib.pyplot as plt
from bokeh.transform import jitter

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **Graph 1: We will plot the flights per day for the entire year (2008) **
# 
# Choose only those columns we are interested in.

# In[175]:


dtype = {'DayOfWeek': np.uint8, 'DayofMonth': np.uint8, 'Month': np.uint8 , 
         'Cancelled': np.uint8, 'Year': np.uint16, 'FlightNum': np.uint16 , 
         'Distance': np.uint16, 'UniqueCarrier': str, 'CancellationCode': str, 
         'Origin': str, 'Dest': str, 'ArrDelay': np.float16, 
         'DepDelay': np.float16, 'CarrierDelay': np.float16, 
         'WeatherDelay': np.float16, 'NASDelay': np.float16, 
         'SecurityDelay': np.float16, 'LateAircraftDelay': np.float16, 
         'DepTime': np.float16}


# In[176]:


path = '../input/btstats/2008.csv'
flights_df = pd.read_csv(path, usecols=dtype.keys(), dtype=dtype)


# In[177]:


flights_df.head()


# Let us eliminate all flights with NaN departure time and create 2 new fields for departure hour and minute. Let us also remove the trailing zero's after the newly created fields.

# In[178]:


flights_df = flights_df[np.isfinite(flights_df['DepTime'])]
flights_df['DepHour'] = flights_df['DepTime'] // 100
flights_df['DepHour'].replace(to_replace=24, value=0, inplace=True)
flights_df['DepMin'] = flights_df['DepTime'] - flights_df['DepHour']*100


# In[179]:


flights_df['DepHour'] = flights_df['DepHour'].apply(lambda f: format(f, '.0f'))
flights_df['DepMin'] = flights_df['DepMin'].apply(lambda f: format(f, '.0f'))


# Let us build 2 new fields - date and date time of the flight. Then let us get the flights per day for the entire year and plot it with bokeh.

# In[180]:


flights_df['Date'] = pd.to_datetime(flights_df.rename(columns={'DayofMonth': 'Day'})[['Year', 'Month', 'Day']])

flights_df['DateTime'] = pd.to_datetime(flights_df.rename(columns={'DayofMonth': 'Day', 'DepHour': 'Hour', 'DepMin':'Minute'})                                        [['Year', 'Month', 'Day', 'Hour', 'Minute']])


# In[181]:


num_flights_by_date = flights_df.groupby('Date').size().reset_index()
num_flights_by_date.columns = ['Date', 'Count']


# In[182]:


TOOLS = "pan, wheel_zoom, box_zoom, box_select,reset, save" # the tools you want to add to your graph
source = ColumnDataSource(num_flights_by_date) # data for the graph


# In[183]:


# Graph has date on the x-axis
p = figure(title="Graph 1: Number of flights per day in 2008", x_axis_type='datetime',tools = TOOLS)

p.line(x='Date', y='Count', source=source) #build a line chart
p.xaxis.axis_label = 'Date'
p.yaxis.axis_label = 'Number of flights'

p.xgrid.grid_line_color = None

# add a hover tool and show the date in date time format
hover = HoverTool()
hover.tooltips=[
    ('Date', '@Date{%F}'),
    ('Count', '@Count')
]
hover.formatters = {'Date': 'datetime'}
p.add_tools(hover)
output_notebook() # show the output in jupyter notebook
show(p)


# **Learning:**
# 
# * How to create your FIRST interactive plot using Bokeh
# * Source data creation
# * Choose the different tools you want to show
# * Show corresponding data on hover

# **Graph2: We will try to show the Carriers that accounted for the flights as a stacked bar chart.**

# In[184]:


# for the sake of image clarity, let us take only a couple of months data
df = flights_df[flights_df['Date']<'03-01-2008']
ct = pd.crosstab(df.Date, df.UniqueCarrier)
carriers = ct.columns.values #list of the carriers


# In[185]:


ct = ct.reset_index() # we want to make the date a column
ct['Date'] = ct['Date'].astype(str) # to show it in the x-axis


# In[186]:


# Graph has date on the x-axis 

source = ColumnDataSource(data=ct) # data for the graph
Date = source.data['Date']

#legend = Legend(items=[LegendItem(legend_data)], position=(0,-30))

# x_range : specifies the x-axis values, in our case Date
p = figure(x_range=Date, title="Graph 2: Flights in the first 2 months of 2008, by carrier",           tools = TOOLS, width=750)

renderers = p.vbar_stack(carriers, x='Date', source=source, width=0.5, color=magma(20),              legend=[value(x) for x in carriers])

p.xaxis.axis_label = 'Date'
p.yaxis.axis_label = 'Number of flights'

p.xgrid.grid_line_color = None

p.y_range.start = 0
p.y_range.end = 25000 #to make room for the legend
p.x_range.range_padding = 0.1
p.xgrid.grid_line_color = None
p.axis.minor_tick_line_color = None
p.outline_line_color = None

#add hover
hover = HoverTool()
hover.tooltips=[
    ('Date', '@Date{%F}'),
    ('Carrier', '$name'), #$name provides data from legend
    ('Count', '@$name') #@$name gives the value corresponding to the legend
]
hover.formatters = {'Date': 'datetime'}
p.add_tools(hover)
p.xaxis.major_label_orientation = math.pi/2 #so that the labels do not come on top of each other

#move legend outside the plot so that it does not interfere with the data
# creating external legend did not work
# so doing a roundabout of creating an intenal legend, copying it over to a new legend
# placing it on right and nulling te internal legend
new_legend = p.legend[0]
p.legend[0].plot = None
p.add_layout(new_legend, 'right')
p.legend.click_policy="hide"

output_notebook()
show(p)


# **Learning:**
# 
# * How to create a stacked bar chart using Bokeh
# * Choose the color palette for the chart
# * How to zoom in and out with your mouse (Wheel Zoom)
# * Hover Data  - shows values corresponding to the data
# * X-axis label formatting
# * Place legend outside the chart area

# **Graph 3: We will show the flights between 2 cities as a bubble chart.**

# In[187]:


# Let us next plot a bubble chart of the flights between 2 cities.

df = flights_df[['Origin', 'Dest', 'UniqueCarrier', 'Distance']]
df['Flight'] = df['Origin']+'-'+df['Dest'] # new variable for flight


# In[188]:


df.head()


# In[189]:


#find number of flights  between any 2 cities and sort them
df_by_flight = df.groupby(['Flight']).agg({'Flight': 'count'}).sort_values(('Flight'), ascending=False)


# In[190]:


df_by_flight.head()


# In[191]:


df_by_flight.columns=['Count']
df_by_flight = df_by_flight.reset_index()

#merge it back with df to get other columns of interest
df_new = df_by_flight.merge(df, on='Flight')


# In[192]:


df_new = df_new.drop_duplicates(subset=['Flight'])


# In[193]:


df_new.head()


# In[194]:


df_new.describe()


# > The above shows that 75 percentile is at 1683 flights which is 1/15th of 13299 (total rows = 5356). We will plot only the top 100 rows of data.

# In[195]:


df_new = df_new.drop(columns=['Flight', 'UniqueCarrier', 'Distance'])


# In[196]:


df_new.head()


# In[197]:


df_new = df_new[0:100]


# In[198]:


df_new['Count_gr'] = df_new['Count']/5000 #for the sake of charting


# In[201]:


source = ColumnDataSource(data=df_new)
Origin_l = df_new['Origin'].unique()
Dest_l = df_new['Dest'].unique()

p = figure(title='Graph 3: Flights between 2 cities (top 100 values only)',x_range=Origin_l, y_range=Dest_l, tools=TOOLS, width=750)

p.circle(x='Origin', y='Dest', radius='Count_gr',
          fill_color='purple', fill_alpha=0.4, source=source,
          line_color=None)

p.x_range.range_padding = 0.5
p.y_range.range_padding = 0.5

#add hover
hover = HoverTool()
hover.tooltips=[
    ('From', '@Origin'),
    ('To', '@Dest'),
    ('Count', '@Count') #@$name gives the value corresponding to the legend
]

p.add_tools(hover)
p.xaxis.major_label_orientation = math.pi/2

output_notebook()
show(p)


# **Learning:**
# 
# * How to present categorical data for a scatter plot
# * Customizable parameters

# **References: **
# 
# [1] Bokeh documentation for handling categorical data
# https://bokeh.pydata.org/en/latest/docs/user_guide/categorical.html
# 
# [2] Hover for Stacked Bar Chart
# (https://github.com/bokeh/bokeh/blob/16e87ed63ca1aecaa42e93293f32d936685dcd3e/sphinx/source/docs/user_guide/examples/categorical_bar_stacked_hover.py)
# 
# [3] Visualization with Bokeh (https://programminghistorian.org/en/lessons/visualizing-with-bokeh)
# 
# [4] Legend outside chart area (https://stackoverflow.com/questions/48240867/how-can-i-make-legend-outside-plot-area-with-stacked-bar)
#  
# 
