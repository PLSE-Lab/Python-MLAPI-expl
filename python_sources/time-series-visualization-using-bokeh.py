#!/usr/bin/env python
# coding: utf-8

# ## Time-Series Visualization using bokeh
# The aim of this notebook is to provide a quick guide where you can find handy examples of plotting bokeh charts. The dataset used here is the SF Monthly Property Crime Report.
# 
# Let us first import the required packages and take a look at the data. 

# In[40]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings('ignore')
from bokeh.plotting import figure, show, output_file, output_notebook
from bokeh.palettes import Spectral11, colorblind, Inferno, BuGn, brewer
from bokeh.models import HoverTool, value, LabelSet, Legend, ColumnDataSource,LinearColorMapper,BasicTicker, PrintfTickFormatter, ColorBar
import datetime

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
output_notebook()

# Any results you write to the current directory are saved as output.


# In[41]:


data = pd.read_csv('../input/Monthly_Property_Crime_2005_to_2015.csv', parse_dates=['Date'])
data.head()


# In[42]:


data.Date.min(), data.Date.max()


# In[43]:


data.Category.value_counts()


# As we can see, the data ranges from 2005-2015 and is a compilation of 6 types of Property Crime - Vehicle Theft, Stolen Property, Larceny/Theft, Burglary, Arson, Vandalism. Let us now do some preprocessing on the data by creating Month and Year columns which will be used for further analysis.

# In[44]:


data['Year'] = data.Date.apply(lambda x: x.year)
data['Month'] = data.Date.apply(lambda x: x.month)
data.head()


# ### Bar Chart
# The purpose of a bar chart is to show the trend/comparison between one quatntity over another. The plot here is a simple bar chart where we plot the average number of crimes by month. For this, we first aggregate by month and use bokeh's `vbar()` function that takes in Month as the x-axis and the sum of incidents as the y-axis (top).
# 
# `TOOLS = "hover,save,pan,box_zoom,reset,wheel_zoom,tap"` enables functionalities such as hover, tap, etc while `p.select_one(HoverTool).tooltips = [
#     ('month', '@x'),
#     ('Number of crimes', '@top')]`
#     is used to customise the tooltip display.
#     
#  Analysis - This is used to analyse the average crimes per month. All the months have between 600-800 average crimes, with February being the least.

# In[30]:


temp_df = data.groupby(['Month']).mean().reset_index()
temp_df.head()


# In[31]:


TOOLS = "hover,save,pan,box_zoom,reset,wheel_zoom,tap"
p = figure(plot_height=350,
    title="Average Number of Crimes by Month",
    tools=TOOLS,
    toolbar_location='above')

p.vbar(x=temp_df.Month, top=temp_df.IncidntNum, width=0.9)

p.y_range.start = 0
p.x_range.range_padding = 0.1
p.xgrid.grid_line_color = None
p.axis.minor_tick_line_color = None
p.outline_line_color = None
p.xaxis.axis_label = 'Month'
p.yaxis.axis_label = 'Average Crimes'
p.select_one(HoverTool).tooltips = [
    ('month', '@x'),
    ('Number of crimes', '@top'),
]
output_file("barchart.html", title="barchart")
show(p)


# ### Line Chart
# Line chart is used primarily to show trend, i.e. whether there is an increase or decrese over a given x-axis.
# This plot here, again is a basic line chart, used to show the overall increase/decrease in crime over the years. The `line()` function is used to plot the line while  `circle()` is used to point out the value of interest, which in this case is the point where the crime rate was the lowest.
# 
# Analysis - This plot shows the trend in number of crimes over the years. It can be seen that the crime rate decreased from 2005-2010, with 2010 having the lowest crime rate. From then on, it has kept increasing steadily with 2015 having the highest number of crimes.

# In[51]:


temp_df = data.groupby(['Year']).sum().reset_index()
temp_df.head()


# In[52]:


TOOLS = 'save,pan,box_zoom,reset,wheel_zoom,hover'
p = figure(title="Year-wise total number of crimes", y_axis_type="linear", plot_height = 400,
           tools = TOOLS, plot_width = 800)
p.xaxis.axis_label = 'Year'
p.yaxis.axis_label = 'Total Crimes'
p.circle(2010, temp_df.IncidntNum.min(), size = 10, color = 'red')

p.line(temp_df.Year, temp_df.IncidntNum,line_color="purple", line_width = 3)
p.select_one(HoverTool).tooltips = [
    ('year', '@x'),
    ('Number of crimes', '@y'),
]

output_file("line_chart.html", title="Line Chart")
show(p)


# ### Stacked Bar Chart
# Stacked bar charts are used to show comparison, not only across the x-axis (years in this case), but also across the categories for a given value of the x-axis. Thus we know which category was maximum or minimum for every value of year, and also the overall change in values for each category over the years.
# 
# For our example,, we first need to convert the data from long format to wide format, i.e., we need a column for each one of the crime types corresponding to a given year. Then, using `vbar_stack()` with Years as x axis and all the crime types as `stackers` we can plot the stacked bar chart.
# 
# Analysis - This chart explores the distribution of crimes among the various categories over the years. In particular, larceny/theft are the most frequently occuring crimes, while stolen property occur the least. 2005 saw a high number of vehicle thefts, which reduced quite a bit subsequently.

# In[53]:


wide = data.pivot(index='Date', columns='Category', values='IncidntNum')
wide.reset_index(inplace=True)
wide['Year'] = wide.Date.apply(lambda x: x.year)
wide['Month'] = wide.Date.apply(lambda x: x.month)

temp_df = wide.groupby(['Year']).sum().reset_index()
temp_df.head()
cats = ['ARSON','BURGLARY','LARCENY/THEFT','STOLEN PROPERTY','VANDALISM','VEHICLE THEFT'] 
temp_df.drop(['Month'], axis = 1, inplace=True)
temp_df.head()


# In[54]:


TOOLS = "save,pan,box_zoom,reset,wheel_zoom,tap"

source = ColumnDataSource(data=temp_df)
p = figure( plot_width=800, title="Category wise count of crimes by year",toolbar_location='above', tools=TOOLS)
colors = brewer['Dark2'][6]

p.vbar_stack(cats, x='Year', width=0.9, color=colors, source=source,
             legend=[value(x) for x in cats])

p.y_range.start = 0
p.x_range.range_padding = 0.1
p.xgrid.grid_line_color = None
p.axis.minor_tick_line_color = None
p.outline_line_color = None
p.xaxis.axis_label = 'Year'
p.yaxis.axis_label = 'Total Crimes'
p.legend.location = "top_left"
p.legend.orientation = "horizontal"

output_file("stacked_bar.html", title="Stacked Bar Chart")

show(p)


# ### Heat Map
# Again, going back to the long format, we group the crime data by month and year. This time, we plot a heat map that shows how crime rate varied with combinations of months and years. The darker colours represent higher number of crimes. 
# 
# Analysis - Months in 2015 have the highest total while the least are in the some months in 2009 and 2010.

# In[55]:


temp_df = data.groupby(['Year', 'Month']).sum().reset_index()
# temp_df['Month_Category'] = pd.concat([temp_df['Month'], temp_df['Category']], axis = 1)
temp_df.head()


# In[56]:


TOOLS = "hover,save,pan,box_zoom,reset,wheel_zoom,tap"
hm = figure(title="Month-Year wise crimes", tools=TOOLS, toolbar_location='above')

source = ColumnDataSource(temp_df)
colors = brewer['BuGn'][9]
colors = colors[::-1]
mapper = LinearColorMapper(
    palette=colors, low=temp_df.IncidntNum.min(), high=temp_df.IncidntNum.max())
hm.rect(x="Year", y="Month",width=2,height=1,source = source,  
    fill_color={
        'field': 'IncidntNum',
        'transform': mapper
    },
    line_color=None)
color_bar = ColorBar(
    color_mapper=mapper,
    major_label_text_font_size="10pt",
    ticker=BasicTicker(desired_num_ticks=len(colors)),
    formatter=PrintfTickFormatter(),
    label_standoff=6,
    border_line_color=None,
    location=(0, 0))

hm.add_layout(color_bar, 'right')
hm.xaxis.axis_label = 'Year'
hm.yaxis.axis_label = 'Month'
hm.select_one(HoverTool).tooltips = [
    ('Year', '@Year'),('Month', '@Month'), ('Number of Crimes', '@IncidntNum')
]

output_file("heatmap.html", title="Heat Map")

show(hm)  # open a browser


# ### Multiline Plot
# This plot is made by plotting multiple line plots on the same chart. It facilitates comaprison between various categories that are represented in the data, in this case the trend in category wise crimes, over the years. The `crosshair` tool is useful when you want to compare values at a given point in time.
# 
# For our data, we first need to create dataframes corresponding to each crime type, and then use individual lines for each of these categories.
# 
# Analysis - This plot shows the distribution of crimes across categories over the years. This plot shows information similar to the stacked bar chart, except that here it is easier to note that arson and property theft amount to almost the same amount of crimes every year. Similarly, vehicle theft(except for 2005), vandalism and burglary have very similar patterns. Only larceny/theft is increasing with every passing year and its count is much higher than any of the other types of crimes.

# In[57]:


burglary = data[data.Category == 'BURGLARY'].sort_values(['Date'])
stolen_property = data[data.Category == 'STOLEN PROPERTY'].sort_values(['Date'])
vehicle_theft = data[data.Category == 'VEHICLE THEFT'].sort_values(['Date'])
vandalism = data[data.Category == 'VANDALISM'].sort_values(['Date'])
larceny = data[data.Category == 'LARCENY/THEFT'].sort_values(['Date'])
arson = data[data.Category == 'ARSON'].sort_values(['Date'])
arson.head()


# In[58]:


TOOLS = 'crosshair,save,pan,box_zoom,reset,wheel_zoom'
p = figure(title="Category-wise crimes through Time", y_axis_type="linear",x_axis_type='datetime', tools = TOOLS)

p.line(burglary['Date'], burglary.IncidntNum, legend="burglary", line_color="purple", line_width = 3)
p.line(stolen_property['Date'], stolen_property.IncidntNum, legend="stolen_property", line_color="blue", line_width = 3)

p.line(vehicle_theft['Date'], vehicle_theft.IncidntNum, legend="vehicle_theft", line_color = 'coral', line_width = 3)

p.line(larceny['Date'], larceny.IncidntNum, legend="larceny", line_color='green', line_width = 3)

p.line(vandalism['Date'], vandalism.IncidntNum, legend="vandalism", line_color="gold", line_width = 3)

p.line(arson['Date'], arson.IncidntNum, legend="arson", line_color="magenta",line_width = 3)

p.legend.location = "top_left"

p.xaxis.axis_label = 'Year'
p.yaxis.axis_label = 'Count'

output_file("multiline_plot.html", title="Multi Line Plot")

show(p)  # open a browser


# In[ ]:





# In[ ]:




