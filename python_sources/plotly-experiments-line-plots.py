#!/usr/bin/env python
# coding: utf-8

# # Plotly
# Plotly is one of my favorite data visualization packages for Python. The wide variety of plots and the level of customization available give the users a high amount of control on how the chart looks. As I learn more about how to work with Plotly, I want to experiment with different chart types through kernels as a way for me to practice and also for the Kaggle community to know how to use them.
# 
# ## Notes about Plotly
# Plotly charts have two major components: data and layout.
# 
# Data - this represents the data that we are trying to plot. This informs the Plotly's plotting function of the type of plots that need to be drawn. It is basically a list of plots that should be part of the chart. Each plot within the chart is referred to as a 'trace'.
# 
# Layout - this represents everything in the chart that is not data. This means the background, grids, axes, titles, fonts, etc. We can even add shapes on top of the chart and annotations to highlight certain points to the user.
# 
# The data and layout are then passed to a "figure" object, which is in turn passed to the plot function in Plotly.
# 
# - Figure
#     - Data
#         - Traces
#     - Layout
#         - Layout options
#         
# ## Line plots
# In previous notebooks [here](https://www.kaggle.com/meetnaren/plotly-experiments-scatterplots) and [here](https://www.kaggle.com/meetnaren/plotly-experiments-bar-column-plots), I had explored how to use scatterplots to examine the relationship between two continuous variables, and bar plots to explore numerical variable across different categories. In this notebook, I will explore line plots.
# 
# Line plots are generally useful to investigate the trend of a numerical variable over time. Any time series analysis is incomplete without a line chart. There are various flavors of line charts - with and without markers, area charts, stepped line charts, linear and smoothed lines, etc. Let us explore these in this notebook.
# 
# In Plotly, line charts are just a variation of scatterplots, only with a line connecting the dots. Thus, we will be using the scatter (or scattergl) function for plotting purposes.
# 
# ## Dataset
# I will be using the Bay Area Bike Share dataset for this notebook. The data has very interesting information on the bikes, stations and bike trips taken in the Bay Area. There are a lot of questions that we can explore through visualizations. Let us read in the dataset and import Plotly packages to start with.

# In[ ]:


import plotly.offline as ply
import plotly.graph_objs as go
from plotly.tools import make_subplots

ply.init_notebook_mode(connected=True)

import pandas as pd
import numpy as np

station = pd.read_csv('../input/station.csv')
trip = pd.read_csv('../input/trip.csv')
weather = pd.read_csv('../input/weather.csv')


# Using a random function to choose 5 colors to use for this notebook...

# In[ ]:


import colorlover as cl
from IPython.display import HTML

chosen_colors=cl.scales['7']['qual'][np.random.choice(list(cl.scales['7']['qual'].keys()))]

print('The color palette chosen for this notebook is:')
HTML(cl.to_html(chosen_colors))


# ## Basic line charts
# Let us start with some basic line charts. The 'trip' dataset seems to have time series data, as can be seen below:

# In[ ]:


trip.head()


# Let us plot a simple line chart that shows the no. of trips taken per day over the span of dates in the dataset. This needs some data manipulation to extract the date portion from the start_date and end_date columns.

# In[ ]:


trip.start_date=pd.to_datetime(trip.start_date,infer_datetime_format=True)
trip.end_date=pd.to_datetime(trip.end_date,infer_datetime_format=True)


# In[ ]:


#For some reason, the add_datepart function that I imported through fastai library in Kaggle does not have the 'time' argument which also extracts the time details such as the hour and minute
#Hence, I am copying the code from Github and pasting it here for using in this notebook.
import re
import datetime
def add_datepart(df, fldname, drop=True, time=False):
    fld = df[fldname]
    fld_dtype = fld.dtype
    if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        fld_dtype = np.datetime64

    if not np.issubdtype(fld_dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)
    targ_pre = re.sub('[Dd]ate$', '', fldname)
    attr = ['Year', 'Month', 'Week', 'Day', 'Date', 'Dayofweek', 'Dayofyear',
            'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
    if time: attr = attr + ['Hour', 'Minute', 'Second']
    for n in attr: df[targ_pre + n] = getattr(fld.dt, n.lower())
    df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9
    if drop: df.drop(fldname, axis=1, inplace=True)


# In[ ]:


add_datepart(trip, 'start_date', drop=False, time=True)


# In[ ]:


add_datepart(trip, 'end_date', drop=False, time=True)


# In[ ]:


trip.head()


# Now that we have the individual components of the date column split up, let us calculate the numbers required for plotting the line chart.

# In[ ]:


trip_count_by_date=trip.groupby(['start_Date'])['id'].count().reset_index()


# In[ ]:


trace1 = go.Scatter(
    x=trip_count_by_date.start_Date,
    y=trip_count_by_date.id,
    mode='lines',
    line=dict(
        color=chosen_colors[0]
    ),
    name='Daily'
)

data=[trace1]

layout = go.Layout(
    title='No. of trips per day in Bay Area cities',
    xaxis=dict(
        title='Date',
        type='date',
        showgrid=False
    ),
    yaxis=dict(
        title='No. of trips'
    ),
    hovermode='closest',
)

figure = go.Figure(data=data, layout=layout)

ply.iplot(figure)


# In[ ]:


layout


# The chart looks too crowded, as we have tried to cram in a wide time period into one chart. Thankfully, Plotly provides a very handy tool called the rangeslider, which will enable the user to just select a specific timeframe very easily. Let's see how.

# In[ ]:


figure['layout'].update(
    xaxis=dict(
        rangeslider=dict(
            visible = True
        ),
        type='date'
    )
)

ply.iplot(figure)


# The 'smaller' graph that you can see below the x-axis is called the range slider. You can click and drag the sliders to zoom in on a specific timeframe.
# 
# One could see that the ridership tends to run low in winter, especially towards the end of the year. Let us plot some moving averages to smooth out the curve and see the pattern.

# In[ ]:


trip_count_by_date['weekly_avg']=trip_count_by_date.rolling(window=7, center=True)['id'].mean()
trip_count_by_date['monthly_avg']=trip_count_by_date.rolling(window=30, center=True)['id'].mean()
trip_count_by_date['quarterly_avg']=trip_count_by_date.rolling(window=90, center=True)['id'].mean()

trace2 = go.Scatter(
    x=trip_count_by_date.start_Date,
    y=trip_count_by_date.weekly_avg,
    mode='lines',
    line=dict(
        color=chosen_colors[1],
    ),
    name='Weekly'
)

trace3 = go.Scatter(
    x=trip_count_by_date.start_Date,
    y=trip_count_by_date.monthly_avg,
    mode='lines',
    line=dict(
        color=chosen_colors[3],
    ),
    name='Monthly'
)

trace4 = go.Scatter(
    x=trip_count_by_date.start_Date,
    y=trip_count_by_date.quarterly_avg,
    mode='lines',
    line=dict(
        color=chosen_colors[5],
    ),
    name='Quarterly'
)

data=[trace1, trace2, trace3, trace4]

figure = go.Figure(data=data, layout=layout)

figure['layout'].update(
    legend=dict(
        orientation="h",
        x=0,
        y=1.1
    ),
    xaxis=dict(
        rangeslider=dict(
            visible = True
        ),
        type='date',
    ),
)

ply.iplot(figure)


# The moving average curves clearly show how the ridership goes down during the winter months.
# 
# Let us now zoom into a specific window of the timeframe and do some analysis. I will focus on Q2 2014.

# In[ ]:


trip_count_q2_2014=trip[(trip.start_Date>=datetime.date(2014,4,1)) & (trip.start_Date<datetime.date(2014,7,1))].groupby(['start_Date'])['id'].count().reset_index()

trace1 = go.Scatter(
    x=trip_count_q2_2014.start_Date,
    y=trip_count_q2_2014.id,
    mode='markers+lines',
    line=dict(
        color=chosen_colors[0]
    ),
    marker=dict(
        color=chosen_colors[1],
    )
)

data=[trace1]

figure = go.Figure(data=data, layout=layout)

ply.iplot(figure)


# We can see that there is a weekly, cyclic pattern to the data. In Time Series analysis, this is referred to as 'seasonality'. Let us color the markers based on the day of week to see if we can discern this pattern.

# In[ ]:


trip_count_q2_2014['day_of_week']=[i.weekday() for i in trip_count_q2_2014.start_Date]

trip_count_q2_2014['is_weekend'] = (trip_count_q2_2014.day_of_week>4)*1

trace1 = go.Scatter(
    x=trip_count_q2_2014.start_Date,
    y=trip_count_q2_2014.id,
    mode='lines',
    line=dict(
        color=chosen_colors[-1]
    ),
    showlegend=False
)

data=[trace1]

trace_names=['Weekday', 'Weekend']

for i in range(2):
    data.append(
        go.Scatter(
            x=trip_count_q2_2014[trip_count_q2_2014.is_weekend==i].start_Date,
            y=trip_count_q2_2014[trip_count_q2_2014.is_weekend==i].id,
            mode='markers',
            marker=dict(
                color=chosen_colors[i]
            ),
            name=trace_names[i]
        )
    )

figure = go.Figure(data=data, layout=layout)

ply.iplot(figure)


# This chart clearly indicates that the ridership goes down on weekends. Why was ridership low on May 26, 2014, though? The answer is [here](https://www.timeanddate.com/holidays/us/memorial-day).
# 
# Another way to highlight the weekends, is to draw boxes through shapes in Plotly. Let's see how.

# In[ ]:


trace1 = go.Scatter(
    x=trip_count_q2_2014.start_Date,
    y=trip_count_q2_2014.id,
    mode='markers+lines',
)

shapes=[]
weekend_dates=trip_count_q2_2014[trip_count_q2_2014.is_weekend==1].start_Date

box_color=chosen_colors[-1]

for i in weekend_dates[::2]:
    shapes.append(
        {
            'type':'rect',
            'xref':'x',
            'x0':i,
            'x1':i+datetime.timedelta(days=1),
            'yref':'paper',
            'y0':0,
            'y1':1,
            'fillcolor': box_color,
            'opacity':0.15,
            'line': {
                'width': 0,
            }
        }
    )

data=[trace1]

figure = go.Figure(data=data, layout=layout)

figure['layout'].update(
    shapes=shapes
)

ply.iplot(figure)


# This chart looks cleaner and simpler, showing that ridership is lower on weekends with minimal color.
# 
# Let us move on to area charts now.

# ## Area charts
# Area charts differ from line charts in that the area under the line gets shaded, giving the viewer a sense of the magnitude of the number being plotted. For example, it may be more appropriate to plot stock price through a line chart, but market cap through an area chart. Area charts are also useful when used in a stacked model, showing the difference between two numerical quantities in a shaded area.
# 
# Let us see the ridership count as an area chart to start with.

# In[ ]:


trace1 = go.Scatter(
    x=trip_count_q2_2014.start_Date,
    y=trip_count_q2_2014.id,
    mode='lines',
    line=dict(
        width=0.0,
        color=chosen_colors[0]
    ),
    fill='tozeroy',
)

data=[trace1]

figure = go.Figure(data=data, layout=layout)

ply.iplot(figure)


# Let us view the split between customer and subscriber and plot the difference on an area chart.

# In[ ]:


trip_count_q2_2014_sub=trip[(trip.start_Date>=datetime.date(2014,4,1)) & (trip.start_Date<datetime.date(2014,7,1))].groupby(['start_Date', 'subscription_type'])['id'].count().reset_index()

data=[]

trace_names=['Subscriber', 'Customer']
fillmodes=[None, 'tonexty']

for i in range(2):
    data.append(
        go.Scatter(
            x=trip_count_q2_2014_sub[trip_count_q2_2014_sub.subscription_type==trace_names[i]].start_Date,
            y=trip_count_q2_2014_sub[trip_count_q2_2014_sub.subscription_type==trace_names[i]].id,
            name=trace_names[i],
            line=dict(
                width=0.0,
                color=chosen_colors[i]
            ),
            fill='tozeroy',
        )
    )

layout = go.Layout(
    title='No. of trips per day in Bay Area cities',
    xaxis=dict(
        title='Date'
    ),
    yaxis=dict(
        title='No. of trips'
    ),
    legend=dict(
        orientation='h',
        x=0,
        y=1.1
    )
    #hovermode='closest',
    #showlegend=True
)

figure = go.Figure(data=data, layout=layout)

ply.iplot(figure)


# We can also stack these two categories on top of each other. Let's see how.

# In[ ]:


def calc_total(row):
    if row.subscription_type=='Customer':
        return trip_count_q2_2014_sub[(trip_count_q2_2014_sub.start_Date==row.start_Date) & (trip_count_q2_2014_sub.subscription_type=='Subscriber')].id.iloc[0]+row.id
    else:
        return row.id

trip_count_q2_2014_sub['total']=trip_count_q2_2014_sub.apply(lambda row:calc_total(row), axis=1)


# In[ ]:


data=[]

trace_names=['Subscriber', 'Customer']
fillmodes=['tozeroy', 'tonexty']

for i in range(2):
    data.append(
        go.Scatter(
            x=trip_count_q2_2014_sub[trip_count_q2_2014_sub.subscription_type==trace_names[i]].start_Date,
            y=trip_count_q2_2014_sub[trip_count_q2_2014_sub.subscription_type==trace_names[i]].total,
            name=trace_names[i],
            line=dict(
                color=chosen_colors[i],
                width=0.0
            ),
            fill=fillmodes[i],
            hoverinfo='text',
            text=[trace_names[i]+': '+k for k in trip_count_q2_2014_sub[trip_count_q2_2014_sub.subscription_type==trace_names[i]].id.astype(str)]
        )
    )

layout = go.Layout(
    title='No. of trips per day in Bay Area cities',
    xaxis=dict(
        title='Date',
        showgrid=False
    ),
    yaxis=dict(
        title='No. of trips'
    ),
    legend=dict(
        orientation='h',
        x=0,
        y=1.1
    )
)

figure = go.Figure(data=data, layout=layout)

ply.iplot(figure)


# To see only the customer trip counts stacked on top of the subscriber trip counts, we can do this:

# In[ ]:


fillmodes[0]=None
for i in range(2): figure['data'][i].update(fill=fillmodes[i])
ply.iplot(figure)


# ## Stepped line charts
# A stepped line chart connects points thorugh vertical and horizontal lines, instead of a straight line joining the two points. This gives the user a view of where there are sharp increases and decreases and where the numbers hold steady.
# 
# Let us try plotting a stepped line chart on the same data, the trip count in Q2 2014.

# In[ ]:


trace1 = go.Scatter(
    x=trip_count_q2_2014.start_Date,
    y=trip_count_q2_2014.id,
    mode='markers+lines',
    line=dict(
        color=chosen_colors[0],
        shape='hv'
    ),
    marker=dict(
        color=chosen_colors[1]
    )
)

data=[trace1]

figure = go.Figure(data=data, layout=layout)

ply.iplot(figure)


# ## Conclusion
# I hope this notebook helped you learn how to plot different types of line charts in Plotly. In a subsequent notebook, I will explore other types of charts such as bubble charts, box plots, etc. Thanks for reading!
