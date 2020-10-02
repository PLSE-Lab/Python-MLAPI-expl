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
# ## Bar plots
# In a [previous notebook](https://www.kaggle.com/meetnaren/plotly-experiments-scatterplots), I had explored how to use scatterplots to examine the relationship between two continuous variables. Here, I will explore bar plots.
# 
# Bar (or column) plots are useful to investigate a numerical quantity across different categories. There are different variations of bar plots - individual, clustered, stacked, etc. Let us try different flavors of bar plots in this notebook.
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


# In[ ]:


import colorlover as cl
from IPython.display import HTML

chosen_colors=cl.scales['5']['qual']['Paired']
print('The color palette chosen for this notebook is:')
HTML(cl.to_html(chosen_colors))


# ## Stations
# Let us examine the stations dataset to see some basic column charts.

# In[ ]:


station.head()


# Which cities have the highest number of stations and capacity (docks)?

# In[ ]:


citygroup=station.groupby(['city'])
temp_df1=citygroup['id'].count().reset_index().sort_values(by='id', ascending=False)
temp_df2=citygroup['dock_count'].sum().reset_index().sort_values(by='dock_count', ascending=False)


# In[ ]:


trace1 = go.Bar(
    x=temp_df1.city,
    y=temp_df1.id,
    name='No. of stations',
    text=temp_df1.id,
    textposition='outside',
    marker=dict(
        color=chosen_colors[0]
    )
)

data=[trace1]

layout = go.Layout(
    title='No. of bike stations in Bay Area cities',
    xaxis=dict(
        title='City'
    ),
    yaxis=dict(
        title='No. of bike stations'
    ),
    #hovermode='closest',
    #showlegend=True
)

figure = go.Figure(data=data, layout=layout)

ply.iplot(figure)


# The above chart can also be depicted through a 'waterfall' chart. Let us see how.

# In[ ]:


trace2 = go.Bar(
    x=temp_df1.city,
    y=temp_df1.id.cumsum().shift(1),
    #name='No. of stations',
    hoverinfo=None,
    marker=dict(
        color='rgba(1,1,1,0.0)'
    )
)
trace1 = go.Bar(
    x=temp_df1.city,
    y=temp_df1.id,
    name='No. of stations',
    text=temp_df1.id,
    textposition='outside',
    marker=dict(
        color=chosen_colors[0]
    )
)


data=[trace2, trace1]

layout = go.Layout(
    title='No. of bike stations in Bay Area cities',
    xaxis=dict(
        title='City'
    ),
    yaxis=dict(
        title='No. of bike stations'
    ),
    barmode='stack',
    hovermode='closest',
    showlegend=False
)

figure = go.Figure(data=data, layout=layout)

ply.iplot(figure)


# Let us look at a "clustered" column chart, with no. of stations and docks for each city grouped together.

# In[ ]:


trace2 = go.Bar(
    x=temp_df2.city,
    y=temp_df2.dock_count,
    name='No. of docks',
    text=temp_df2.dock_count,
    textposition='auto',
    marker=dict(
        color=chosen_colors[1]
    )
)

data=[trace1, trace2]

figure = go.Figure(data=data, layout=layout)

figure['layout'].update(dict(title='No. of bike stations and docks in Bay Area cities'), barmode='group')

ply.iplot(figure)


# ## Trips
# Let us move on to the trips dataset. This contains information about the bike trips taken from and to the various stations that we saw above.

# In[ ]:


trip.head()


# The start and end date columns contain date and time information that could be useful for our analysis. Let us extract more information from these columns. [Jeremy Howard](https://www.kaggle.com/jhoward)'s Fast AI library has a very useful function called add_datepart ([see here](https://github.com/fastai/fastai/blob/master/fastai/structured.py)). Let us use this function to add more info to this dataframe.

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
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
            'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
    if time: attr = attr + ['Hour', 'Minute', 'Second']
    for n in attr: df[targ_pre + n] = getattr(fld.dt, n.lower())
    df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9
    if drop: df.drop(fldname, axis=1, inplace=True)


# In[ ]:


add_datepart(trip, 'start_date', drop=False, time=True)


# In[ ]:


add_datepart(trip, 'end_date', drop=False, time=True)


# Let us ask some questions to answer through our visualizations.
# - What is the distribution of duration of trips?
# - What are the popular months / days / hours among bike renters?
# - Which bike stations are the most popular?
# - How does subscription type affect these parameters?
# 
# We'll see charts answering each question aove, along with variations caused by the difference in subscription types.
# 
# ## Duration distribution
# Although histograms are strictly not column charts, they depict magnitude through columns and so I will include histograms in this notebook.

# In[ ]:


trip['duration_min']=trip.duration/60
trace1 = go.Histogram(
    x=trip[trip.duration_min<60].duration_min, #To remove outliers
    marker=dict(
        color=chosen_colors[0]
    )    
)

data=[trace1]

layout = go.Layout(
    title='Distribution of bike trip duration in Bay Area',
    xaxis=dict(
        title='Trip Duration (minutes)'
    ),
    yaxis=dict(
        title='Count'
    ),
    #hovermode='closest',
    #showlegend=True
)

figure = go.Figure(data=data, layout=layout)

ply.iplot(figure)


# Now let us split the histogram by subscription type, and see if the trip duration varies between customers and subscribers.

# In[ ]:


data=[]

trace_names=['Subscriber', 'Customer']

for i in range(2):
    data.append(
        go.Histogram(
            x=trip[(trip.subscription_type==trace_names[i]) & (trip.duration_min<60)].duration_min,
            name=trace_names[i],
            marker=dict(
                color=chosen_colors[i]
            ),
            opacity=0.5
        )
    )

layout = go.Layout(
    title='Distribution of bike trip duration in Bay Area',
    barmode='overlay',
    xaxis=dict(
        title='Trip Duration (minutes)'
    ),
    yaxis=dict(
        title='Count'
    ),
    #hovermode='closest',
    #showlegend=True
)

figure = go.Figure(data=data, layout=layout)

ply.iplot(figure)


# It is clear that customers tend to use bikes for longer duration than do subscribers! Given below is a variation of the same plot, drawn as subplots instead of overlay.

# In[ ]:


trace_names=['Subscriber', 'Customer']

figure=make_subplots(rows=2, cols=1, subplot_titles = ['Trip duration (minutes) - '+i for i in trace_names])


for i in range(2):
    figure.append_trace(
        go.Histogram(
            x=trip[(trip.subscription_type==trace_names[i]) & (trip.duration_min<60)].duration_min,
            name=trace_names[i],
            showlegend=False,
            marker=dict(
                color=chosen_colors[i]
            )
        ),
        i+1, 1
    )

figure['layout'].update(
    height=1000,
    title='Distribution of trip duration by subscription type', 
    xaxis1=dict(title='Duration'),
    xaxis2=dict(title='Duration'),
    yaxis1=dict(title='Count'),
    yaxis2=dict(title='Count'),
)

ply.iplot(figure)


# ## Popular times for bike trips
# When do people take bike trips mostly? Which hours of the day, days of the week and months of the year are the most popular among bikers? And how does that vary between customers and subscribers? Let us explore.

# ### Popular months of the year
# Let us start with plotting the no. of trips by month.

# In[ ]:


trip_count_by_month=trip.groupby(['start_Year','start_Month'])['id'].count().reset_index()


# In[ ]:


trace1 = go.Bar(
    x=trip_count_by_month.start_Month.astype(str)+'-'+trip_count_by_month.start_Year.astype(str),
    y=trip_count_by_month.id,
    name='No. of trips',
    marker=dict(
        color=chosen_colors[0]
    )
)

data=[trace1]

layout = go.Layout(
    title='No. of bike trips by month',
    xaxis=dict(
        title='Month'
    ),
    yaxis=dict(
        title='No. of bike trips'
    ),
    #hovermode='closest',
    #showlegend=True
)

figure = go.Figure(data=data, layout=layout)

ply.iplot(figure)


# Does subscription type differ by month? Let us see.

# In[ ]:


trip_count_by_month_sub=trip.groupby(['start_Year','start_Month', 'subscription_type'])['id'].count().reset_index()


# In[ ]:


data=[]

trace_names=['Subscriber', 'Customer']

for i in range(2):
    temp_df=trip_count_by_month_sub[(trip_count_by_month_sub.subscription_type==trace_names[i])]
    data.append(
        go.Bar(
            x=temp_df.start_Month.astype(str)+'-'+temp_df.start_Year.astype(str),
            y=temp_df.id,
            name=trace_names[i],
            marker=dict(
                color=chosen_colors[i]
            )
        )
    )

layout = go.Layout(
    title='No. of trips per month in Bay Area cities',
    xaxis=dict(
        title='Month'
    ),
    yaxis=dict(
        title='No. of trips'
    ),
    barmode='stack'
    #hovermode='closest',
    #showlegend=True
)

figure = go.Figure(data=data, layout=layout)

ply.iplot(figure)


# Overall, the month-wise analysis of the no. of trips shows us that ridership tends to be low in the winter months and it tends to gradually increase from spring to summer and fall. The proportion of customer to subscriber does not seem to change much based on the month. Let us now see how the day of week affects ridership.

# ### Popular days of the week
# On which days of the week are bike trips higher? Does the trend vary by weekday and weekend? Let us explore. Since the dataset is bigger, I will focus our analysis on the last three months of 2013 only.

# In[ ]:


trip['start_date_dt']=[i.date() for i in trip.start_date]
trip['end_date_dt']=[i.date() for i in trip.end_date]


# In[ ]:


trip_count_by_date=trip.groupby(['start_date_dt'])['id'].count().reset_index()

trip_count_by_date['day_of_week']=[i.weekday() for i in trip_count_by_date.start_date_dt]

trip_count_by_date['is_weekend'] = (trip_count_by_date.day_of_week>4)*1


# In[ ]:


data=[]

trace_names=['Weekday', 'Weekend']

for i in range(2):
    data.append(
        go.Bar(
            x=trip_count_by_date[(trip_count_by_date.is_weekend==i) & (trip_count_by_date.start_date_dt<datetime.date(2014, 1, 1))].start_date_dt,
            y=trip_count_by_date[(trip_count_by_date.is_weekend==i)  & (trip_count_by_date.start_date_dt<datetime.date(2014, 1, 1))].id,
            name=trace_names[i],
            marker=dict(
                color=chosen_colors[i]
            )
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
    #hovermode='closest',
    #showlegend=True
)

figure = go.Figure(data=data, layout=layout)

ply.iplot(figure)


# Clearly, the usage is lower on the weekends than it is on the weekdays. How does subscription type differ on these days?

# In[ ]:


trip_count_by_date_sub=trip.groupby(['start_date_dt', 'subscription_type'])['id'].count().reset_index()

trip_count_by_date_sub['day_of_week']=[i.weekday() for i in trip_count_by_date_sub.start_date_dt]

trip_count_by_date_sub['is_weekend'] = (trip_count_by_date_sub.day_of_week>4)*1


# In[ ]:


data=[]

trace_names=['Subscriber', 'Customer']

for i in range(2):
    data.append(
        go.Bar(
            x=trip_count_by_date_sub[(trip_count_by_date_sub.subscription_type==trace_names[i]) & (trip_count_by_date_sub.start_date_dt<datetime.date(2014, 1, 1))].start_date_dt,
            y=trip_count_by_date_sub[(trip_count_by_date_sub.subscription_type==trace_names[i]) & (trip_count_by_date_sub.start_date_dt<datetime.date(2014, 1, 1))].id,
            name=trace_names[i],
            marker=dict(
                color=chosen_colors[i]
            )
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
    barmode='stack'
    #hovermode='closest',
    #showlegend=True
)

figure = go.Figure(data=data, layout=layout)

ply.iplot(figure)


# It looks like the subcriber usage is higher on weekdays, and lower on weekends. I think it would be better if we plot the percentage numbers in a stacked column than the absolute numbers. We can accomplish it this way:

# In[ ]:


def calc_percent(row, col):
    dt=row[col]
    total=trip_count_by_date[trip_count_by_date[col]==dt].id.iloc[0]
    count=row['id']
    
    return count*1./total*100


# In[ ]:


trip_count_by_date_sub['percent']=trip_count_by_date_sub.apply(lambda row: calc_percent(row, 'start_date_dt'), axis=1)


# In[ ]:


data=[]

trace_names=['Subscriber', 'Customer']

for i in range(2):
    data.append(
        go.Bar(
            x=trip_count_by_date_sub[(trip_count_by_date_sub.subscription_type==trace_names[i]) & (trip_count_by_date_sub.start_date_dt<datetime.date(2014, 1, 1))].start_date_dt,
            y=trip_count_by_date_sub[(trip_count_by_date_sub.subscription_type==trace_names[i]) & (trip_count_by_date_sub.start_date_dt<datetime.date(2014, 1, 1))].percent,
            name=trace_names[i],
            marker=dict(
                color=chosen_colors[i]
            )
        )
    )

layout = go.Layout(
    title='Percentage of trips per day in Bay Area cities',
    xaxis=dict(
        title='Date'
    ),
    yaxis=dict(
        title='% of trips',
        ticksuffix='%'
    ),
    barmode='stack'
    #hovermode='closest',
    #showlegend=True
)

figure = go.Figure(data=data, layout=layout)

ply.iplot(figure)


# Let us see if we can highlight the weekends also, along with the subscription type.

# In[ ]:


data=[]

trace_names=['Subscriber', 'Customer']
trace_names1=['weekdays', 'weekends']

weekend=[0,1]

for i in range(2):
    for j in range(2):
        data.append(
            go.Bar(
                x=trip_count_by_date_sub[(trip_count_by_date_sub.subscription_type==trace_names[i]) & 
                                         (trip_count_by_date_sub.is_weekend==weekend[j]) &
                                         (trip_count_by_date_sub.start_date_dt<datetime.date(2014, 1, 1))].start_date_dt,
                y=trip_count_by_date_sub[(trip_count_by_date_sub.subscription_type==trace_names[i]) & 
                                         (trip_count_by_date_sub.is_weekend==weekend[j]) &
                                         (trip_count_by_date_sub.start_date_dt<datetime.date(2014, 1, 1))].percent,
                name=trace_names[i]+' on '+trace_names1[j],
                marker=dict(
                    color=chosen_colors[i*2+j]
                )
            )
        )

layout = go.Layout(
    title='Percentage of trips per day in Bay Area cities',
    xaxis=dict(
        title='Date'
    ),
    yaxis=dict(
        title='% of trips',
        ticksuffix='%'
    ),
    barmode='stack',
    #hovermode='closest',
    legend=dict(
        orientation="h",
        x=0,
        y=1.1
    )
)

figure = go.Figure(data=data, layout=layout)

ply.iplot(figure)


# We can now clearly see that while the proportion of subscriber count is higher during the weekdays, it drops down significantly during the weekends. Customer tend to use bikes more during the weekends than on weekdays.
# 
# Let us now see a **summary** of the trips on ALL dates in the dataset by day of week.

# In[ ]:


trip_count_by_DOW=trip.groupby(['start_Dayofweek'])['id'].count().reset_index()

DOW=[
    'Monday',
    'Tuesday',
    'Wednesday',
    'Thursday',
    'Friday',
    'Saturday',
    'Sunday'
]

data=[
    go.Bar(
        x=DOW,
        y=trip_count_by_DOW['id'],
        name='No. of trips',
        marker=dict(
            color=chosen_colors[0]
        )
    )
]

layout = go.Layout(
    title='No. of bike trips by day of week in Bay Area cities',
    xaxis=dict(
        title='Day of week'
    ),
    yaxis=dict(
        title='No. of trips'
    ),
    #hovermode='closest',
    barmode='stack'
)

figure = go.Figure(data=data, layout=layout)

ply.iplot(figure)


# Again, splitting this plot by subscription type:

# In[ ]:


trip_count_by_DOW_sub=trip.groupby(['start_Dayofweek','subscription_type'])['id'].count().reset_index()

data=[]

trace_names=['Subscriber', 'Customer']

for i in range(2):
    data.append(
        go.Bar(
            x=DOW,
            y=trip_count_by_DOW_sub[(trip_count_by_DOW_sub.subscription_type==trace_names[i])].id,
            name=trace_names[i],
            marker=dict(
                color=chosen_colors[i]
            )
        )
    )

layout = go.Layout(
    title='No. of bike trips by day of week in Bay Area cities',
    xaxis=dict(
        title='Day of week'
    ),
    yaxis=dict(
        title='No. of trips'
    ),
    barmode='stack'
)

figure = go.Figure(data=data, layout=layout)

ply.iplot(figure)


# And again, plotting percentages instead of absolutes:

# In[ ]:


def calc_percent1(row, col):
    dt=row[col]
    total=trip_count_by_DOW[trip_count_by_DOW[col]==dt].id.iloc[0]
    count=row['id']
    
    return count*1./total*100


# In[ ]:


trip_count_by_DOW_sub['percent']=trip_count_by_DOW_sub.apply(lambda row:calc_percent1(row, 'start_Dayofweek'), axis=1)


# In[ ]:


data=[]

trace_names=['Subscriber', 'Customer']

for i in range(2):
    data.append(
        go.Bar(
            x=DOW,
            y=trip_count_by_DOW_sub[trip_count_by_DOW_sub.subscription_type==trace_names[i]].percent,
            name=trace_names[i],
            marker=dict(
                color=chosen_colors[i]
            )
        )
    )

layout = go.Layout(
    title='Percentage of trips by day of week in Bay Area cities',
    xaxis=dict(
        title='Day of week',
    ),
    yaxis=dict(
        title='% of trips',
        ticksuffix='%'
    ),
    barmode='stack'
)

figure = go.Figure(data=data, layout=layout)

ply.iplot(figure)


# We can deduce the following from the analysis of the no. of trips on different days of the week:
# - Subscribers mostly tend to use the bikes during weekdays. This indicates that they may be using it for commuting to and from work (we can confirm this later when we do the analysis by hour)
# - Customers mostely tend to use the bikes during weekends and holidays (if you scroll up a bit, you can see that customer usage was higher on Christmas than that of subscribers, even though it was a weekday)

# ### Popular hours of the day
# Let us now see during which hours of the day bikes are highly used. Since this is a lot of data points, I will examine just 1 week worth of data, say the 1st week of Dec 2013.

# In[ ]:


trip_count_by_hour=trip.groupby(['start_date_dt','start_Hour'])['id'].count().reset_index()


# In[ ]:


temp_df=trip_count_by_hour[(trip_count_by_hour.start_date_dt>datetime.date(2013, 11, 30)) & (trip_count_by_hour.start_date_dt<datetime.date(2013, 12, 8))]
trace1 = go.Bar(
    x=temp_df.start_Hour.astype(str)+'                     '+temp_df.start_date_dt.astype(str),
    #x=temp_df.start_Hour.astype(str),
    y=temp_df.id,
    text=['Hour:'+str(i) for i in temp_df.start_Hour],
    name='No. of trips',
    marker=dict(
        color=chosen_colors[0]
    )
)

data=[trace1]

layout = go.Layout(
    title='No. of bike trips by hour',
    xaxis=dict(
        title='Hour',
        categoryorder='array',
        categoryarray=temp_df.start_Hour,
        type='category'
    ),
    yaxis=dict(
        title='No. of bike trips'
    ),
)

figure = go.Figure(data=data, layout=layout)

ply.iplot(figure)


# The number of trips tends to be higher during weekdays, especially during the morning and evening (8AM and 5PM). We already saw the subscribers use the bikes mostly during weekdays. This confirms our assumption that subscribers are people who use it mostly for their daily commute to and from work.
# 
# Let us now see a summary of trips by hour across the whole dataset.

# In[ ]:


temp_df=trip.groupby(['start_Hour'])['id'].count().reset_index()

trace1 = go.Bar(
    x=temp_df.start_Hour,
    y=temp_df.id,
    name='No. of trips',
    marker=dict(
        color=chosen_colors[0]
    )
)

data=[trace1]

layout = go.Layout(
    title='No. of bike trips by hour',
    xaxis=dict(
        title='Hour',
    ),
    yaxis=dict(
        title='No. of bike trips'
    ),
    barmode='stack'
)

figure = go.Figure(data=data, layout=layout)

ply.iplot(figure)


# Let us check if subscription type differs by hour.

# In[ ]:


trip_count_by_hour_sub=trip.groupby(['start_Hour','subscription_type'])['id'].count().reset_index()


# In[ ]:


#temp_df=trip_count_by_hour_sub[(trip_count_by_hour_sub.start_date_dt>datetime.date(2013, 11, 30)) & (trip_count_by_hour_sub.start_date_dt<datetime.date(2013, 12, 8))]
temp_df=trip_count_by_hour_sub.copy()
data=[]

trace_names=['Subscriber', 'Customer']

for i in range(2):
    temp_df1=temp_df[(temp_df.subscription_type==trace_names[i])]
    data.append(
        go.Bar(
            x=temp_df1.start_Hour,
            y=temp_df1.id,
            name=trace_names[i],
            marker=dict(
                color=chosen_colors[i]
            )
        )
    )

layout = go.Layout(
    title='No. of trips per hour in Bay Area cities',
    xaxis=dict(
        title='Hour'
    ),
    yaxis=dict(
        title='No. of trips'
    ),
    barmode='stack'
)

figure = go.Figure(data=data, layout=layout)

ply.iplot(figure)


# It looks like the proportion of subscription type does not differ significantly between different hours of the day.

# ## Popular bike stations
# Where to bike trips start and end? Are some bike stations more popular than the others? One would think that stations near the city center would be more popular. Let us investigate.

# In[ ]:


trip_count_start_station=trip.groupby(['start_station_name']).id.count().reset_index().sort_values(by='id', ascending=False)
trip_count_end_station=trip.groupby(['end_station_name']).id.count().reset_index().sort_values(by='id', ascending=False)


# In[ ]:


trace1 = go.Bar(
    x=trip_count_start_station[:10].start_station_name,
    y=trip_count_start_station[:10].id,
    name='No. of trips starting',
    marker=dict(
        color=chosen_colors[0]
    )
)

trace2 = go.Bar(
    x=trip_count_start_station[:10].start_station_name,
    y=trip_count_end_station[trip_count_end_station.end_station_name.isin(trip_count_start_station[:10].start_station_name)].id,
    name='No. of trips ending',
    marker=dict(
        color=chosen_colors[1]
    )
)

data=[trace1, trace2]

layout = go.Layout(
    title='No. of bike trips by starting station',
    xaxis=dict(
        title='Station name',
    ),
    yaxis=dict(
        title='No. of bike trips'
    )
)

figure = go.Figure(data=data, layout=layout)

ply.iplot(figure)


# The x-axis labels are getting truncated as they are quite long. This makes a good situation to use a horizontal bar chart instead of a vertical one.

# In[ ]:


trace1 = go.Bar(
    x=trip_count_start_station[:10].id,
    y=trip_count_start_station[:10].start_station_name,
    orientation='h',
    name='No. of trips starting',
    marker=dict(
        color=chosen_colors[0]
    )
)

trace2 = go.Bar(
    x=trip_count_end_station[trip_count_end_station.end_station_name.isin(trip_count_start_station[:10].start_station_name)].id,
    y=trip_count_start_station[:10].start_station_name,
    orientation='h',
    name='No. of trips ending',
    marker=dict(
        color=chosen_colors[1]
    )
)

data=[trace1, trace2]

layout = go.Layout(
    title='No. of bike trips by starting station',
    yaxis=dict(
        title='Station name',
    ),
    xaxis=dict(
        title='No. of bike trips'
    ),
    margin=dict(
        l=350
    ),
    legend=dict(
        orientation='h',
        x=0,
        y=1.1
    )
)

figure = go.Figure(data=data, layout=layout)

ply.iplot(figure)


# Since we saw that most of the bike trips were used for commuting to and from work, the stations with the highest number of trips (starting and ending) seem to be stations for boarding on and off public transport, such as a train station or a ferry terminal.

# # Conclusion
# I hope this notebook helped you understand the different flavors of bar plots in Plotly. The dataset has so much potential for more powerful visualizations. Maybe I will explore them in a different notebook later. I will try exploring line plots in a subsequent notebook. Thanks!
