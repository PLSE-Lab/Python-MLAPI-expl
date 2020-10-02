#!/usr/bin/env python
# coding: utf-8

# # TPS | Data Analyst Job Interview Notes
# 
# Hello! Last year I had gone to a job interview for a Data Analyst for the [Toronto Police Services Data Analytics Division](http://data.torontopolice.on.ca/pages/data-analytics). One of the rounds of the interview required me to find insights from their Bicycle Thefts dataset. This notebook represents notes that I had made for that interview. This notebook contains a lot of code I wrote for data querying, visualization, and analysis that I found very useful, and I hope you can find it useful as well.
# 
# 
# If you enjoyed my content, please rememer to upvote the notebook. If you have any questions, feel free to comment below or message me on [LinkedIn](https://www.linkedin.com/in/jaydeep-mistry/)

# In[ ]:


import numpy as np
import pandas as pd
import geopandas as gpd

import datetime as dt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df_bicycle_thefts = gpd.read_file("/kaggle/input/TPS_Toronto_Bicycle_Thefts.geojson")


# In[ ]:


df_bicycle_thefts.info()


# In[ ]:


df_bicycle_thefts.sample(3)


# # Data Cleaning

# In[ ]:


# Convert the cost of bicycles from string representation to a float for numerical analysis

def clean_cost(value):
    if type(value) == str:
        return float(value)
    else:
        return value

df_bicycle_thefts.Cost_of_Bike = df_bicycle_thefts.Cost_of_Bike.apply(clean_cost)


# A major issue with this dataset is that the date and time are save as two separate variables. I personally prefer to combine them into one datetime object as shown below.

# In[ ]:




df_bicycle_thefts.Occurrence_Date = pd.to_datetime(df_bicycle_thefts.Occurrence_Date)
df_bicycle_thefts.Occurrence_Time = pd.to_datetime(df_bicycle_thefts.Occurrence_Time)
df_bicycle_thefts['Occurance_DateTime'] = dt.datetime.now()

for index, row in df_bicycle_thefts.iterrows():    
    df_bicycle_thefts.at[index,'Occurance_DateTime'] = dt.datetime(
        row['Occurrence_Date'].year,
        row['Occurrence_Date'].month,
        row['Occurrence_Date'].day,
        row['Occurrence_Time'].hour,
        row['Occurrence_Time'].minute,
        row['Occurrence_Time'].second
    )


# In[ ]:


df_bicycle_thefts = df_bicycle_thefts[[
    'event_unique_id',
    'Primary_Offence',       
    'Division',
    'City',
    'Location_Type',
    'Bike_Make',
    'Bike_Model',
    'Bike_Type',
    'Bike_Speed',
    'Bike_Colour',
    'Cost_of_Bike',
    'Status',
    'Lat',
    'Long',
    'Occurance_DateTime',
]]


# I prefer to add columns that store the computed values for the dates and time of all rows. This helps keep the code a bit cleaner instead of computing these values for every graph, and reduces the computation required

# In[ ]:


df_bicycle_thefts['Year'] = df_bicycle_thefts.Occurance_DateTime.apply(lambda x: x.year)
df_bicycle_thefts['Month'] = df_bicycle_thefts.Occurance_DateTime.apply(lambda x: x.month)
df_bicycle_thefts['Day'] = df_bicycle_thefts.Occurance_DateTime.apply(lambda x: x.day)
df_bicycle_thefts['Hour'] = df_bicycle_thefts.Occurance_DateTime.apply(lambda x: x.hour)
df_bicycle_thefts['Minute'] = df_bicycle_thefts.Occurance_DateTime.apply(lambda x: x.minute)
df_bicycle_thefts['Day_of_week'] = df_bicycle_thefts.Occurance_DateTime.apply(lambda x: x.isoweekday())
df_bicycle_thefts['Day_of_year'] = df_bicycle_thefts.Occurance_DateTime.apply(lambda x: x.toordinal() - dt.date(x.year, 1, 1).toordinal() + 1)
df_bicycle_thefts['Week_of_year'] = df_bicycle_thefts.Occurance_DateTime.apply(lambda x: x.isocalendar()[1])


# In[ ]:


# The final format of our dataframe

df_bicycle_thefts.iloc[0]


# # Data Visualization & Analysis

# This section contains graphs followed by insights that I noted for each graph

# In[ ]:


import plotly.express as px
import plotly.graph_objects as go


# In[ ]:


# making normalize to be True will show the percentage of the total for all traces instead of the tally the proportions
df_data = df_bicycle_thefts['Status'].value_counts(normalize = False)

fig = px.bar(
    data_frame = df_data,
    x = df_data.index,
    y = df_data
)
fig.update_layout(
    xaxis_title = 'Status of Bicycle Theft',
    yaxis_title = 'Proportion of Incidents',
    title = 'Proportion of Bicycles Thefts by their Stolen Status'
)
fig.show()


# Insight: Most stolen bicycles have not been recovered

# In[ ]:


fig = go.Figure()

for status in df_bicycle_thefts.Status.value_counts().sort_index().index:
    df_data = df_bicycle_thefts[df_bicycle_thefts.Status == status].Year.value_counts().sort_index()
    fig.add_trace(
        go.Scatter(
            x = df_data.index,
            y = df_data,
            mode = 'lines+markers',
            name = status
        )
    )
    
fig.update_layout(
    xaxis_title = 'Year',
    yaxis_title = 'Proportion of Incidents',
    title = 'Proportion of Bicycles Thefts in Toronto by Stolen Status and Year',
    yaxis_type = 'log'
)
fig.show()


# Insight: The number of Bicycle thefts have increased year over year, whereas the number of recovered bicycles has practically remained the same

# In[ ]:


df_data = df_bicycle_thefts['Division'].value_counts(normalize = False)

fig = px.bar(
    data_frame = df_data,
    x = df_data.index,
    y = df_data,    
)
fig.update_layout(
    xaxis_title = 'Division',
    yaxis_title = 'Proportion of Incidents',
    title = 'Proportion of Bicycles Thefts by the Division in Toronto they were Stolen in',
    xaxis_type = 'category'
)
fig.show()


# Insight: Certain Divisions report significantly more bicycle thefts than others, with a long tail distribution

# In[ ]:


df_data = df_bicycle_thefts['Location_Type'].value_counts(normalize = False).head(12)

fig = px.bar(
    data_frame = df_data,
    x = df_data.index,
    y = df_data,    
)
fig.update_layout(
    xaxis_title = 'Location Type',
    yaxis_title = 'Proportion of Incidents',
    title = 'Top 12 Types of Locations for Bicycle Thefts in Toronto',
    xaxis_type = 'category'
)
fig.show()


# Insight: Roads sides, Appartments, and Homes are where bicycle thefts happen the most

# In[ ]:


df_data = df_bicycle_thefts['Primary_Offence'].value_counts(normalize = False).head(12)

fig = px.bar(
    data_frame = df_data,
    x = df_data.index,
    y = df_data,    
)
fig.update_layout(
    xaxis_title = 'Offense Type',
    yaxis_title = 'Proportion of Incidents',
    title = 'Top 12 Types of Offences that involve Bicycle Thefts in Toronto',
    xaxis_type = 'category'
)
fig.show()


# Insight: Most bicycles stolen are under $ 5,000 price range, and just a bicycle theft, and not a combinaton of other types of offences

# In[ ]:


df_data = df_bicycle_thefts['Bike_Colour'].value_counts(normalize = False).head(12)

fig = px.bar(
    data_frame = df_data,
    x = df_data.index,
    y = df_data,    
)
fig.update_layout(
    xaxis_title = 'Bicycle Colour',
    yaxis_title = 'Proportion of Incidents',
    title = 'Top 12 types of Bicycles Colours of Stolen Bicycles',
    xaxis_type = 'category'
)
fig.show()


# Insight: Black is the most common colour of bicycles to be stolen. However, this may just be a reflection of the bicycle industry as a whole as most bicycles made may have a black colour, and not that black bicycles are more likely to be stolen.

# In[ ]:


df_data = df_bicycle_thefts['Year'].value_counts(normalize = False)

fig = px.bar(
    data_frame = df_data,
    x = df_data.index,
    y = df_data,    
)
fig.update_layout(
    xaxis_title = 'Year',
    yaxis_title = 'Proportion of Incidents',
    title = 'Proportion of Bicycles Thefts in Toronto by Year',
)
fig.show()


# Insight: Number of bicycle thefts are increasing year over year

# In[ ]:


months = ['January', 'February', 'March', 'April', 
          'May', 'June', 'July', 'August',
          'September', 'October', 'November', 'December']


# In[ ]:


df_data = df_bicycle_thefts['Month'].value_counts(normalize = False)

fig = px.bar(
    data_frame = df_data,
    x = df_data.index,
    y = df_data,
)
fig.update_layout(
    xaxis_title = 'Month',
    yaxis_title = 'Proportion of Incidents',
    title = 'Proportion of Bicycles Thefts in Toronto by Month of Year',
)
fig.update_xaxes(
    tickvals = list(range(1,13)),
    ticktext = months
)
fig.show()


# In[ ]:


fig = go.Figure()

for year in df_bicycle_thefts.Year.value_counts().sort_index().index:
    fig.add_trace(
        go.Scatter(
            x = df_bicycle_thefts[df_bicycle_thefts.Year == year].Month.value_counts().sort_index().index,
            y = df_bicycle_thefts[df_bicycle_thefts.Year == year].Month.value_counts().sort_index(),
            mode = 'lines+markers',
            name = year
        )
    )
    
fig.update_layout(
    xaxis_title = 'Month',
    yaxis_title = 'Proportion of Incidents',
    title = 'Proportion of Bicycles Thefts in Toronto by Month of all Years',
)
fig.update_xaxes(
    tickvals = list(range(1,len(months) + 1)),
    ticktext = months
)
fig.show()


# Insight: Bicycle thefts seen to be increasing in the summer months more than the previous years

# In[ ]:


df_data = df_bicycle_thefts['Week_of_year'].value_counts(normalize = False)

fig = px.bar(
    data_frame = df_data,
    x = df_data.index,
    y = df_data,
)
fig.update_layout(
    xaxis_title = 'Week of Year',
    yaxis_title = 'Proportion of Incidents',
    title = 'Proportion of Bicycles Thefts in Toronto by Week of Year',
)
fig.show()


# In[ ]:


fig = go.Figure()

for year in df_bicycle_thefts.Year.value_counts().sort_index().index:
    fig.add_trace(
        go.Scatter(
            x = df_bicycle_thefts[df_bicycle_thefts.Year == year].Week_of_year.value_counts().sort_index().index,
            y = df_bicycle_thefts[df_bicycle_thefts.Year == year].Week_of_year.value_counts().sort_index(),
            mode = 'lines+markers',
            name = year
        )
    )
    
fig.update_layout(
    xaxis_title = 'Week of Year',
    yaxis_title = 'Proportion of Incidents',
    title = 'Proportion of Bicycles Thefts in Toronto by Week of Year',
)
fig.show()


# Insight: Bicycle thefts seem to follow a cyclical pattern where the insidents proportions keep increasing and decreasing periodically over 4 weeks peaks/troughs

# In[ ]:


days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']


# In[ ]:


df_data = df_bicycle_thefts['Day_of_week'].value_counts(normalize = False)

fig = px.bar(
    data_frame = df_data,
    x = df_data.index,
    y = df_data,
)
fig.update_layout(
    xaxis_title = 'Day of Week',
    yaxis_title = 'Proportion of Incidents',
    title = 'Proportion of Bicycles Thefts in Toronto by Day of Week',
)
fig.update_xaxes(
    tickvals = list(range(1,len(days) + 1)),
    ticktext = days
)
fig.show()


# In[ ]:


fig = go.Figure()

for year in df_bicycle_thefts.Year.value_counts().sort_index().index:
    fig.add_trace(
        go.Scatter(
            x = df_bicycle_thefts[df_bicycle_thefts.Year == year].Day_of_week.value_counts().sort_index().index,
            y = df_bicycle_thefts[df_bicycle_thefts.Year == year].Day_of_week.value_counts().sort_index(),
            mode = 'lines+markers',
            name = year
        )
    )
    
fig.update_layout(
    xaxis_title = 'Day of Week',
    yaxis_title = 'Proportion of Incidents',
    title = 'Proportion of Bicycles Thefts in Toronto by Day of Week',
)
fig.update_xaxes(
    tickvals = list(range(1,len(days) + 1)),
    ticktext = days
)
fig.show()


# Insight: Over the years the number of bicycle thefts have shifted from being stolen on Fridays, to now the middle of the week Wednedays/Thursdays

# In[ ]:


df_data = df_bicycle_thefts['Hour'].value_counts(normalize = False)

fig = px.bar(
    data_frame = df_data,
    x = df_data.index,
    y = df_data,
)
fig.update_layout(
    xaxis_title = 'Hour (24 hours)',
    yaxis_title = 'Proportion of Incidents',
    title = 'Proportion of Bicycles Thefts in Toronto by Hour of Day',
)
fig.show()


# In[ ]:


fig = go.Figure()

for year in df_bicycle_thefts.Year.value_counts().sort_index().index:
    fig.add_trace(
        go.Scatter(
            x = df_bicycle_thefts[df_bicycle_thefts.Year == year].Hour.value_counts().sort_index().index,
            y = df_bicycle_thefts[df_bicycle_thefts.Year == year].Hour.value_counts().sort_index(),
            mode = 'lines+markers',
            name = year
        )
    )
    
fig.update_layout(
    xaxis_title = 'Hour (24 hours)',
    yaxis_title = 'Proportion of Incidents',
    title = 'Proportion of Bicycles Thefts in Toronto by Hour of Day',
)
fig.show()


# Insight: Bicycle thefts seem to be happen most during 9am, 12noon, and 4pm rush hours. This may be a reflection of when the bicycle thefts are reported, and not necessarily when the incidents happen. People who have their bicycle's stolen may be going to or leaving work, and that is when they discover their bicycle is stolen and report the crime, however the crime itself may be happening just before those peak hours.

# In[ ]:


fig = go.Figure(
    data = go.Scatterpolar(
        r = df_bicycle_thefts.Hour.value_counts().sort_index(),
        theta = df_bicycle_thefts.Hour.apply(lambda x: str(x).zfill(2) + ' Hr').value_counts().sort_index().index,
        fill = 'toself',
    )
)

fig.update_layout(
    title = 'Proportion of Bicycles Thefts in Toronto by Hour of Day',
    showlegend = False,
    polar = dict(
        radialaxis = dict(
            visible = True,
            angle = 90,
        ),
        angularaxis = dict(
            rotation = -90,
            direction = 'clockwise'
        )
    )    
)

fig.show()


# Insight: Bicycle thefts have shifted to happen moreso over weekdays during working hours of 9am to 4pm

# In[ ]:


df_data = pd.pivot_table(
    df_bicycle_thefts,
    values = 'Day',
    index = 'Year',
    columns = 'Month',
    aggfunc = len
)

#fig = px.imshow(df_data)
fig = go.Figure(
    data = go.Heatmap(
        z = df_data,
        x = df_data.columns.map(lambda x: months[x-1]),
        y = df_data.index,
    )
)
fig.update_layout(
    xaxis_title = 'Months',
    yaxis_title = 'Proportion of Incidents',
    title = 'Heatmap of Bicycles Thefts in Toronto by Month and Year',
)
fig.show()


# In[ ]:


df_data = pd.pivot_table(
    df_bicycle_thefts,
    values = 'Day',
    index = 'Year',
    columns = 'Day_of_week',
    aggfunc = len
)

#fig = px.imshow(df_data)
fig = go.Figure(
    data = go.Heatmap(
        z = df_data,
        x = df_data.columns.map(lambda x: days[x-1]),
        y = df_data.index,
    )
)
fig.update_layout(
    xaxis_title = 'Weekday',
    yaxis_title = 'Proportion of Incidents',
    title = 'Heatmap of Bicycles Thefts in Toronto by Weekday and Year',
)
fig.show()


# In[ ]:


df_data = pd.pivot_table(
    df_bicycle_thefts,
    values = 'Day',
    index = 'Month',
    columns = 'Day_of_week',
    aggfunc = len
)

#fig = px.imshow(df_data)
fig = go.Figure(
    data = go.Heatmap(
        z = df_data,
        x = df_data.columns.map(lambda x: days[x-1]),
        y = df_data.index.map(lambda x: months[x-1]),
    )
)
fig.update_layout(
    xaxis_title = 'Weekday',
    yaxis_title = 'Proportion of Incidents',
    title = 'Heatmap of Bicycles Thefts in Toronto by Weekday and Month',
)
fig.show()


# In[ ]:


df_data = pd.pivot_table(
    df_bicycle_thefts,
    values = 'Day',
    index = 'Year',
    columns = 'Hour',
    aggfunc = len
)

#fig = px.imshow(df_data)
fig = go.Figure(
    data = go.Heatmap(
        z = df_data,
        x = df_data.columns,
        y = df_data.index,
    )
)
fig.update_layout(
    xaxis_title = 'Hour (24 hours)',
    yaxis_title = 'Proportion of Incidents',
    title = 'Heatmap of Bicycles Thefts in Toronto by Hour of day and Year',
)
fig.show()


# In[ ]:


df_data = pd.pivot_table(
    df_bicycle_thefts,
    values = 'Day',
    index = 'Month',
    columns = 'Hour',
    aggfunc = len
)

#fig = px.imshow(df_data)
fig = go.Figure(
    data = go.Heatmap(
        z = df_data,
        x = df_data.columns,
        y = df_data.index.map(lambda x: months[x-1]),
    )
)
fig.update_layout(
    xaxis_title = 'Hour (24 hours)',
    yaxis_title = 'Proportion of Incidents',
    title = 'Heatmap of Bicycles Thefts in Toronto by Hour of Day and Month',
)
fig.show()


# In[ ]:


df_data = pd.pivot_table(
    df_bicycle_thefts,
    values = 'Day',
    index = 'Day_of_week',
    columns = 'Hour',
    aggfunc = len
)

#fig = px.imshow(df_data)
fig = go.Figure(
    data = go.Heatmap(
        z = df_data,
        x = df_data.columns,
        y = df_data.index.map(lambda x: days[x-1]),
    )
)
fig.update_layout(
    xaxis_title = 'Hour (24 hours)',
    yaxis_title = 'Proportion of Incidents',
    title = 'Heatmap of Bicycles Thefts in Toronto by Hour of Day and Weekday',
)
fig.show()


# In[ ]:


df_bicycle_thefts.iloc[0]


# ## free map styles:
# * white-bg
# * open-street-map
# * carto-positron
# * carto-darkmatter
# * stamen-terrain
# * stamen-toner
# * stamen-watercolor

# In[ ]:


fig = px.scatter_mapbox(
    data_frame = df_bicycle_thefts[df_bicycle_thefts.Cost_of_Bike.notnull()],
    lat = 'Lat',
    lon = 'Long',
    color = 'Status',
    size = 'Cost_of_Bike',
)
fig.update_layout(
    title = "Map of Bicycle Theft incidents by Stolen Status",
    mapbox_style = "carto-darkmatter"
)
fig.show()


# In[ ]:


fig = px.scatter_mapbox(
    data_frame = df_bicycle_thefts[df_bicycle_thefts.Cost_of_Bike.notnull()],
    lat = 'Lat',
    lon = 'Long',
    color = 'Primary_Offence',
    size = 'Cost_of_Bike',
)
fig.update_layout(
    title = "Map of Bicycle Theft incidents by their Primary Offence",
    mapbox_style="carto-darkmatter"
)
fig.show()


# In[ ]:


fig = px.scatter_mapbox(
    data_frame = df_bicycle_thefts[df_bicycle_thefts.Cost_of_Bike.notnull()],
    lat = 'Lat',
    lon = 'Long',
    color = 'Location_Type',
    size = 'Cost_of_Bike',
)
fig.update_layout(
    title = "Map of Bicycle Theft incidents by their Type of Location",
    mapbox_style = "carto-darkmatter"
)
fig.show()


# If you enjoyed my content, please rememer to upvote the notebook. If you have any questions, feel free to comment below or message me on [LinkedIn](https://www.linkedin.com/in/jaydeep-mistry/)
