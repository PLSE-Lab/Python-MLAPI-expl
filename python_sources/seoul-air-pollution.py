#!/usr/bin/env python
# coding: utf-8

# This is my first dataset that uses Plotly after studying for a few days. If you like these visualisations, please give me a upvote or leave a comment, so I can see they're appropriate :)

# This dataset deals with air pollution measurement information in Seoul, South Korea.
# Seoul Metropolitan Government provides many public data, including air pollution information, through the 'Open Data Plaza'
# I made a structured dataset by collecting and adjusting various air pollution related datasets provided by the Seoul Metropolitan Government
# 
# Content
# This data provides average values for six pollutants (SO2, NO2, CO, O3, PM10, PM2.5).
# 
# Data were measured every hour between 2017 and 2019.
# Data were measured for 25 districts in Seoul.
# This dataset is divided into four files.
# Measurement info: Air pollution measurement information
# 
# 1 hour average measurement is provided after calibration
# Instrument status:
# 0: Normal, 1: Need for calibration, 2: Abnormal
# 4: Power cut off, 8: Under repair, 9: abnormal data
# Measurement item info: Information on air pollution measurement items
# 
# Measurement station info: Information on air pollution instrument stations
# 
# Measurement summary: A condensed dataset based on the above three data.
# 
# Acknowledgements
# Data is provided from here.
# 
# https://data.seoul.go.kr/dataList/OA-15526/S/1/datasetView.do
# https://data.seoul.go.kr/dataList/OA-15516/S/1/datasetView.do
# https://data.seoul.go.kr/dataList/OA-15515/S/1/datasetView.do
# Thank you to Seoul City, Seoul Open Data Plaza, and Air Quality Analysis Center for providing data.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv("../input/air-pollution-in-seoul/AirPollutionSeoul/Measurement_summary.csv")


# In[ ]:


df.head()


# By checking the columns of this dataset, it's probably worth dropping the Address feature, since data was collected from 25 different districts, so we only have 25 different address. It'll be more useful to use Latitude and Longitude, so we can probably plot this data into a x-y graph later.

# In[ ]:


df['Station code'].unique()


# In[ ]:


table = df.Address.groupby(df.Address, as_index=True)
print("The dataset contains ",len(table), "different addresses")
table1 = df.Latitude.groupby(df.Latitude, as_index=True)
print("The dataset contains ",len(table1), "different Latitudes")
table2 = df.Longitude.groupby(df.Longitude, as_index=True)
print("The dataset contains ",len(table2), "different Longitudes")


# In[ ]:


df.drop("Address", axis=1, inplace=True)


# In[ ]:


df['Measurement date'] = pd.to_datetime(df['Measurement date'])


# According to the Data, these are the levels for air quality:
#         
#         We'll use these values to plot interesting data

# In[ ]:


polluents = {'SO2':[0.02,0.05,0.15,1],'NO2':[0.03,0.06,0.2,2],'CO':[2,9,15,50],'O3':[0.03,0.09,0.15,0.5],'PM2.5':[15,35,75,500],'PM10':[30,80,150,600]}
quality = ['Good','Normal','Bad','Very Bad']
seoul_standard = pd.DataFrame(polluents, index=quality)
seoul_standard


# We'll start with Station 101

# In[ ]:


df_101 = pd.DataFrame(df.loc[(df['Station code']==101)])


# In[ ]:


df_101.head()


# In[ ]:


df_101.drop("Station code", axis=1, inplace=True)


# In[ ]:


import plotly
import plotly.graph_objs as go
import plotly.offline as py


# In[ ]:


plotly.offline.init_notebook_mode(connected=True)


# In[ ]:


data = [go.Scatter(x=df_101['Measurement date'],
                   y=df_101['SO2'])]
       
##layout object
layout = go.Layout(title='SO2 Levels',
                    yaxis={'title':'Level (ppm)'},
                    xaxis={'title':'Date'})
                    
## Figure object

fig = go.Figure(data=data, layout=layout)

## Plotting
py.iplot(fig)


# We can see that there are some -1 negative values on this data. Since I believe this is not possible, we'll search this lines and drop then from the dataset. Maybe this is a wrong reading of the instrument used to measure polluent levels.
# 
# This won't affect much our data, since only a few data points are negative.
# 
# As this may have also happened for other columns, we'll first count the negative elements, then delete them.

# In[ ]:


print("We have", df_101['SO2'].loc[(df_101['SO2']<0)].count(),"negative values for SO2")


# In[ ]:


print("We have", df_101['NO2'].loc[(df_101['NO2']<0)].count(),"negative values for NO2")


# In[ ]:


print("We have", df_101['O3'].loc[(df_101['O3']<0)].count(),"negative values for O3")


# In[ ]:


print("We have", df_101['CO'].loc[(df_101['CO']<0)].count(),"negative values for CO")


# In[ ]:


print("We have", df_101['PM2.5'].loc[(df_101['PM2.5']<0)].count(),"negative values for PM2.5")


# In[ ]:


print("We have", df_101['PM10'].loc[(df_101['PM10']<0)].count(),"negative values for PM10")


# Wow, we can see that SO2, NO2, O3 and CO contain the same amount of negative values. Let's try to plot them and check if they happen at the same day

# In[ ]:


data = [go.Scatter(x=df_101['Measurement date'],
                   y=df_101['SO2'], name='SO2'),
        go.Scatter(x=df_101['Measurement date'],
                   y=df_101['NO2'], name='NO2'),
        go.Scatter(x=df_101['Measurement date'],
                   y=df_101['CO'], name='CO'),
        go.Scatter(x=df_101['Measurement date'],
                   y=df_101['O3'], name='O3')]
       
##layout object
layout = go.Layout(title='Gases Levels',
                    yaxis={'title':'Level (ppm)'},
                    xaxis={'title':'Date'})
                    
## Figure object

fig = go.Figure(data=data, layout=layout)

## Plotting
py.iplot(fig)


# As we can see by this cool iterative graph, the points are at the same day. Something weird happened on those days. Let's drop them from the dataset.

# In[ ]:


to_drop = df_101.loc[(df_101['SO2']<0) | (df_101['NO2']<0) | (df_101['CO']<0) | (df_101['O3']<0)]
to_drop


# You can see on the data, that most of them are from one sequence of measurements. From Day 02/05/2017 up to 02/07/2017. Also there are some other spare wrong measurements.

# In[ ]:


df_101.drop(to_drop.index, axis=0, inplace=True)


# In[ ]:


data = [go.Scatter(x=df_101['Measurement date'],
                   y=df_101['SO2'], name='SO2'),
        go.Scatter(x=df_101['Measurement date'],
                   y=df_101['NO2'], name='NO2'),
        go.Scatter(x=df_101['Measurement date'],
                   y=df_101['CO'], name='CO'),
        go.Scatter(x=df_101['Measurement date'],
                   y=df_101['O3'], name='O3')]
       
##layout object
layout = go.Layout(title='Gases Levels',
                    yaxis={'title':'Level (ppm)'},
                    xaxis={'title':'Date'})
                    
## Figure object

fig = go.Figure(data=data, layout=layout)

## Plotting
py.iplot(fig)


# Now that we got rid of these wrong values for gases. Let's take a look on PM2.5 and PM10 values, as they had more negative values than the others

# In[ ]:


data = [go.Scatter(x=df_101['Measurement date'],
                   y=df_101['PM2.5'], name='PM2.5'),
        go.Scatter(x=df_101['Measurement date'],
                   y=df_101['PM10'], name='PM10'),
        ]
       
##layout object
layout = go.Layout(title='SO2 Levels',
                    yaxis={'title':'Level (ppm)'},
                    xaxis={'title':'Date'})
                    
## Figure object

fig = go.Figure(data=data, layout=layout)

## Plotting
py.iplot(fig)


# It's pretty clear that these values are high in number and -1 is clearly wrong. Let's check their data.
# 
# Also, if you take a deeper look, you'll also notice some values that are 0. We'll check for those too.

# In[ ]:


to_drop_PM = df_101.loc[(df_101['PM2.5']<0) | (df_101['PM10']<0) | (df_101['PM2.5']==0) | (df_101['PM10']==0)]
to_drop_PM


# Despite having more cases than the gases, we already deleted some of the lines when we got rid of the lines with negative values from gases. So, it makes sense this table be smaller.

# In[ ]:


df_101.drop(to_drop_PM.index, axis=0, inplace=True)


# In[ ]:


data = [go.Scatter(x=df_101['Measurement date'],
                   y=df_101['PM2.5'], name='PM2.5'),
        go.Scatter(x=df_101['Measurement date'],
                   y=df_101['PM10'], name='PM10'),
        ]
       
##layout object
layout = go.Layout(title='SO2 Levels',
                    yaxis={'title':'Level (ppm)'},
                    xaxis={'title':'Date'})
                    
## Figure object

fig = go.Figure(data=data, layout=layout)

## Plotting
py.iplot(fig)


# Now that we removed undesired data from our dataset, let's take a individual look at each column.

# In[ ]:


df_101.head(2)


# In[ ]:


df_101.tail(2)


# In[ ]:


seoul_standard


# In[ ]:


data = [go.Scatter(x=df_101['Measurement date'],
                   y=df_101['SO2'])]
       
##layout object
layout = go.Layout(title='SO2 Levels',
                    yaxis={'title':'Level (ppm)'},
                    xaxis={'title':'Date'})
                    
## Figure object

fig = go.Figure(data=data, layout=layout)

    

##Adding the text and positioning it
fig.add_trace(go.Scatter(
    x=['2017-03-01 00:00:00', '2017-07-31 23:00:00'],
    y=[0.2, 0.15],
    text=["Safe Level - Green", "Normal Level - Orange"],
    mode="text",
            ))

##Adding horizontal line
fig.add_shape(
        # Line Horizontal
            type="line",
            x0='2017-01-01 00:00:00',
            y0=0.02,
            x1='2019-12-31 23:00:00',
            y1=0.02,
            line=dict(
                color="Green",
                width=4,
                dash="dashdot",
            ))

fig.add_shape(
        # Line Horizontal
            type="line",
            x0='2017-01-01 00:00:00',
            y0=0.05,
            x1='2019-12-31 23:00:00',
            y1=0.05,
            line=dict(
                color="Orange",
                width=4,
                dash="dashdot",
            ))


## Plotting
py.iplot(fig)


# In[ ]:


data = [go.Scatter(x=df_101['Measurement date'],
                   y=df_101['NO2'])]
       
##layout object
layout = go.Layout(title='NO2 Levels',
                    yaxis={'title':'Level (ppm)'},
                    xaxis={'title':'Date'})
                    
## Figure object

fig = go.Figure(data=data, layout=layout)

    

##Adding the text and positioning it
fig.add_trace(go.Scatter(
    x=['2017-03-01 00:00:00', '2017-07-31 23:00:00'],
    y=[0.2, 0.15],
    text=["Safe Level - Green", "Normal Level - Orange"],
    mode="text",
            ))

##Adding horizontal line
fig.add_shape(
        # Line Horizontal
            type="line",
            x0='2017-01-01 00:00:00',
            y0=0.03,
            x1='2019-12-31 23:00:00',
            y1=0.03,
            line=dict(
                color="Green",
                width=4,
                dash="dashdot",
            ))

fig.add_shape(
        # Line Horizontal
            type="line",
            x0='2017-01-01 00:00:00',
            y0=0.06,
            x1='2019-12-31 23:00:00',
            y1=0.06,
            line=dict(
                color="Orange",
                width=4,
                dash="dashdot",
            ))


## Plotting
py.iplot(fig)


# In[ ]:


data = [go.Scatter(x=df_101['Measurement date'],
                   y=df_101['CO'])]
       
##layout object
layout = go.Layout(title='CO Levels',
                    yaxis={'title':'Level (ppm)'},
                    xaxis={'title':'Date'})
                    
## Figure object

fig = go.Figure(data=data, layout=layout)

    

##Adding the text and positioning it
fig.add_trace(go.Scatter(
    x=['2017-03-01 00:00:00', '2017-07-31 23:00:00'],
    y=[15, 10],
    text=["Safe Level - Green", "Normal Level - Orange"],
    mode="text",
            ))

##Adding horizontal line
fig.add_shape(
        # Line Horizontal
            type="line",
            x0='2017-01-01 00:00:00',
            y0=2,
            x1='2019-12-31 23:00:00',
            y1=2,
            line=dict(
                color="Green",
                width=4,
                dash="dashdot",
            ))

fig.add_shape(
        # Line Horizontal
            type="line",
            x0='2017-01-01 00:00:00',
            y0=9,
            x1='2019-12-31 23:00:00',
            y1=9,
            line=dict(
                color="Orange",
                width=4,
                dash="dashdot",
            ))


## Plotting
py.iplot(fig)


# In[ ]:


data = [go.Scatter(x=df_101['Measurement date'],
                   y=df_101['O3'])]
       
##layout object
layout = go.Layout(title='O3 Levels',
                    yaxis={'title':'Level (ppm)'},
                    xaxis={'title':'Date'})
                    
## Figure object

fig = go.Figure(data=data, layout=layout)

    

##Adding the text and positioning it
fig.add_trace(go.Scatter(
    x=['2017-03-01 00:00:00', '2017-07-31 23:00:00'],
    y=[0.2, 0.15],
    text=["Safe Level - Green", "Normal Level - Orange"],
    mode="text",
            ))

##Adding horizontal line
fig.add_shape(
        # Line Horizontal
            type="line",
            x0='2017-01-01 00:00:00',
            y0=0.03,
            x1='2019-12-31 23:00:00',
            y1=0.03,
            line=dict(
                color="Green",
                width=4,
                dash="dashdot",
            ))

fig.add_shape(
        # Line Horizontal
            type="line",
            x0='2017-01-01 00:00:00',
            y0=0.09,
            x1='2019-12-31 23:00:00',
            y1=0.09,
            line=dict(
                color="Orange",
                width=4,
                dash="dashdot",
            ))


## Plotting
py.iplot(fig)


# In[ ]:


data = [go.Scatter(x=df_101['Measurement date'],
                   y=df_101['PM2.5'])]
       
##layout object
layout = go.Layout(title='PM2.5 Levels',
                    yaxis={'title':'Level (ppm)'},
                    xaxis={'title':'Date'})
                    
## Figure object

fig = go.Figure(data=data, layout=layout)

    

##Adding the text and positioning it
fig.add_trace(go.Scatter(
    x=['2017-03-01 00:00:00', '2017-07-31 23:00:00'],
    y=[0.2, 0.15],
    text=["Safe Level - Green", "Normal Level - Orange"],
    mode="text",
            ))

##Adding horizontal line
fig.add_shape(
        # Line Horizontal
            type="line",
            x0='2017-01-01 00:00:00',
            y0=15,
            x1='2019-12-31 23:00:00',
            y1=15,
            line=dict(
                color="Green",
                width=4,
                dash="dashdot",
            ))

fig.add_shape(
        # Line Horizontal
            type="line",
            x0='2017-01-01 00:00:00',
            y0=35,
            x1='2019-12-31 23:00:00',
            y1=35,
            line=dict(
                color="Orange",
                width=4,
                dash="dashdot",
            ))


## Plotting
py.iplot(fig)


# In[ ]:


data = [go.Scatter(x=df_101['Measurement date'],
                   y=df_101['PM10'])]
       
##layout object
layout = go.Layout(title='PM10 Levels',
                    yaxis={'title':'Level (ppm)'},
                    xaxis={'title':'Date'})
                    
## Figure object

fig = go.Figure(data=data, layout=layout)

    

##Adding the text and positioning it
fig.add_trace(go.Scatter(
    x=['2017-03-01 00:00:00', '2017-07-31 23:00:00'],
    y=[0.2, 0.15],
    text=["Safe Level - Green", "Normal Level - Orange"],
    mode="text",
            ))

##Adding horizontal line
fig.add_shape(
        # Line Horizontal
            type="line",
            x0='2017-01-01 00:00:00',
            y0=30,
            x1='2019-12-31 23:00:00',
            y1=30,
            line=dict(
                color="Green",
                width=4,
                dash="dashdot",
            ))

fig.add_shape(
        # Line Horizontal
            type="line",
            x0='2017-01-01 00:00:00',
            y0=80,
            x1='2019-12-31 23:00:00',
            y1=80,
            line=dict(
                color="Orange",
                width=4,
                dash="dashdot",
            ))


## Plotting
py.iplot(fig)


# From the graphs we can see that these parameters are mostly normal or good
# 
#         SO2 and CO
# 
# However, we can see that levels of these parameters are often going beyong the normal level
#         
#         NO2
#         O3
#         PM2.5
#         PM10
#         
# I'm not an expert on air quality, so I'll leave solutions to this for those who knows better. For better understanding of air quality, you can check the following links:
# 
# https://www.eea.europa.eu/themes/air/air-quality-concentrations/air-quality-standards
# 
# https://www.epa.gov/criteria-air-pollutants/naaqs-table
# 
# 
# Finally, it would be very interesting to have data up to today as well. AS the corona virus outbreak continue, air quality is improving in many cities, data for this period would be extremely helpful to extract insights and make notes and extra visualisations:
# 
# https://www.france24.com/en/20200322-air-quality-is-improving-in-countries-coronavirus-quarantine-pollution-environment
