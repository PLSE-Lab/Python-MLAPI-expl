#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()

raw_data = pd.read_csv("../input/nypd-motor-vehicle-collisions.csv", low_memory=False)
# raw_data.head()

borough_data = raw_data[raw_data.BOROUGH.notnull()]                .filter(items=['BOROUGH','NUMBER OF PERSONS INJURED','NUMBER OF PERSONS KILLED'])                .groupby(['BOROUGH'])                .sum()

# first chart
data1 = [go.Bar(
            x=borough_data.index,
            y=borough_data['NUMBER OF PERSONS INJURED']
    )]
layout1 = go.Layout(
    barmode='group',
    title="Number of persons injured per Borough",
)
fig1 = go.Figure(data=data1, layout=layout1)
iplot(fig1)


# In[ ]:


trace1 = go.Bar(
    x=borough_data.index,
    y=borough_data['NUMBER OF PERSONS INJURED'],
    name='INJURED'
)
trace2 = go.Bar(
    x=borough_data.index,
    y=borough_data['NUMBER OF PERSONS KILLED'],
    name='KILLED'
)

data = [trace1, trace2]
layout = go.Layout(
    barmode='group',
    title="Number of persons injured and killed per Borough",
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# In[ ]:


import calendar
raw_data['DATE'] = pd.to_datetime(raw_data['DATE'], format="%Y-%m-%d")
months_data = raw_data                .filter(items=['DATE','NUMBER OF PERSONS INJURED','NUMBER OF PERSONS KILLED'])                .groupby([raw_data.DATE.dt.year, raw_data.DATE.dt.month])                .sum()
months_data['date'] = months_data.index
months_data['date'] = pd.to_datetime(months_data['date'], format="(%Y, %m)")
months_data = months_data.reset_index(drop=True)

# making a chart
trace3 = go.Scatter(
    x=months_data.date, # assign x as the dataframe column 'x'
    y=months_data['NUMBER OF PERSONS INJURED'],
    name='INJURED'
)
trace4 = go.Scatter(
    x=months_data.date,
    y=months_data['NUMBER OF PERSONS KILLED'],
    name='KILLED'
)

data2 = [trace3, trace4]
layout2 = dict(title="Number of persons injured and killed per month",
              xaxis=dict(title='Date', ticklen=10, zeroline=False))
fig = dict(data=data2, layout=layout2)
iplot(fig)


# In[ ]:




