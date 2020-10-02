#!/usr/bin/env python
# coding: utf-8

# **[Dashboarding with Scheduled Notebooks:](https://www.kaggle.com/rtatman/dashboarding-with-notebooks-day-5)**<br>
# 
# <h1>Service Requests Received by the Oakland Call Center</h1>
# This is a scheduled notebook that is updated daily at 12:00am Pacific Standard Time by Google Cloud Services.<br>
# [Dataset page here](https://www.kaggle.com/cityofoakland/oakland-call-center-public-work-service-requests) 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import plotly.plotly as ply
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()

import sys
get_ipython().system('{sys.executable} -m pip install csvvalidator')
from csvvalidator import *

import csv
from io import StringIO

today = pd.Timestamp.now(tz='America/Los_Angeles')


# In[ ]:


# CSV Validation
field_names = ('DATETIMEINIT',
               'REQCATEGORY',
               'REQADDRESS',
               'STATUS',
               'DATETIMECLOSED',
               )
validator = CSVValidator(field_names)

validator.add_value_check('REQCATEGORY', # the name of the field
                          str, # a function that accepts a single argument and 
                               # raises a `ValueError` if the value is not valid.
                               # Here checks that "key" is an integer.
                          'EX1', # code for exception
                          'Category must be a string'# message to report if error thrown
                         )
validator.add_value_check('REQADDRESS', # the name of the field
                          str, # a function that accepts a single argument and 
                               # raises a `ValueError` if the value is not valid.
                               # Here checks that "key" is an integer.
                          'EX2', # code for exception
                          'Address must be a string'# message to report if error thrown
                         )
validator.add_value_check('STATUS', # the name of the field
                          enumeration('OPEN', 'CLOSED', 'CANCEL','PENDING'), # a function that accepts a single argument and 
                               # raises a `ValueError` if the value is not valid.
                               # Here checks that "key" is an integer.
                          'EX3', # code for exception
                          'Status not recognized'# message to report if error thrown
                         )

# sample csv
good_data = StringIO("""DATETIMEINIT,REQCATEGORY,REQADDRESS,STATUS,DATETIMECLOSED
2017-06-05 11:43:13,ROW,Some long address,CLOSED,2017-06-17 13:40:55
""")
# read text in as a csv
test_csv = csv.reader(good_data)
# validate our good csv
validator.validate(test_csv)

cc = pd.read_csv('../input/service-requests-received-by-the-oakland-call-center.csv', parse_dates = ['DATETIMEINIT', 'DATETIMECLOSED'])


# In[ ]:


def process_data(raw_data):
    #Data processing goes here
    processed_data = raw_data
    processed_data.DATETIMEINIT = processed_data.DATETIMEINIT.dt.tz_localize('America/Los_Angeles', ambiguous=True)
    processed_data.DATETIMECLOSED = processed_data.DATETIMECLOSED.dt.tz_localize('America/Los_Angeles')
    processed_data['DAYSOPEN'] = pd.to_timedelta((processed_data.DATETIMECLOSED - processed_data.DATETIMEINIT).dt.round('h')).astype('timedelta64[D]')
    return processed_data


def lat(x):
    s = x.split("'")
    for i in range(0, len(s)):
        if (s[i] == 'latitude'):
            return float(s[i+2])
    return np.NaN


def lon(x):
    s = x.split("'")
    for i in range(0, len(s)):
        if (s[i] == 'longitude'):
            return float(s[i+2])
    return np.NaN

test_data = StringIO("""DATETIMEINIT,REQCATEGORY,REQADDRESS,STATUS,DATETIMECLOSED
2017-06-05 11:43:13,ROW,Some long address,OPEN,2017-06-17 13:40:55
""")
test_csv = pd.read_csv(test_data)  

output_data_correct = StringIO("""DATETIMEINIT,REQCATEGORY,REQADDRESS,STATUS,DATETIMECLOSED
2017-06-05 11:43:13,ROW,Some long address,OPEN,2017-06-17 13:40:55
""")
output_csv_correct = pd.read_csv(output_data_correct)

#pd.testing.assert_frame_equal(output_csv_correct, process_data(test_csv))
cc = process_data(cc)


# In[ ]:


gdf = cc

# Trim to requests made in the past 7 days and add column with no. of days open
gdf = gdf[pd.to_timedelta((today - gdf['DATETIMEINIT']).dt.round('d')).astype('timedelta64[D]') < 8]
gdf.loc[pd.isnull(gdf.DAYSOPEN),'DAYSOPEN'] = pd.to_timedelta((today - gdf.DATETIMEINIT).dt.round('h')).astype('timedelta64[D]')

# Drop unnecessary columns and then drop rows with NaN values, in that order!
gdf = gdf[['REQCATEGORY', 'REQADDRESS', 'DAYSOPEN', 'DATETIMEINIT', 'DESCRIPTION']].dropna()

# Parse latitude and longitude values from address
gdf.loc[:,'latitude'] = gdf.REQADDRESS.apply(lat)
gdf.loc[:,'longitude'] = gdf.REQADDRESS.apply(lon)


# ## Maintainence Update, 6/13/19:
# Mapbox authentication is failing, disabling rendering of the scatter map. I hope to resolve this soon.

# In[ ]:


# Plot with plotly and Mapbox
mapbox_access_token = 'pk.eyJ1Ijoid3RzY290dCIsImEiOiJjangwa2UyOGkxOHQ5M3lvNWppaDY1cTN4In0.DrkJn6BRhRkDmjrzFjbf7Q'

data = [
    go.Scattermapbox(
        lat=gdf.latitude,
        lon=gdf.longitude,
        mode='markers',
        text=gdf.DESCRIPTION,
        marker=dict(
            size=7,
            color=gdf.DAYSOPEN,
            colorscale='Bluered',
            reversescale=True,
            colorbar=dict(
                title='DAYS OPEN'
            )
        ),
    )
]

layout = go.Layout(
    autosize=True,
    hovermode='closest',
    title='Oakland Service Requests Opened in the Past 7 Days, Updated ' + str(today.date()),
    mapbox=dict(
        accesstoken=mapbox_access_token,
        bearing=0,
        center=dict(
            lat=37.805,
            lon=-122.250
        ),
        pitch=0,
        zoom=11
    ),
)

fig = dict(data=data, layout=layout)
iplot(fig, filename='Oakland Mapbox')


# In[ ]:


# Plot data with plotly
df = cc.dropna(subset=['DAYSOPEN'])
df = pd.DataFrame(df.groupby('REQCATEGORY')['DAYSOPEN'].mean()).reset_index()
df.sort_values(by='DAYSOPEN', inplace=True)

data = [go.Bar(y=df.DAYSOPEN, x=df.REQCATEGORY)]
layout = dict(title = 'How Long Are Requests Open, on Average?', 
              xaxis = dict(title = 'Request Category', ticklen=5, automargin=True, tickangle=45),
              yaxis = dict(title = 'Avg. Days to Close', automargin=True))
fig = dict(data = data, layout = layout)
iplot(fig)

