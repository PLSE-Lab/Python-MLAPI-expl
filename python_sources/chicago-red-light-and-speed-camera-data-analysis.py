#!/usr/bin/env python
# coding: utf-8

# This is a dataset hosted by the City of Chicago.
# 
# * red-light-camera-locations.csv - This dataset shows the location, first operational date, and approaches of the red light cameras in the City of Chicago. The approach describes the originating direction of travel which is monitored by a red light camera.
# * red-light-camera-violations.csv - This dataset reflects the daily volume of violations created by the City of Chicago Red Light Program for each camera. The data reflects violations that occurred from July 1, 2014 until present, minus the most recent 14 days. This data may change due to occasional time lags between the capturing of a potential violation and the processing and determination of a violation. The most recent 14 days are not shown due to revised data being submitted to the City of Chicago during this period. The reported violations are those that have been collected by the camera system and reviewed by two separate City contractors. In some instances, due to the inability the registered owner of the offending vehicle, the violation may not be issued as a citation. However, this dataset contains all violations regardless of whether a citation was actually issued, which provides an accurate view into the Red Light Program. Because of occasional time lags between the capturing of a potential violation and the processing and determination of a violation, as well as the occasional revision of the determination of a violation, this data may change. 
# More information on the Red Light Program can be found here: http://www.cityofchicago.org/city/en/depts/cdot/supp_info/red-light_cameraenforcement.html. 
# The corresponding dataset for speed camera violations is https://data.cityofchicago.org/id/hhkd-xvj4.
# * speed-camera-locations.csv - This dataset shows the location, first operational date, and approaches of the speed cameras in the City of Chicago. The approach describes the originating direction of travel which is monitored by a speed camera.
# * speed-camera-violations.csv - This dataset reflects the daily volume of violations that have occurred in Children's Safety Zones for each camera. The data reflects violations that occurred from July 1, 2014 until present, minus the most recent 14 days. This data may change due to occasional time lags between the capturing of a potential violation and the processing and determination of a violation. The most recent 14 days are not shown due to revised data being submitted to the City of Chicago. The reported violations are those that have been collected by the camera and radar system and reviewed by two separate City contractors. In some instances, due to the inability the registered owner of the offending vehicle, the violation may not be issued as a citation. However, this dataset contains all violations regardless of whether a citation was issued, which provides an accurate view into the Automated Speed Enforcement Program violations taking place in Children's Safety Zones. 
# More information on the Safety Zone Program can be found here: http://www.cityofchicago.org/city/en/depts/cdot/supp_info/children_s_safetyzoneporgramautomaticspeedenforcement.html. 
# The corresponding dataset for red light camera violations is https://data.cityofchicago.org/id/spqx-js37.

# In[ ]:


import pandas as pd
import numpy as np
import warnings
warnings.simplefilter("ignore")
# import plotly
import plotly.plotly as py
import plotly.graph_objs as go

# these two lines are what allow your code to show up in a notebook
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()


# In[ ]:


rlcam_loc_data = pd.read_csv("../input/red-light-camera-locations.csv")
rlcam_loc_data.head()


# In[ ]:


rlcam_violations_data = pd.read_csv("../input/red-light-camera-violations.csv")
rlcam_violations_data.head()


# There are some general guidelines you can use to figure out what information to include in a dashboard.
# 
# * What information is changing relatively quickly (every day or hour)? Information that only changes every quarter or year probably belong in a report, not a dashboard.
# * What information is the most important to your mission? If you're a company, things like money or users are probably going to be pretty important, but if you're a school district you probably care more about things like attendance or grades.
# * What will affect the choices you or others will need to make? Are you running A/B tests and need to choose which model to keep in production based on them? Then it's probably important that you track your metrics and other things that might affect those metrics, like sales that are running at the same time. Is there some outside factor that might affect your business, like the weather forecast next week? Then it might make sense to pull in another dataset and show that as well.
# * What changes have you made? If you're tuning parameters or adjusting teaching schedules, you want to track the fact that you've made those changes and also how they've affected outcomes.

# In[ ]:


print("Total Number of Red Light Camera Locations: {}" .format(rlcam_loc_data.shape[0]))
print("Total Number of Red Light Camera Violations: {}" .format(rlcam_violations_data.shape[0]))


# In[ ]:


null_columns=rlcam_loc_data.columns[rlcam_loc_data.isnull().any()]
rlcam_loc_data[null_columns].isnull().sum()


# In[ ]:


null_columns_violations=rlcam_violations_data.columns[rlcam_violations_data.isnull().any()]
rlcam_violations_data[null_columns_violations].isnull().sum()


# In[ ]:


print(rlcam_loc_data['FIRST APPROACH'].unique())
print(rlcam_loc_data['FIRST APPROACH'].value_counts())
data = [go.Bar(x= ['NB', 'SB', 'EB', 'WB', 'SEB'], y=rlcam_loc_data['FIRST APPROACH'].value_counts())]

iplot({'data': data,
       'layout': {
           'title': 'Value Counts',
           'xaxis': {
               'title': 'First Approach Value Counts'},
           'yaxis' : {
               'title': 'Counts'}
       }})


# In[ ]:


print(rlcam_loc_data['SECOND APPROACH'].unique())
print(rlcam_loc_data['SECOND APPROACH'].value_counts())
data = [go.Bar(x= ['WB', 'EB', 'SB', 'NB', 'SEB'], y=rlcam_loc_data['SECOND APPROACH'].value_counts())]

iplot({'data': data,
       'layout': {
           'title': 'Value Counts',
           'xaxis': {
               'title': 'SECOND Approach Value Counts'},
           'yaxis' : {
               'title': 'Counts'}
       }})


# In[ ]:


print(rlcam_loc_data['THIRD APPROACH'].unique())
print(rlcam_loc_data['THIRD APPROACH'].value_counts())
data = [go.Bar(x= ['NB', 'EB', 'SB'], y=rlcam_loc_data['THIRD APPROACH'].value_counts())]

iplot({'data': data,
       'layout': {
           'title': 'Value Counts',
           'xaxis': {
               'title': 'THIRD Approach Value Counts'},
           'yaxis' : {
               'title': 'Counts'}
       }})


# In[ ]:


data = [
    go.Scattermapbox(
        lat=rlcam_loc_data['LATITUDE'],
        lon=rlcam_loc_data['LONGITUDE'],
        mode='markers',
        marker=dict(
            size=11, 
            color = "rgb(180,31,33)"
        ),
        text=rlcam_loc_data['LOCATION'],
    )
]

layout = go.Layout(
    autosize=True,
    hovermode='closest',
    mapbox=dict(
        accesstoken="pk.eyJ1IjoiYW5pcnVkaDgzOCIsImEiOiJjanB3d2hmemwwZTB4NDNwOXlhendtYnBxIn0.zwms8s--1hpsc6KTyAtU6A",
        bearing=0,
        zoom = 10.3,
        center=dict(
            lat=41.895737080,
            lon=-87.6731264831),
    ),
        height = 1000,
        width = 1000,
        title = "Red Light Camera Locations"
)

fig = dict(data=data, layout=layout)
iplot(fig, filename='Red Light Cameras in Chicago')


# In[ ]:




