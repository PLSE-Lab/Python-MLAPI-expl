#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


import seaborn as sns
import geopandas as gpd
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
import plotly.plotly as py
from plotly import tools
import plotly.figure_factory as ff
import folium
from folium import plugins
from io import StringIO
init_notebook_mode(connected=True)


# In[ ]:


speed_cam_data = pd.read_csv("../input/speed-camera-locations.csv")
speed_cam_data.head()


# In[ ]:


speed_cam_v_data = pd.read_csv("../input/speed-camera-violations.csv")
speed_cam_v_data.head()


# In[ ]:


pd.isnull(speed_cam_v_data['LATITUDE']).count()


# Code need to verfied and modified to taker consider the data with null values for lattitude and longitude. 
# These  columns can be populated  based on address field which is common to both csv files

# **Extract  date  parameters and  as columns to dataframe**

# In[ ]:


import datetime
speed_cam_v_data['Violation date'] = pd.to_datetime(speed_cam_v_data['VIOLATION DATE']) # timestamp
speed_cam_v_data['day_name']=speed_cam_v_data['Violation date'].dt.day_name()  # get day of the week
speed_cam_v_data['month_name']=speed_cam_v_data['Violation date'].dt.month_name()
speed_cam_v_data['month']=speed_cam_v_data['Violation date'].dt.month
speed_cam_v_data['month'] =  speed_cam_v_data.month.map("{:02}".format)
speed_cam_v_data['year']=speed_cam_v_data['Violation date'].dt.year
speed_cam_v_data['yearmonth'] = speed_cam_v_data['year'].astype(str) + '-'+speed_cam_v_data['month']
#speed_cam_v_data['day']=speed_cam_v_data['Violation date'].dt.day
speed_cam_v_data.head()


# **Analyze last 7 days of violations  data  from speed camera** 

# In[ ]:


# To analyze last 7 days data
startdate=max(speed_cam_v_data['Violation date']) + datetime.timedelta(days= -6) # To analyze last 7 days data
lastweek_v = speed_cam_v_data[(speed_cam_v_data['Violation date'] >= startdate)]
# get total violations  grouped by ....
lastweek_v_g=lastweek_v.groupby(['ADDRESS', 'LATITUDE', 'LONGITUDE'])['VIOLATIONS'].sum().sort_values(ascending=False).reset_index()
max(lastweek_v_g['VIOLATIONS'])


# In[ ]:


# Note locations with more than half of max violations identified by speed camera. 
#Such locations can be closely watched to see for  more accidents etc 
v_limit = int(max(lastweek_v_g['VIOLATIONS'])/2)
lastweek_v_max = lastweek_v_g


# In[ ]:


v_map = folium.Map(location=[41.878, -87.62], height = 700, tiles='OpenStreetMap', zoom_start=12)
for i in range(0,len(lastweek_v_max)):
    folium.Marker([lastweek_v_max.iloc[i]['LATITUDE'], lastweek_v_max.iloc[i]['LONGITUDE']], popup=lastweek_v_max.iloc[i]['ADDRESS'], icon=folium.Icon(color= 'red' if lastweek_v_max.iloc[i]['VIOLATIONS'] > v_limit else 'green', icon='circle')).add_to(v_map)
v_map


# **Analyze  effectiveness of speed camera checking, are  violations  coming down  fin last 12 months?**

# In[ ]:


#prepare the data for the last one year and draw a scatter plot
# get the todate for analysis. Assume todate as the  last day of the previous month
maxdate = max(speed_cam_v_data['Violation date'])
maxday = maxdate.day
todate = maxdate + datetime.timedelta(days= -maxday) # last day of the previous month
#
from dateutil.relativedelta import relativedelta
#years_ago = datetime.datetime.now() - relativedelta(years=5)
fromdate = todate - relativedelta(years=1)
oneyear_v = speed_cam_v_data[(speed_cam_v_data['Violation date'] <= todate) & (speed_cam_v_data['Violation date'] > fromdate)]
oneyear_v.count()


# In[ ]:


# get total violations  grOuped by
oneyear_v_g=oneyear_v.groupby(['CAMERA ID','ADDRESS', 'LATITUDE', 'LONGITUDE'])['VIOLATIONS'].sum().sort_values(ascending=False).reset_index()
threshold = int(max(oneyear_v_g['VIOLATIONS'])/2)


# In[ ]:


cols = ['CAMERA ID','ADDRESS','LATITUDE', 'LONGITUDE', 'year','month','month_name','yearmonth','VIOLATIONS']
oneyear_vg = oneyear_v[cols]
oneyear_ag = oneyear_vg.groupby(['CAMERA ID','yearmonth','ADDRESS', 'LATITUDE', 'LONGITUDE']).agg({'VIOLATIONS':sum})


# In[ ]:


oneyear_v_treshold =oneyear_ag.groupby(['CAMERA ID','ADDRESS', 'LATITUDE', 'LONGITUDE']).filter( lambda x : x['VIOLATIONS'].sum() > threshold)


# In[ ]:


oneyear_v_treshold_noind_go = oneyear_v_treshold.reset_index()
#oneyear_v_treshold_noind_go


# In[ ]:


data = []
for name1 , group in oneyear_v_treshold_noind_go[["CAMERA ID", "yearmonth", "VIOLATIONS"]].groupby("CAMERA ID"):
    data.append(go.Scatter(x=group.yearmonth, y=group.VIOLATIONS,name=name1))
    
layout = dict(title = "Top violations by month for last 12 months",
              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'Violations',ticklen= 5,zeroline= False))
            

# create and show our figure
fig = dict(data = data, layout = layout)
iplot(fig)


# In[ ]:


# For concerned authorities to follow-up the effectiveness of speed camera system 
#address of camera  will be easier, instead of camera-id 
data = []
for name1 , group in oneyear_v_treshold_noind_go[["ADDRESS", "yearmonth", "VIOLATIONS"]].groupby("ADDRESS"):
    data.append(go.Scatter(x=group.yearmonth, y=group.VIOLATIONS,name=name1))
    
layout = dict(title = "Top violations by month for last 12 months",
              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'Violations',ticklen= 5,zeroline= False))
            

# create and show our figure
fig = dict(data = data, layout = layout)
iplot(fig)

