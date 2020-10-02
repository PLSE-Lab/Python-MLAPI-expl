#!/usr/bin/env python
# coding: utf-8

# # Seattle Road Weather - DashboardingPlotly
# ![](https://assets.smoothradio.com/2013/30/weather-1375260252-article-1.jpg)

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

import os
import warnings
print(os.listdir("../input"))

warnings.filterwarnings('ignore')
pd.options.display.float_format = '{:20,.2f}'.format
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data = pd.read_csv("../input/road-weather-information-stations.csv", parse_dates=[2])


#  I'm using just **2015 ** of the data because I run out of memory too many times.

# In[ ]:


data = data[(data.DateTime.dt.year ==2015)]


# In[ ]:


data.shape


#  Let's check the columns and their types

# In[ ]:


data.head()


# In[ ]:


data.info()


# # It's always a good practice to extract some useful descriptive statistics before going deep
# - RoadSurfaceTemperature and AirTemperature seems to be in Farenheit, I couldn't find out a complete description of these fields.
# - RoadSurfaceTemperature and AirTemperature are very similar

# In[ ]:


data.describe()


# In[ ]:


data.StationName.value_counts()


# In[ ]:


data.StationLocation.value_counts()


# **What information is changing relatively quickly (every day or hour)?**
# > AirTemperature and RoadTemperature were recorded every minute since 2014 in each Station, So I would like to contrast the trend against other months and years.
# 
# **What information is the most important to your mission?**
# > The AirTemperature and RoadTemperature in each Station
# 
# **What will affect the choices you or others will need to make?**
# > Which are the months with highest temperature and if there is a relation of causasity between the time either month, specific weeks,  and the temperature (AirTemperature, RoadSurfaceTemperature).
# 
# **What changes have you made?**
# > Track the temperature variability  thorugh the years.
# 
# 

# In[ ]:


data = data.set_index("DateTime", drop=False)
data["year"] = data.DateTime.dt.year
data["month"] = data.DateTime.dt.month
data["day"] = data.DateTime.dt.day


# In[ ]:


data.shape


# In[ ]:


data.head()


# In[ ]:


dictionaries_stationlocation = data["StationLocation"].apply(lambda x: eval(x))


# In[ ]:


data.drop("StationLocation", axis=1, inplace=True)


# In[ ]:


def downcast_dtypes(df):   
    # Select columns to downcast
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols =   [c for c in df if df[c].dtype == "int64"]
    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols]   = df[int_cols].astype(np.int32)
    return df


# In[ ]:


data = downcast_dtypes(data)


# In[ ]:


data["longitude"] = dictionaries_stationlocation.apply(lambda x: x["longitude"])


# In[ ]:


data["latitude"] = dictionaries_stationlocation.apply(lambda x: x["latitude"])


# In[ ]:


data["latitude"] = data["latitude"].astype(np.float32)
data["longitude"] = data["longitude"].astype(np.float32)


# In[ ]:


dictionaries_stationlocation = 0


# In[ ]:


data = downcast_dtypes(data)


# In[ ]:


data.head()


# **SeriesTime plot with plotly **
#  ![](https://upload.wikimedia.org/wikipedia/commons/thumb/3/37/Plotly-logo-01-square.png/220px-Plotly-logo-01-square.png)

# In[ ]:


data_feb = data[data.month==2].groupby(["StationName",pd.Grouper(freq='D')])["AirTemperature","RoadSurfaceTemperature"].mean()


# In[ ]:


data_feb.reset_index(inplace=True)
data_feb.set_index("DateTime", inplace=True)


# In[ ]:


stations =  data_feb.StationName.unique()
stations


# In[ ]:


import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools

from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()


# # Subplots
# Let's create a plot for every *Station* in a grid

# In[ ]:


scatter_list = []
subtitles = ()
for index, item in enumerate(stations):
    subtitles = subtitles + (item,)

print(subtitles)
fig = tools.make_subplots(rows=10, cols=1,
                          subplot_titles=subtitles,shared_xaxes=True)


# Create The lineplots  for each Station
for index, item in enumerate(stations):
    line = go.Scatter(x=data_feb[data_feb["StationName"]==item].index, 
                       y=data_feb[data_feb["StationName"]==item].AirTemperature,    
                       mode = 'lines+markers',
                       name = item,
                      marker = dict(
                      size = 3,
                      line = dict(
                                width = 2
                              )
                            )
                      )
    scatter_list.append(line)

# place each plot in a specific position
num_plots = len(scatter_list)
count = 0
for i in range(0,9):
#    for j in range(0,2):
    fig.append_trace(scatter_list[count],i+1, 1)
    count = count+1
    if count==num_plots:
        break

layout = dict(title = "Mean AirTemperature per Station daily",
              xaxis= dict(title= 'day',ticklen= 8,zeroline= False,ticks='outside',tickcolor='#000',
                          rangeslider=dict(visible = True), type='date'),
              yaxis = dict(title= 'day',ticklen= 8,zeroline= False,ticks='outside'))

fig.layout.xaxis.update(rangeslider = dict(visible = True))
fig.layout.update(height=1000, width=1200, title = "Mean AirTemperature per Station daily")
# fig =  go.Figure(data=scatter_list, layout=layout)
iplot(fig)    


# In[ ]:


data_resume = data_feb.groupby(pd.Grouper(freq='D')).mean()
line1 = go.Scatter(x=data_resume.index, 
                   y=data_resume.AirTemperature,    
                   mode = 'lines+markers',
                   name = 'AirTemperature',
                   marker = dict(
                       color = 'rgb(166, 206, 204)',
                      size = 4,
                      line = dict(
                        color = 'rgb(118, 206, 249)',
                        width = 1
                      )
                    )
                  )
line2 = go.Scatter(x=data_resume.index, 
                   y=data_resume.RoadSurfaceTemperature,    
                   mode = 'lines',
                   name = 'RoadSurfaceTemperature',
                   marker = dict(
                      color = 'rgb(232, 138, 62)',
                      size = 4,
                      line = dict(
                        color = 'rgb(232, 167, 72)',
                        width = 1
                      )
                    )
                  
                )

data_plotly=[line1, line2]

layout = dict(title = "Temperature",
              xaxis= dict(title= 'day',ticklen= 8,zeroline= False,ticks='outside',tickcolor='#000',
                          rangeslider=dict(visible = True), type='date'),
              yaxis = dict(title= 'day',ticklen= 8,zeroline= False,ticks='outside'))

fig =  go.Figure(data=data_plotly, layout=layout)
iplot(fig)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




