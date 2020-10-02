#!/usr/bin/env python
# coding: utf-8

# <p style="letter-spacing: 0.1em; color: #EF798A; font-size: 2em"> Bikesharing Competition Data Preparation Using Plotly </p>

# In[ ]:


# load required packages
import pandas as pd
import numpy as np
from scipy import stats
import calendar
from datetime import datetime

#plotly
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go

import missingno as msno

# Block the warning messages
import warnings 
warnings.filterwarnings('ignore')

import os
print(os.listdir("../input"))


# In[ ]:


# load dataset
data_train = pd.read_csv("../input/train.csv")
data_test = pd.read_csv("../input/test.csv")

data_bike = data_train
# merge train and test sets
#frame = [data_train, data_test]
#data = pd.concat(frame)


# <p style="font-size: 2em; color: #4ECDC4;"> Identify the Question </p>
# Before starting analyzing data I want to think a little more about why we want to forcast bike rental demands.<br>
# It can help the company in two major way:
# <ul>
# <li>Improve their services
# <li>Increase their costumers
# </ul>
# <p style="font-size: 1.25em; color: #6ED6CE;"> Improve Services:</p>
# <ul>
#     <li> During busy hours provide more bikes.
#     <li> Make these insights available for costumers so they can decide better on when and where to count on availability of bikes.
# </ul>
# <p style="font-size: 1.25em; color: #8EDFD9;"> Increase Costumers:</p>
# <ul>
#     <li> During low-demand hours and for low-demand palces offer lower prices
# </ul>
# 
# By analyzing historical data:
# <ul>
#     <li>The company can compare the progress through years and compare yearly patterns to check profitability.<br>
#     <li>Also, they can identify kiosks and routes with higher demand and improve services for those places, like repairing bikes, regularly checking on kiosks to work properly.
# </ul>
# 

# <p style="font-size: 2em; color: #C28CAE;"> Data Preparation </p>
# <p>Check the size of dataset:</p>

# In[ ]:


data_bike.shape


# <p> What kind of variables contribute our data: </p>

# In[ ]:


data_bike.dtypes


# In[ ]:


data_bike.head(2)


# Extract data from <strong>datetime</strong> column into new columns: date, hour, weekDay ,month.<br>
# (datetime structure: YYYY-MM-DD hh:mm:ss)<br>
# You may want to learn about <a  href="https://www.geeksforgeeks.org/python-string-split/">split</a>, <a href="https://www.geeksforgeeks.org/python-lambda-anonymous-functions-filter-map-reduce/">lambda</a>, <a href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.apply.html">apply</a>, <a href="https://docs.python.org/3/library/datetime.html">datetime</a>

# In[ ]:


data_bike["date"] = data_bike.datetime.apply(lambda x : x.split()[0])
data_bike["hour"] = data_bike.datetime.apply(lambda x : x.split()[1].split(":")[0])
data_bike["weekDay"] = data_bike.date.apply(lambda dateString : calendar.day_name[datetime.strptime(dateString,"%Y-%m-%d").weekday()])
data_bike["month"] = data_bike.date.apply(lambda dateString : calendar.month_name[datetime.strptime(dateString,"%Y-%m-%d").month])

# Convert to Category Type
categoryVariableList = ["hour","weekDay","month"]
for var in categoryVariableList:
    data_bike[var] = data_bike[var].astype("category")


# <strong>Optional Datatype Changing</strong><br>
# We can change datatype of "season", "holiday", "workingday" and "weather" columns to category or leave them the way they are. Btw the code for applying this conversion is provided below.

# In[ ]:


data_bike["season"] = data_bike.season.map({1: "Spring", 2 : "Summer", 3 : "Fall", 4 :"Winter" })
data_bike["weather"] = data_bike.weather.map({1: " Clear + Few clouds + Partly cloudy + Partly cloudy",                                        2 : " Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist ",                                         3 : " Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds",                                         4 :" Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog " })

# Convert to Category Type
categoryVariableList = ["season","weather","holiday","workingday"]
for var in categoryVariableList:
    data_bike[var] = data_bike[var].astype("category")


# <strong style="color: #0B6E4F">Find Missing Data</strong><br>
# <ul style="color: #08A045;">
#         <li> .info() .describe() .isnull()
# </ul>
# .info compare the total number of non-null with the total number of entries and check for missing values if we observe any difference between these two.<br>
# Then using .describe() check a summary statistics of observed features.<br>
# .isnull().sum() tells us the total number of NaN in our data.
# 
# <ul style="color: #6BBF59;">
#         <li> Missingno
# </ul>
# 
# Missingno is a great package to quickly display missing values in a dataset using a Matrix which shows patterns in data completion or a Bar Chart which visualize nullity by column. For more info you can check <a href="https://github.com/ResidentMario/missingno">this</a>.<br>
# We can also use Heatmap to measure how strongly the presence or absence of one variable affects the presence of another.

# In[ ]:


data_bike.info()


# This will display a summary statistics of all observed features and labels.

# In[ ]:


data_bike.describe()


# In[ ]:


data_bike.isnull().sum()


# In[ ]:


msno.matrix(data_bike)


# In[ ]:


data_bike.head()


# <p style="font-size: 1.5em; color: #FF7F11;">Outlier Analysis</p>
# We can find outliers using visulization methods(Box plot, Scatter plot) or mathematical methods(Z-Score, IQR score).<br>
# Using Plotly <a href="https://plot.ly/python/box-plots/">Box Plot</a> we demonstrate box plot of count, season, hour and ... .
# 

# In[ ]:


trace0 = go.Box(y=data_bike["count"], marker=dict(color='#9FA0FF'))
layout = go.Layout(title = 'Boxplot on Count')
fig0 = go.Figure(data=[trace0], layout=layout)
iplot(fig0)


# In[ ]:


trace1 = go.Box( y=data_bike["count"], x=data_bike["season"], marker=dict(color='#CC7E85'))
layout = go.Layout(title = 'Boxplot on Count across Season')
fig1 = go.Figure(data=[trace1], layout=layout)
iplot(fig1)


# In[ ]:


trace2 = go.Box( y=data_bike["count"], x=data_bike["hour"], marker=dict(color='#3F8EFC'))
layout = go.Layout(title = 'Boxplot on Count across Hour')
fig2 = go.Figure(data=[trace2], layout=layout)
iplot(fig2)


# Removing outliers from Count column.

# In[ ]:


data_bike_NoOutliers = data_bike[np.abs(data_bike["count"]-data_bike["count"].mean())<=(3*data_bike["count"].std())]


# In[ ]:


trace4 = go.Box(y=data_bike_NoOutliers["count"], marker=dict(color='#9FA0FF'))
layout = go.Layout(title = 'Boxplot on Count')
fig4 = go.Figure(data=[trace4], layout=layout)
iplot(fig4)


# <p style="font-size: 1.5em; color: #DA627D;">Correlation Analysis</p>
# This step is required to find out if there are possible connection between variables.
# It seems like we should expect relaitivity between Count and temp, humidity, windspeed, and maybe season, hour, month, weekday.

# In[ ]:


corrMatt = data_bike[["temp","atemp","casual","registered","humidity","windspeed","count"]].corr()


# In[ ]:


trace = go.Heatmap(z = corrMatt, 
                   x = ['temp','atemp','casual','registered','humidity','windspeed','count'], 
                   y = ['temp','atemp','casual','registered','humidity','windspeed','count'])
data=[trace]
iplot(data, filename='basic-heatmap')


# In[ ]:




