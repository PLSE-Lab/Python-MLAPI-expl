#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **Reading the data**

# In[ ]:


data = pd.read_csv("../input/accidents_2017.csv")


# **Replacing the null values**

# In[ ]:


for col_name in data.columns:
    if data[col_name].dtype == "object":
        mode_var = data[col_name].mode()[0]
        data[col_name].fillna(mode_var,inplace = True)
    else:
        mean_var = data[col_name].mean()
        data[col_name].fillna(mean_var,inplace = True)


# **Column Names**

# In[ ]:


data.columns


# **Number of columns and rows in the dataset**

# In[ ]:


data.shape


# **Importing Plotyly library**

# In[ ]:


import plotly.graph_objs as go 
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot


# **Statistics of particular variables of dataset**

# In[ ]:


data[['Mild injuries','Serious injuries','Victims','Vehicles involved']].describe()


# **Importing necessary libraries**

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plotly import __version__
import plotly.graph_objs as go 
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import cufflinks as cf
cf.go_offline()
get_ipython().run_line_magic('matplotlib', 'inline')


# **Sorting of data**

# In[ ]:


part_day_data = data[['Vehicles involved','Victims','Part of the day','Mild injuries','Serious injuries',]]
part_day_data = part_day_data.set_index('Part of the day')
part_day_data = part_day_data.groupby(level=[0]).sum()
part_day_data = part_day_data.sort_index()
part_day_data.head()


# **Bar chart showing details of accidents by part of the day**

# In[ ]:


part_day_data = part_day_data.reindex(['Morning','Afternoon','Night'])
layout = dict(title='Details of Accidents by part of the day',geo=dict(showframe=False))
part_day_data.iplot(kind='bar',layout=layout, color=['red', 'orange', 'green', 'blue'])


# ** Finding**
# 1. Highest number of accidents have occured during the afternoon part of the day.
# 2. Highest number of vehicles involved in the accidents were found to be 9746.
# 3. The injuries caused were mostly Mild.
# 

# **Sorting the data**

# In[ ]:


hour_day_data = data[['Hour','Vehicles involved','Victims','Mild injuries','Serious injuries',]]
hour_day_data = hour_day_data.set_index('Hour')
hour_day_data = hour_day_data.groupby(level=[0]).sum()
hour_day_data = hour_day_data.sort_index()
hour_day_data.head()


# **Bar chart showing details of accidents by the hour of the day**

# 

# In[ ]:


layout = dict(title='Details of Accidents by hour of the day',geo=dict(showframe=False))
hour_day_data.iplot(kind='bar',layout=layout,color=['red', 'orange', 'green', 'blue'])


# **Findings**
# 1. The no of accidents observed during the midnight is lowest.
# 2. Where as the accident count increses drastically from the office time.
# 3. The occurance of accident reached to its peak point at the afternoon.
# 4. We found a slight increase in the occurance at the evening and again decreased by the night.  

# **Sorting of data**

# In[ ]:


weekday_data = data[['Weekday','Vehicles involved','Victims','Mild injuries','Serious injuries',]]
weekday_data = weekday_data.set_index('Weekday')
weekday_data = weekday_data.groupby(level=[0]).sum()
weekday_data.head(7)


# **Bar Chart showing details of accidents by weekday**

# In[ ]:


weekday_data = weekday_data.reindex(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
layout = dict(title='Details of Accidents by weekday',geo=dict(showframe=False))
weekday_data.iplot(kind='bar',layout=layout,color=['red', 'orange', 'green', 'blue'])


# **Findings**
# 1. The occurance of accidents were found out to be max during the weekdays compare to weekends.
# 2. Among the weekdays the highest no of victims were 2031 which was on Friday.
# 3. The count of accidents, victims, no of injuries was found relatively low on the weekends.

# **Sorting of data**

# In[ ]:


month_data = data[['Month','Vehicles involved','Victims','Mild injuries','Serious injuries',]]
month_data = month_data.set_index('Month')
month_data = month_data.groupby(level=[0]).sum()
month_data = month_data.sort_index()
month_data.head()


# **Bar Graphs showing details of accidents by the months**

# In[ ]:


month_data = month_data.reindex(['January',
                                 'February',
                                 'March',
                                 'April',
                                 'May',
                                 'June',
                                 'July',
                                 'August',
                                 'September',
                                 'October',
                                 'November',
                                 'December'])
layout = dict(title='Details of Accidents by Month',geo=dict(showframe=False))
month_data.iplot(kind='bar',layout=layout,color=['red', 'orange', 'green', 'blue'])


# **Findings**
# 1. The occurence of vehicle involved accidents was found to be almost equal through out the year.
# 2. In certain months the no of Victims are fallen down due to public holidays and football premier league.
# 

# **Line graph between day and victims**

# In[ ]:



plt.figure(figsize = (30,15))
data.groupby('Day')['Victims'].count().plot(kind='line')


# In[ ]:


plt.figure(figsize = (15,10))
plt.bar(data['District Name'],data['Mild injuries'],label='Mild injuries', color='blue')
plt.bar(data['District Name'],data['Serious injuries'],label='Serious injuries', color='green')
plt.legend()


# **Pie Chart describing the classification of injuries**

# In[ ]:


data[['Mild injuries','Serious injuries']].sum()


# In[ ]:


import matplotlib.pyplot as plt
plt.figure(figsize=(8,8))
labels = ["Mild Injuries","Serious Injuries"]
values = [11933,241]
colors=['blue','green']
plt.pie(values, labels=labels, colors=colors)
plt.show()


# **Findings**
# Out of the total injuries maximum injuries were mild, hence we can conclude that most of the accidents occured was not so fatal. 

# **Importing Matplot**

# In[ ]:


import matplotlib.pyplot as plt
import geopandas as gpd

get_ipython().run_line_magic('matplotlib', 'inline')


# **Scatter Plot for Latitude and longitude **

# In[ ]:


data.plot(kind="scatter",label='Victims', x="Longitude", y="Latitude", alpha=0.4)
plt.show()

