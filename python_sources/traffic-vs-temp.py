#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#imports
import numpy as np 
import pandas as pd 
import datetime as dt

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
sns.set_style("whitegrid")

import os
print(os.listdir("../input"))


# In[ ]:


df = pd.read_csv('../input/Metro_Interstate_Traffic_Volume.csv')
df.shape


# In[ ]:


df['date_time']=pd.to_datetime(df['date_time'])
df['dayofweek']=df['date_time'].dt.dayofweek
df=df[["weather_main","dayofweek","traffic_volume","temp","date_time"]]
df.columns


# In[ ]:


df.head()


# In[ ]:


df.describe(include='O')


# In[ ]:


df.describe()


# In[ ]:


df.isnull().sum()


# In[ ]:


df.weather_main.value_counts()


# In[ ]:


sns.barplot(x="weather_main", y="traffic_volume", data=df)
#min traffic in Squall season and highest traffic in Cloudy and Haze season


# In[ ]:


sns.barplot(x="dayofweek", y="traffic_volume", data=df) #0 is monday
#Minimum traffic on weekends


# In[ ]:


sns.barplot(x="weather_main", y="traffic_volume", hue ="dayofweek",data=df.loc[df.weather_main=='Clouds'])


# In[ ]:


sns.regplot(x="temp", y="traffic_volume", data=df)


# In[ ]:


df["year"] = df["date_time"].dt.year
df["month"] = df["date_time"].dt.month
df["week"] = df["date_time"].dt.week


# In[ ]:


sns.regplot(x="month", y="temp", data=df[df.temp>200])


# In[ ]:


sns.regplot(x="week", y="temp", data=df[df.temp>200])


# In[ ]:


df.year.value_counts()


# In[ ]:


df['week_modify'] = df.week


# In[ ]:


df.loc[df.year==2013,'week_modify'] = df.loc[df.year==2013,'week_modify'] + df.week_modify[df.year==2012].max()
df.loc[df.year==2014,'week_modify'] = df.loc[df.year==2014,'week_modify'] + df.week_modify[df.year==2013].max() 
df.loc[df.year==2015,'week_modify'] = df.loc[df.year==2015,'week_modify'] + df.week_modify[df.year==2014].max() 
df.loc[df.year==2016,'week_modify'] = df.loc[df.year==2016,'week_modify'] + df.week_modify[df.year==2015].max() 
df.loc[df.year==2017,'week_modify'] = df.loc[df.year==2017,'week_modify'] + df.week_modify[df.year==2016].max() 
df.loc[df.year==2018,'week_modify'] = df.loc[df.year==2018,'week_modify'] + df.week_modify[df.year==2017].max() 


# In[ ]:


df.loc[df.year==2018,'week_modify'].max()


# In[ ]:


sns.regplot(x="week_modify", y="temp", data=df[df.temp>200])


# In[ ]:


sns.regplot(x="week_modify", y="traffic_volume", data=df[(df.week_modify>40) & (df.week_modify<100)])


# In[ ]:


df["date"] = df["date_time"].dt.date


# In[ ]:


df['timedates'] = df['date_time'].map(lambda x: x.strftime('%Y-%m'))


# In[ ]:


def plot(x, y, data=None, label=None, **kwargs):
    sns.pointplot(x, y, data=data, label=label, **kwargs)

g = sns.FacetGrid(df, size=8, aspect=1.5)
g.map_dataframe(plot, 'timedates', 'traffic_volume')
plt.show()


# In[ ]:


def plot(x, y, data=None, label=None, **kwargs):
    sns.pointplot(x, y, hue='year',data=data, label=label, **kwargs)

g = sns.FacetGrid(df, size=8, aspect=1.5)
g.map_dataframe(plot, 'week', 'traffic_volume')
plt.show()


# In[ ]:


sns.set(rc={'figure.figsize':(16,9)})
sns.pointplot(x='week', y='traffic_volume', hue='year',data=df)


# In[ ]:


#Final
sns.set(rc={'figure.figsize':(16,9)})
sns.pointplot(x='week', y='traffic_volume', hue='year',errwidth=0,data=df)


# In[ ]:




