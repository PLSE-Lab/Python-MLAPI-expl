#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import re
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv("/kaggle/input/air-pollution-dataset-india-2019/combined.csv", low_memory=False)


# In[ ]:


df.head()


# In[ ]:


def lookup(s):
    """
    This is an extremely fast approach to datetime parsing.
    For large data, the same dates are often repeated. Rather than
    re-parse these, we store all unique dates, parse them, and
    use a lookup to convert all dates.
    """
    dates = {date:pd.to_datetime(date) for date in s.unique()}
    return s.map(dates)


# In[ ]:


df['datetime'] = lookup(df['datetime'])


# In[ ]:


#df.set_index('datetime', inplace=True)


# In[ ]:


df.head()


# In[ ]:


df.info(memory_usage='deep')


# In[ ]:


df.describe()


# In[ ]:


df.drop('live', axis=1, inplace=True) #dropping unwanted column


# In[ ]:


df.shape


# ### Pollutant Types

# #### PM2.5 and PM10
# PM2.5 refers to the atmospheric particulate matter that has a diameter of less than 2.5 micrometres, which is about 3% of the diameter of human hair.
# 
# The particles in PM2.5 category are so small that they can only be detected with the help of the electron microscope. These are smaller than PM10 particles. PM10 are the particles with a diameter of 10 micrometers and they are also called fine particles. An environmental expert says that PM10 is also known as respirable particulate matter.

# #### SO2 and NO2 pullutants
# The most important sources of SO2 are fossil fuel combustion, smelting, manufacture of sulphuric acid, conversion of wood pulp to paper, incineration of refuse and production of elemental sulphur. Coal burning is the single largest man-made source of SO2 accounting for about 50% of annual global emissions, with oil burning accounting for a further 25-30%.
# 
# Globally, quantities of nitrogen oxides produced naturally (by bacterial and volcanic action and lightning) far outweigh anthropogenic (man-made) emissions. Anthropogenic emissions are mainly due to fossil fuel combustion from both stationary sources, i.e. power generation (21%), and mobile sources, i.e. transport (44%). Other atmospheric contributions come from non-combustion processes, for example nitric acid manufacture, welding processes and the use of explosives.
# 
# Nitrogen dioxide (NO2), sulfur dioxide (SO2), and carbon monoxide are important ambient air pollutants. High-intensity, confined space exposure to NO2 has caused catastrophic injury to humans, including death. Ambient NO2 exposure may increase the risk of respiratory tract infections through the pollutant's interaction with the immune system. Sulfur dioxide (SO2) contributes to respiratory symptoms in both healthy patients and those with underlying pulmonary disease.

# #### CO
# The major source of atmospheric CO is the spark ignition combustion engine. Smaller contributions come from processes involving the combustion of organic matter, for example in power stations and waste incineration.

# #### OZONE
# Most O3 in the troposphere (lower atmosphere) is formed indirectly by the action of sunlight on nitrogen dioxide - there are no direct emissions of O3 to the atmosphere. About 10 - 15% of tropospheric O3 is transported from the stratosphere where it is formed by the action of ultraviolet (UV) radiation on O2.

# ### Fillna values by the mean of the day and mean of the week

# In[ ]:


print(df['SO2'].isna().sum())
print(df['NO2'].isna().sum())
print(df['NH3'].isna().sum())
print(df['CO'].isna().sum())
print(df['OZONE'].isna().sum())


# In[ ]:


#mean of the day
dayMeandf =  df.loc[:,['id','SO2','NO2','NH3','CO','OZONE']].groupby([df['datetime'].dt.dayofyear ,'id']).transform('mean')


# In[ ]:


df['SO2'].fillna(dayMeandf['SO2'], inplace=True)
df['NO2'].fillna(dayMeandf['NO2'], inplace=True)
df['NH3'].fillna(dayMeandf['NH3'], inplace=True)
df['CO'].fillna(dayMeandf['CO'], inplace=True)
df['OZONE'].fillna(dayMeandf['OZONE'], inplace=True)


# In[ ]:


#mean of the week
weekMeandf =  df.loc[:,['id','SO2','NO2','NH3','CO','OZONE']].groupby([df['datetime'].dt.weekofyear ,'id']).transform('mean')


# In[ ]:


df['SO2'].fillna(weekMeandf['SO2'], inplace=True)
df['NO2'].fillna(weekMeandf['NO2'], inplace=True)
df['NH3'].fillna(weekMeandf['NH3'], inplace=True)
df['CO'].fillna(weekMeandf['CO'], inplace=True)
df['OZONE'].fillna(weekMeandf['OZONE'], inplace=True)


# In[ ]:


print(df['SO2'].isna().sum())
print(df['NO2'].isna().sum())
print(df['NH3'].isna().sum())
print(df['CO'].isna().sum())
print(df['OZONE'].isna().sum())


# In[ ]:


df.dropna(how='all', inplace=True)


# ### Number of stations per state

# In[ ]:


#number of state
df['stateid'].unique()


# In[ ]:


def myfunc(x):
    return (x['id'].nunique())

stationsPerState = df.loc[:,['stateid','id']].groupby('stateid').apply(myfunc).reset_index()


# In[ ]:


stationsPerState.rename({0:'count'}, axis=1, inplace=True)


# In[ ]:


stationsPerState.sort_values('count', ascending=False, inplace=True)
stationsPerState


# In[ ]:


fig = plt.figure(figsize=[10,4])
axes = fig.add_axes([0,0,1,1])
axes.bar(stationsPerState['stateid'], stationsPerState['count'])
fig.autofmt_xdate(rotation=60)
axes.set_xlabel('State',{"fontsize":12})
axes.set_ylabel('Number of stations', {"fontsize":12})
axes.set_title('Number of stations per state')


# In[ ]:


def myfunc(x):
    return (x['id'].nunique())

stationsByState = df.loc[:,['stateid','cityid','id']].groupby(['stateid','cityid']).apply(myfunc)
stationsByState


# In[ ]:


fig = plt.figure(figsize=[20,20])
axes = fig.subplots(7,3)
axes = axes.flatten()
fig.tight_layout()
fig.subplots_adjust(hspace=1, wspace=0.2)

for i,state in enumerate(stationsByState.index.get_level_values(0).unique()):
    axes[i].bar(stationsByState[state].index,stationsByState[state].values)
    axes[i].tick_params(axis='x', labelrotation=45)
    axes[i].set_xlabel(state,{"fontsize":14})
    axes[i].set_ylabel("stations", {"fontsize":14})


# ### States with High Pollution

# In[ ]:


averageByState = df.loc[:,['stateid','PM2.5', 'PM10', 'NO2', 'NH3', 'SO2', 'CO', 'OZONE']].groupby('stateid', as_index=False).mean()
averageByState


# In[ ]:


fig = plt.figure(figsize=[20,50])
pollutants = ['PM2.5', 'PM10', 'NO2', 'NH3', 'SO2', 'CO', 'OZONE']
colors = ['#00965d','#829600', '#dbeb34', '#34ebd3', '#004496','#7a0096','#960052']
axes = fig.subplots(len(pollutants))
fig.subplots_adjust(hspace=0.4)

for i,pollutant in enumerate(pollutants):
    axes[i].bar(averageByState['stateid'], averageByState[pollutant], color=colors[i])
    axes[i].tick_params(axis='x', labelrotation=45)
    axes[i].set_xlabel('States', {"fontsize":16})
    axes[i].set_ylabel('value', {"fontsize":16})
    axes[i].set_title(pollutant, {"fontsize":18})


# In[ ]:


fig = plt.figure(figsize=[20,6])
axes = fig.add_axes([0,0,1,1])
axes.plot(averageByState['stateid'], averageByState.loc[:,['PM2.5', 'PM10', 'NO2', 'NH3', 'SO2', 'CO', 'OZONE']])
axes.tick_params(axis='x', labelrotation=45)
axes.legend(labels=['PM2.5', 'PM10', 'NO2', 'NH3', 'SO2', 'CO', 'OZONE'],prop={'size': 18})
axes.set_xlabel('State', {"fontsize":18})
axes.set_ylabel('Pollutant value', {"fontsize":18})


# In[ ]:


averageByState.sort_values(['PM2.5', 'PM10', 'NO2', 'NH3', 'SO2', 'CO', 'OZONE'], ascending=False).head()


# In[ ]:


averageByState.sort_values(['PM2.5', 'PM10', 'NO2', 'NH3', 'SO2', 'CO', 'OZONE'], ascending=True).head()


# ### Top 5 states are
# 'Delhi', 'Uttar Pradesh', 'Bihar', 'Haryana', 'Chandigarh'

# ### Bottom 5 states are
# 'Meghalaya ', 'Kerala', 'Karnataka', 'Andhra Pradesh', 'Maharashtra'

# ### Is seasonality exists in the pollution?

# In[ ]:


df['month'] = df['datetime'].dt.month_name()


# In[ ]:


df['month'] = df['month'].astype(pd.CategoricalDtype(['January', 'February', 'March', 'April', 'May', 'June', 'July' ,'August', 'September', 'October', 'November', 'December'], ordered=True))


# In[ ]:


monthlyAverage = df.loc[:,['month','PM2.5', 'PM10', 'NO2', 'NH3', 'SO2', 'CO', 'OZONE']].groupby('month', as_index=False).mean()
monthlyAverage


# In[ ]:


fig = plt.figure(figsize=[20,6])
axes = fig.subplots()
axes.scatter(monthlyAverage['month'],monthlyAverage['PM2.5'], label = 'PM2.5')
axes.scatter(monthlyAverage['month'],monthlyAverage['PM10'], label= 'PM10')
axes.scatter(monthlyAverage['month'],monthlyAverage['SO2'], label='SO2')
axes.scatter(monthlyAverage['month'],monthlyAverage['NO2'], label='NO2')
axes.scatter(monthlyAverage['month'],monthlyAverage['NH3'], label='NH3')
axes.scatter(monthlyAverage['month'],monthlyAverage['CO'], label= 'CO')
axes.scatter(monthlyAverage['month'],monthlyAverage['OZONE'], label='OZONE')
axes.legend()


# ### Pollution is low during august and september and high during winters (November, December, January)

# ### What time of the day has more pollution ?

# In[ ]:


df['hour'] = df['datetime'].dt.hour


# In[ ]:


df['hour'] = df['hour'].astype(pd.CategoricalDtype([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23], ordered=True))


# In[ ]:


hourlyAverage = df.loc[:,['hour','PM2.5', 'PM10', 'NO2', 'NH3', 'SO2', 'CO', 'OZONE']].groupby('hour', as_index=False).mean()


# In[ ]:


fig = plt.figure(figsize=[20,6])
axes = fig.subplots()
axes.scatter(hourlyAverage['hour'],hourlyAverage['PM2.5'], label = 'PM2.5')
axes.scatter(hourlyAverage['hour'],hourlyAverage['PM10'], label= 'PM10')
axes.scatter(hourlyAverage['hour'],hourlyAverage['SO2'], label='SO2')
axes.scatter(hourlyAverage['hour'],hourlyAverage['NO2'], label='NO2')
axes.scatter(hourlyAverage['hour'],hourlyAverage['NH3'], label='NH3')
axes.scatter(hourlyAverage['hour'],hourlyAverage['CO'], label= 'CO')
axes.scatter(hourlyAverage['hour'],hourlyAverage['OZONE'], label='OZONE')
axes.set_xticks(range(24))
axes.set_xlabel('hours', {"fontsize":16})
axes.set_ylabel('value', {"fontsize":16})
axes.set_title('Pollution by Hour', {"fontsize":18})
axes.legend()


# #### PM2.5 and PM10 pollutants are low during 4:00 PM and 5:00 PM and O3(OZONE) pollutants are high during 2:00 PM and 3:00 PM
# #### Perfect time for jogging or outdoor activities is during 4:00 PM - 5:00 PM or 5:00 AM - 6:00 AM

# ### Pollution on weekends vs weekdays

# In[ ]:


df['dayoftheweek'] = df['datetime'].dt.dayofweek


# In[ ]:


df['weekend'] = 'weekday'
df.loc[df['dayoftheweek'].isin([5,6]),'weekend'] = 'weekend'


# In[ ]:


df['weekend'] = df['weekend'].astype(pd.CategoricalDtype(['weekday','weekend']))


# In[ ]:


weekendGrp = df.loc[:,['weekend','PM2.5', 'PM10', 'NO2', 'NH3', 'SO2', 'CO', 'OZONE']].groupby('weekend', as_index=False).mean()
weekendGrp


# In[ ]:


weekendGrp[weekendGrp['weekend'] == 'weekend'].loc[:,['PM10', 'NO2', 'NH3', 'SO2', 'CO', 'OZONE']].values[0]


# In[ ]:


fig = plt.figure(figsize=[20,6])
axes = fig.subplots()
axes.plot(['PM10', 'NO2', 'NH3', 'SO2', 'CO', 'OZONE'],weekendGrp[weekendGrp['weekend'] == 'weekend'].loc[:,['PM10', 'NO2', 'NH3', 'SO2', 'CO', 'OZONE']].values[0], label='weekend')
axes.plot(['PM10', 'NO2', 'NH3', 'SO2', 'CO', 'OZONE'],weekendGrp[weekendGrp['weekend'] == 'weekday'].loc[:,['PM10', 'NO2', 'NH3', 'SO2', 'CO', 'OZONE']].values[0], label='weekday')
axes.legend(prop={'size':18})
axes.set_xlabel('Pollutant', {"fontsize":18})
axes.set_ylabel('value', {"fontsize":18})
axes.set_title('Pollution on weekdays vs weekends')


# #### It seems pollution is almost equal during both weekends and weekdays

# ### Pollution during festive seasons
# ###### In India, the festivals which causes air pollution are New Year and Deepavali
# ###### Deepavali is on Sunday, 27 October 2019 and New year is on 01 January

# In[ ]:


df[df['datetime'] == '2019-10-27'].mean()


# In[ ]:


df[df['datetime'] == '2019-01-01'].mean()


# In[ ]:




