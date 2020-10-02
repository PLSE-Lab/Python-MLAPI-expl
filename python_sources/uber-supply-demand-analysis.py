#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import calendar

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Reading Input File and Creating a dataframe

# In[ ]:


# Reading Data
df = pd.read_csv('/kaggle/input/Uber Request Data.csv')
df.head()


# # Cleaning and imuting data

# In[ ]:


# Cleaning and imuting data
# Check for expected datetime columns, in this case; 'Request timestamp', 'Drop timestamp'
df['Request timestamp'] = pd.to_datetime(df['Request timestamp'])
df['Drop timestamp'] = pd.to_datetime(df['Drop timestamp'])

# dropping the duplicates -- though no duplicates present
df = df.drop_duplicates()

# Getting only the time Of Day of Pickup Point to view the frequency throughout the day
df['Request timeOfDay'] = df['Request timestamp'].dt.time
df['Request Date'] = df['Request timestamp'].dt.date

df['Request wkOfDay'] = df['Request timestamp'].dt.date.apply(lambda x: calendar.day_name[x.weekday()])

df['No of Requests'] = 1
df.head()


# In[ ]:


dfTotal = df.groupby(by = ['Status'])['Request id'].count().reset_index()
dfTotal.head()


# # Analysing Supply Demand Gap Over the dates given

# In[ ]:


dfGrp = df.groupby(by = ['Request Date', 'Status'])['No of Requests'].count().reset_index()
dfGrp

reqPrcntageList = list()
for x in dfGrp.index:
    #reqPrcntageList.append(sum(dfGrp['No of Requests'].loc[dfGrp['Request Date'] == dfGrp.loc[x, 'Request Date']]))
    reqPrcntageList.append(round(dfGrp.loc[x, 'No of Requests'] * 100 /  sum(dfGrp['No of Requests'].loc[dfGrp['Request Date'] == dfGrp.loc[x, 'Request Date']]), 2))

dfGrp['DayPercentage'] = reqPrcntageList
dfGrp

plt.figure(figsize=(10,8))
sns.barplot(x = 'Request Date', y = 'DayPercentage', hue = 'Status', data = dfGrp)
plt.title('Avg Uber Supply Demand Analysis per given date')


# # Analysing Demand Supply Gap on the day of the Week

# In[ ]:


dfGrp = df.groupby(by = ['Request wkOfDay', 'Status'])['No of Requests'].count().reset_index()


reqPrcntageList = list()
for x in dfGrp.index:
    #reqPrcntageList.append(sum(dfGrp['No of Requests'].loc[dfGrp['Request Date'] == dfGrp.loc[x, 'Request Date']]))
    reqPrcntageList.append(round(dfGrp.loc[x, 'No of Requests'] * 100 /  sum(dfGrp['No of Requests'].loc[dfGrp['Request wkOfDay'] == dfGrp.loc[x, 'Request wkOfDay']]), 2))

dfGrp['DayPercentagePerDayOfWk'] = reqPrcntageList
plt.figure(figsize=(10,8))
sns.barplot(x = 'Request wkOfDay', y = 'DayPercentagePerDayOfWk', hue = 'Status', data = dfGrp)
plt.title('Average Uber Supply Demand Analysis per the day of Week')


# ## Insights from above plot:
# ### In overall,
# #### Point 1: Thursday and Friday having most cars unavailability
# #### Point 2: Monday and Wednesdays serving most number of successful trip
# #### Point 3: Of all days, Cancellation are steady at below 20%

# # Analysing Demand Supply Gap over the day of week - From Airport to City

# In[ ]:


# Analysing Cars availability from 'Airport To City' and 'City To Airport' prespective.

dfGrp = df.groupby(by = ['Request wkOfDay', 'Status', 'Pickup point'])['No of Requests'].count().reset_index()
dfGrpAirport = dfGrp.loc[dfGrp['Pickup point'] == 'Airport'].reset_index()

reqPrcntageList = list()
for x in dfGrpAirport.index:
    #reqPrcntageList.append(sum(dfGrp['No of Requests'].loc[dfGrp['Request Date'] == dfGrp.loc[x, 'Request Date']]))
    reqPrcntageList.append(round(dfGrpAirport.loc[x, 'No of Requests'] * 100 /  sum(dfGrpAirport['No of Requests'].loc[dfGrpAirport['Request wkOfDay'] == dfGrpAirport.loc[x, 'Request wkOfDay']]), 2))

dfGrpAirport['DayPercentage'] = reqPrcntageList
dfGrpAirport

plt.figure(figsize=(10,8))
sns.barplot(x = 'Request wkOfDay', y = 'DayPercentage', hue = 'Status', data = dfGrpAirport)
plt.title('Cars availability on day of the week - From "Airport" to "City"')
plt.ylabel('Avg calcuations per day')


# ## Insights from above plot:
# ### Observations:
# #### 1. Cancellations at Airport is much less compared to City
# #### 2. Usually Thursdays at airport having more cars unavailability and less successful trips compared to other days

# # Analysing Demand Supply Gap over the day of week - From City To Airport

# In[ ]:


dfGrp = df.groupby(by = ['Request wkOfDay', 'Status', 'Pickup point'])['No of Requests'].count().reset_index()
dfGrpCity = dfGrp.loc[dfGrp['Pickup point'] == 'City'].reset_index()

reqPrcntageList = list()
for x in dfGrpCity.index:
    reqPrcntageList.append(round(dfGrpCity.loc[x, 'No of Requests'] * 100 /  sum(dfGrpCity['No of Requests'].loc[dfGrpCity['Request wkOfDay'] == dfGrpCity.loc[x, 'Request wkOfDay']]), 2))

dfGrpCity['DayPercentage'] = reqPrcntageList
dfGrpCity

plt.figure(figsize=(10,8))
sns.barplot(x = 'Request wkOfDay', y = 'DayPercentage', hue = 'Status', data = dfGrpCity)
plt.title('Cars availability on day of the week - From "City" to "Airport"')
plt.ylabel('Avg calcuations per day')


# ## Insights:
# ### Observations:
# #### 1. Usually successful trips are more from City to Airport; compared to Airport to City
# ##### 1. Question to Ask: Why cars are returning empty
# ### 2. Cancellations to go to the airport are more; compared to no cars availability
# ### 3. Monday being the huge money maker from City to Airport

# # Analysing typical cars supply demand stats throughout the day

# In[ ]:


df['Request Hour'] = df['Request timestamp'].dt.hour + 1
dfGrp = df.groupby(by = ['Status', 'Request Hour'])['No of Requests'].count().unstack()

dfGrp = dfGrp.T
dfGrp['Total Requests'] = dfGrp['Cancelled'] + dfGrp['No Cars Available'] + dfGrp['Trip Completed']
dfGrp.plot(kind = 'bar', figsize=(10,8))
plt.title('Total pickup orders Vs Supply Demand Plot')
plt.ylabel('Total number of orders')


# In[ ]:


dfGrp['Supply Gap'] = dfGrp['Total Requests'] - dfGrp['Trip Completed']
dfGrpSubSet = dfGrp.loc[:, ['Total Requests', 'Supply Gap']]
dfGrpSubSet.plot(kind = 'bar', figsize=(10,8))
plt.ylabel('Total number of orders')
plt.title('Exact Supply Demand Gap')


# ## Insights:
# ### Observations:
# #### 1. Most number of Pickup orders are received from 4AM onwards till 11AM; and then 5AM onwards till 11PM
# #### 2. In morning high peak time slot, we have more number of Cancellations and have same level number of Successful Trips too.
# #### 3. Whereas in evening high peak time slot, we have more cars unavailability whereas Successful trips are being maintained at steady level throughout the hours of day

# # Analysing typical Supply Demand plot for the course of day from/to Airport/City
# ## ============== FROM AIRPORT TO CITY ==============

# In[ ]:


dfGrp1 = df.groupby(by = ['Pickup point', 'Status', 'Request Hour'])['No of Requests'].count().unstack()
dfGrp1 = dfGrp1.fillna(0)
dfGrp1


# In[ ]:


dfGrp1.loc['Airport'].T.plot(kind = 'bar', figsize = (10,8))
plt.ylabel('Number of requests')
plt.title('Supply Demand plot for the day at Airport')


# In[ ]:


dfGrpAirport = dfGrp1.loc['Airport'].T

dfGrpAirport['Total Requests'] = dfGrpAirport['Cancelled'] +  dfGrpAirport['No Cars Available'] + dfGrpAirport['Trip Completed']
dfGrpAirport['SupplyGap'] = dfGrpAirport['Total Requests'] - dfGrpAirport['Cancelled'] +  dfGrpAirport['No Cars Available']
dfGrpAirport.head()

dfGrpAirport.loc[:, ['Total Requests', 'SupplyGap']].plot(kind = 'bar', figsize=(10,8))
plt.title('Demand Supply Gap analysis from "Airport" to "City"')
plt.ylabel('Total number of requests')


# In[ ]:


dfGrp1.loc['City'].T.plot(kind = 'bar', figsize=(10,8))
plt.ylabel('Number of requests')
plt.title('Supply Demand plot for the day at City')


# In[ ]:


dfGrpCity = dfGrp1.loc['City'].T

dfGrpCity['Total Requests'] = dfGrpCity['Cancelled'] +  dfGrpCity['No Cars Available'] + dfGrpCity['Trip Completed']
dfGrpCity['SupplyGap'] = dfGrpCity['Total Requests'] - dfGrpCity['Cancelled'] +  dfGrpCity['No Cars Available']
dfGrpCity.head()

dfGrpCity.loc[:, ['Total Requests', 'SupplyGap']].plot(kind = 'bar', figsize=(10,8))
plt.title('Demand Supply Gap analysis from "City" to "Airport"')
plt.ylabel('Total number of requests')

