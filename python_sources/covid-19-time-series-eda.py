#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv("/kaggle/input/corona-virus-report/covid_19_clean_complete.csv")


# In[ ]:


df


# In[ ]:


df.info()


# lets convert 'Date' to datetime object

# In[ ]:


df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=False)


# In[ ]:


df.info()


# lets group data together by country

# In[ ]:


df1 = df.groupby(df["Country/Region"])


# In[ ]:


df1.head(10)


# Now to see the name of the groups

# In[ ]:


df1.groups.keys()


# the total number of countries in the world right now is 185, but the length of df1 is 1850, also the keys in df1 are 185 only, so lets investigate the additional rows

# In[ ]:


df1.size().sort_values()   #To count rows in each group - group.size()


# To select a particular group use - get_group

# In[ ]:


df1.get_group('China')


# the data for china is added on the basis of provinces for each day, its for some other countries as well,.. will combine this data later when we'll use it.

# To group data by month, we can use a very handy method called pd.grouper which is extremely useful in time series data

# In[ ]:


by_month = df.groupby(pd.Grouper(key='Date',freq='M')).size()


# In[ ]:


by_month


# In[ ]:


plt.plot(by_month)


# to get the latest numbers

# In[ ]:


latest = df.drop(['Lat','Long'],axis=1)


# In[ ]:


daily_latest = latest.groupby(pd.Grouper(key='Date',freq = 'D')).sum()


# In[ ]:


daily_latest


# the total number of cases in world are 2,472,253 out of which there are 169,985 deaths and 633,181 recovered

# In[ ]:


# fatality_rate = 
conf = daily_latest['Confirmed'][-1]
death = daily_latest['Deaths'][-1]
rec = daily_latest['Recovered'][-1]
fatality_rate = (death/conf)*100
survival_rate = (rec/conf)*100
print("Fatality rate: ", fatality_rate)
print("Survival rate: ", survival_rate)


# In[ ]:


labels = 'Deaths', 'Recovered', 'Current Cases'
sizes = [fatality_rate, survival_rate, 100-(fatality_rate+survival_rate)]
explode = (0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()


# In[ ]:


plt.plot(daily_latest)


# as we can see in plot the graph is increaing exponentially

# In[ ]:


weekly_data = latest.groupby(pd.Grouper(key='Date',freq = 'W')).sum()


# In[ ]:


weekly_data


# In[ ]:


weekly_spike = weekly_data.diff(axis=0)  #to get the growth in number of cases between different weeks


# In[ ]:


weekly_spike


# In[ ]:


weekly_spike.dropna()  #to drop the first row that contains nan value


# In[ ]:


weekly_spike.plot.bar(rot=0,subplots=True)


# In[ ]:


plt.plot(weekly_spike)


# Now lets do daily analysis on data

# In[ ]:


df2 = df.groupby(df["Date"])


# In[ ]:


df2.head()


# In[ ]:


df2.groups.keys()


# we can see that data has been grouped by date, now lets select the latest date

# In[ ]:


total_cases = df2.get_group(df['Date'].iloc[-1])


# In[ ]:


total_cases


# The number of rows are 262, but it should be 185(number of countries)

# In[ ]:


total_cases = total_cases.groupby(total_cases['Country/Region']).sum()  #to sum up the total cases in the country, by different provinces in the county as well


# In[ ]:


total_cases


# In[ ]:


total_cases = total_cases.sort_values('Confirmed',ascending=False)      #sort such that highest confirmed cases are shown first


# In[ ]:


total_cases.head(10)


# here we can see the top 10 countries which are worst affected

# In[ ]:


top_10 = total_cases.head(10).drop(['Lat','Long'],axis=1)


# In[ ]:


top_10.plot.bar()


# In[ ]:




