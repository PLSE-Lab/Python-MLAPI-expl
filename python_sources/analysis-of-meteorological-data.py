#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


file_path = '../input/weather-dataset/weatherHistory.csv'


# This dataset provides historical data on many meteorological parameters 
# such as pressure, temperature, humidity, wind_speed, visibility, etc. The dataset has hourly temperature recorded for last 10 years starting from 2006-04-01 00:00:00.000 +0200 to 2016-09-09 23:00:00.000 +0200. It corresponds to Finland, a country in the Northern Europe.

# In[ ]:



df = pd.read_csv(file_path)
df.shape #96453 records and 12 columns


# In[ ]:


df.dtypes


# But before visualization, we need to make date features -> date time object . 
# For this we use to_datetime() fn

# In[ ]:


df['Formatted Date'] = pd.to_datetime(df['Formatted Date'], utc=True)
df['Formatted Date']


# In[ ]:


df.dtypes


# In[ ]:


df = df.set_index('Formatted Date')
df.head()


# Now since we have been given hourly data, we need to resample it monthly. *Resampling is a convenient method for frequency conversion*.
# *Object must have a datetime like index*

# **After resampling:**

# In[ ]:


data_columns = ['Apparent Temperature (C)', 'Humidity']
df_monthly_mean = df[data_columns].resample('MS').mean()
df_monthly_mean.head()


# Here ***"MS"*** denotes: Month starting
# We are displaying the average apparent temperature and humidity using mean() function.

# ### Plotting the variation in Apparent Temperature and Humidity with time

# In[ ]:


import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
plt.figure(figsize=(14,6))
plt.title("Variation in Apparent Temperature and Humidity with time")
sns.lineplot(data=df_monthly_mean)


# **Observation **: From the above plot, we can say that humidity remained almost constant in these years. Even the average apparent temperature is almost same (since peaks lie on the same line)

# If we want to specifically retrieve the data of a particular month from every year, say April in this case then :

# In[ ]:


df1 = df_monthly_mean[df_monthly_mean.index.month==4]
print(df1)
df1.dtypes


# Plotting the variation in Apparent Temperature and Humidity for the month of April every year:

# In[ ]:


import matplotlib.dates as mdates
fig, ax = plt.subplots(figsize=(15,5))
ax.plot(df1.loc['2006-04-01':'2016-04-01', 'Apparent Temperature (C)'], marker='o', linestyle='-',label='Apparent Temperature (C)')
ax.plot(df1.loc['2006-04-01':'2016-04-01', 'Humidity'], marker='o', linestyle='-',label='Humidity')
ax.set_xticks(['04-01-2006','04-01-2007','04-01-2008','04-01-2009','04-01-2010','04-01-2011','04-01-2012','04-01-2013','04-01-2014','04-01-2015','04-01-2016'])
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %m %Y'))
ax.legend(loc = 'center right')
ax.set_xlabel('Month of April')


# **Observation **: No change in average humidity. Increase in average apparent temperature can be seen in the year 2009 then again it dropped in 2010 then there was a slight increase in 2011 then a significant drop is observed in 2015 and again it increased in 2016 .

# In[ ]:




