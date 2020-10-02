#!/usr/bin/env python
# coding: utf-8

# <h2> Were we heading for an ice-age ?</h2>
# 
# As a young kid I grew up in Toronto and I heard that we were heading for an ice age. Was this possible ?
# 
# A simple check on the data shows that prior to our "Global Warming" phase, the yearly temperatures were going down!!!

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Load the CSV file and do some clean-up
weather = pd.read_csv('../input/Toronto_temp.csv')
weather['Year'] = weather['Year'].str.replace(',','')
weather = weather.query('Year > "1937"') #Note: 1937 has some bad data

# Get the yearly averages                      
to_avg = weather[['Year','Mean Temp (C)']].groupby(['Year'], as_index=False).mean()

to_avg.plot(kind='line',x='Year', y=['Mean Temp (C)'], grid='on',title="Toronto Average Temperatures", figsize=(15, 8)  )
plt.text(15,9,"Between 1950 and 1972 the \ntemperature was trending down")
plt.plot([15, 30], [8.5, 6.6], color='red')
plt.text(45,10.5,"After 1978 the temperature was started trending up")
plt.plot([45, 85], [7.2, 10], color='red')

plt.show()


# It is interesting to note that in recent years Toronto has had record hot days but it also has had record cold days.

# In[ ]:


# Get the summer yearly averages
summer = weather.query('season == "Summer"')
summer.rename(columns={'Max Temp (C)':'Summer Temp'}, inplace = True)                      
sum_avg = summer[['Year','Summer Temp']].groupby(['Year'], as_index=False).max()

# Get the winter yearly averages
winter = weather.query('season == "Winter"')
winter.rename(columns={'Min Temp (C)':'Winter Temp'}, inplace = True)
win_avg = winter[['Year','Winter Temp']].groupby(['Year'], as_index=False).min()

# Merge the winter and summer averages

year_avg = win_avg
year_avg = year_avg.merge(sum_avg, on= 'Year')

year_avg.plot(subplots=True,kind='line',x='Year', secondary_y=['Summer Temp', 'Winter Temp'],mark_right=False, grid='on',
              title="Toronto Max/Min Temperatures",sharex=True,figsize=(15, 8))

plt.text(69,23,"Hotest/Coldest temps \nare after 2012 ")
plt.show()


# If we look at the summer and winter temperatures independently, we see that there is a almost a predictable cycling of temperatures. So the 2019 and 2020 winters could be cold.

# In[ ]:




# Get the summer yearly averages
summer = weather.query('season == "Summer"')
summer.rename(columns={'Mean Temp (C)':'Summer Temp'}, inplace = True)                      
sum_avg = summer[['Year','Summer Temp']].groupby(['Year'], as_index=False).mean()

# Get the winter yearly averages
winter = weather.query('season == "Winter"')
winter.rename(columns={'Mean Temp (C)':'Winter Temp'}, inplace = True)
win_avg = winter[['Year','Winter Temp']].groupby(['Year'], as_index=False).mean()

# Merge the winter and summer averages

year_avg = win_avg
year_avg = year_avg.merge(sum_avg, on= 'Year')

year_avg.plot(subplots=True,kind='line',x='Year', secondary_y=['Summer Temp', 'Winter Temp'],mark_right=False, grid='on',
              title="Toronto Average Yearly Temperatures",sharex=True,figsize=(15, 8))

plt.text(60,18,"There appears to be ~3 year\ncycles in the weather ")
plt.show()                          


# <h2>Summary</h2>
# 
# Well the old timers might have been right...we could have been heading towards an ice age between 1950 and 1970. 
# 
# Since 1972 there appears to be a definite warming trend, but along with it are coming some extreme spikes for both record hot and record cold days. Toronto has had it's coldest and hotest days in the past 5 years.
