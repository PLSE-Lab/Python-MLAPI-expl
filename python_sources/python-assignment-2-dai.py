#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt


# ## Q.1 Code for solving two variable linear equation 

# In[ ]:


# for two equation of the form a1x+b1y+c1=0 & a2x+b2y+c2=0 below written function will 
# take input for all constants and provide corresponding value of x&y
def solve():
    try:
        a1 = float(input("Enter a1 :"))
        b1 = float(input("Enter b1 :"))
        c1 = float(input("Enter c1 :"))
        a2 = float(input("Enter a2 :"))
        b2 = float(input("Enter b2 :"))
        c2 = float(input("Enter c2 :"))
        y = (c1*a2-c2*a1)/(b2*a1-b1*a2)
        x = (-c1 - b1*y)/a1
        print('x =',x, 'y =',y)
    except ZeroDivisionError:
        print("Solution is indefinite")


# ## Q.2 Loading and processing weather data

# In[ ]:


#First we will load the weather data file from its directory as a pandas dataframe.
data = pd.read_csv('../input/daily_weather.csv')
data


# In[ ]:


#Now all the unnecessary columns could be removed as follows
a = data.columns[2]
b = data.columns[7]
weather_data=data[[a,b]]
#For one month period top 30 entries is taken as follows
weather_data = weather_data.dropna()
weather_data = weather_data.head(30)
weather_data


# In[ ]:


#finally we can use either 'mean' or 'describe' function to find the average of both the 
#temperature and rain accumulation over the period of one month
weather_data.mean()


# As calculated above average data is as follows  
# Avg_temp = 64.30 degrees Fahranheit  
# Avg_rain = 0.31 millimeters  

# ### Plotting the results

# In[ ]:


# The below lines of code plot the variation of temp and rainfall accumulation over the 
# period of month.
plt.plot(weather_data['air_temp_9am'])
plt.plot(weather_data['rain_accumulation_9am'])


# As it can be observed there were rainfall (orange curve) only on a few days and the temperature (blue curve) corresponding to the maximum rain accumulation day is also very low and that also make sense.
