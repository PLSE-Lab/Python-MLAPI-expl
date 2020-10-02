#!/usr/bin/env python
# coding: utf-8

# Here we study Weather Data from Beutenberg Campus situated in Southern Jena, Germany. 
# I downloaded 2 Years worth of weather data from Jan 2017 - Dec 2018 sampled every 10 minutes. 
# You can download the data from here (https://github.com/HimanshuGautam17/Personal-Data-Repo). 
# 
# Let us start with some exploratory analysis to understand the Data. 

# In[ ]:


#Importing required packages
from pathlib import Path
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.cluster import KMeans


# In[ ]:


weather = pd.read_csv('/kaggle/input/Beutenberg.csv', encoding='latin', parse_dates=['Date.Time'])
print("Rows:", weather.shape)
weather.head()


# In[ ]:


#Get Date & Hour from Date.Time Column
weather['Year'] = weather['Date.Time'].dt.year
weather['Month'] = weather['Date.Time'].dt.month
weather['Date'] = weather['Date.Time'].dt.date
weather['Hour'] = weather['Date.Time'].dt.hour
weather = weather[weather['Year'] <=2018]
print(weather.columns)
weather.head()


# In[ ]:


#Compute Hourly Avg Temp on monthly basis
w_temp = weather.pivot_table(index=['Year', 'Month'], columns=['Hour'], values=['T..degC.']) 
print(w_temp.shape)
w_temp.head()


# In[ ]:


#Lets draw a heatmap to visualize Seasons
import seaborn as sb
plt.figure(figsize=(20,10))
heat_map = sb.heatmap(w_temp, cmap=sb.color_palette("RdBu_r", 15), annot=True)


# This one chart clearly depicts the season cycles for 2 Years. Also it is easily seen that Feb 2018 was the coldest month and July 2018 was the hottest month. Also Temperature varies between 11 degree to ~20 degrees during summer months and stays between ~0 degree to 9 degrees during winter months. 
# 
# Let us see some other variable in a similar plot to know more:

# In[ ]:


#Compute Hourly Sum Rains on monthly basis
w_rain = weather.pivot_table(index=['Year', 'Month'], columns=['Hour'], values=['rain..mm.'], aggfunc=np.sum)
plt.figure(figsize=(20,10))
heat_map = sb.heatmap(w_rain, cmap=sb.color_palette("Blues", 15), annot=True)
#TotalRain = list(w_rain.sum(axis=1))


# Heat map clearly shows little rains on most months. However, it rains often between 3PM to 5PM. Lately heavy rains happened in July & Sep of 2018. 
# 
# Let us now perform the KMeans Clustering in the Weather Data

# In[ ]:


#Compute Hourly avg Wind speed on monthly basis
w_wind = weather.pivot_table(index=['Year', 'Month'], columns=['Hour'], values=['wv..m.s.'], aggfunc=np.mean)
plt.figure(figsize=(20,10))
heat_map = sb.heatmap(w_wind, cmap=sb.color_palette("Blues", 15), annot=True)


# There are particularly high winds seen during general working hours (9AM - 5PM). This looks quite weird and probably some external factor is impacting this data point. 
# 
# Let us also check CO2 levels with the familiar heatmap:

# In[ ]:


#Compute Hourly avg CO2 on monthly basis
w_co2 = weather.pivot_table(index=['Year', 'Month'], columns=['Hour'], values=['CO2..ppm.'], aggfunc=np.mean)
plt.figure(figsize=(20,10))
heat_map = sb.heatmap(w_co2, cmap=sb.color_palette("Blues", 15))


# As per standards CO2 levels upto 350 ppm are fine. This suggests the region is in compliance with pollution standards. Interestingly some very low values ~0 (probably due to sensor issues) puts the heat map in a dark perception. Also a very interesting phenomenon is seen, where CO2 concentration is much higher in Night time as compared to Day time. With my limited understanding of the environment science i would attribute this difference to the photosynthesis process in plants during day time. 
# 
# Please share your feedbacks/suggestions. Hope you enjoyed it. 
# 
# References:
# 1.Beutenerg Campus: https://en.wikipedia.org/wiki/Beutenberg_Campus
# 2.CO2 Level Norms: https://www.kane.co.uk/knowledge-centre/what-are-safe-levels-of-co-and-co2-in-rooms
