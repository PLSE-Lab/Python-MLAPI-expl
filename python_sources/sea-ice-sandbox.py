#!/usr/bin/env python
# coding: utf-8

# **My First Kaggle Notebook - woo!**
# 
# I'm just getting started with Kaggle and this is my first-ish notebook. (I forked at least one or two others, but this is my first one all on my own). I figured I'd start with sea ice data - it seems like a pretty straightforward dataset. Let's see what's in it!

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates 
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('fivethirtyeight')


# In[ ]:


# Now I'll just import the dataset using pandas
seaice = pd.read_csv('../input/seaice.csv')
# Get some Basic stats
seaice['Extent'].describe()
# Convert the Year, Month, Day columns to date format
seaice['Date'] = pd.to_datetime(seaice[['Year','Month','Day']])


# I'd like to plot average annual ice extent changes over years. To do this, I'll need to compute the average ice extent for each year, since each year has many data points on different dates.

# In[ ]:


# Group the dataset by year
iceyears = seaice.groupby('Year')
# Pull out mean of the 'Extent' variable
annualextent = iceyears.mean()['Extent']
# plot the average annual extent
plt.plot(annualextent)
plt.title('Average sea ice extent')
plt.ylabel(r'Area (*10$^6$ km$^2$)')
plt.xlabel('Year')


# The plot above is not terribly informative. Taking the mean of each year doesn't make sense since we know that there is a strong seasonal component to ice extent. Additionally, we're averaging the northern and southern hemispheres, which will result in a loss of most of the relevant signals. 

# In[ ]:


# Plot hemispheres separately
# Northern hemisphere df
northice = seaice[seaice['hemisphere']=='north']
# Southern hemisphere df
southice = seaice[seaice['hemisphere']=='south']

# Plot the annual maximum and minimum sea ice for each hemisphere
northice_years = northice.groupby('Year')
southice_years = southice.groupby('Year')

# Set up axes
f, axarr = plt.subplots(2, sharex=True, figsize=(7.5,6))
northmax = northice_years['Extent'].max()
northmin = northice_years['Extent'].min()
axarr[0].plot(northmax)
axarr[0].plot(northmin)
axarr[0].set_title('Northern hemisphere')

southmax = southice_years['Extent'].max()
southmin = southice_years['Extent'].min()
axarr[1].plot(southmax)
axarr[1].plot(southmin)
axarr[0].set_title('Southern hemisphere')


# In[ ]:




