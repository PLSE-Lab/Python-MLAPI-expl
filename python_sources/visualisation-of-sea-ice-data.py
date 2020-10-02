#!/usr/bin/env python
# coding: utf-8

# #Introductory Data Visualisation of Sea-Ice Data
# 
# _Mathew Savage_

# The [National Snow and Ice Data Center][1] has provided Kaggle with a very extensive dataset of the extent of sea-ice data, collected every other day since 1978 and daily since around 1988, here is some exploratory data analysis and visualisation of that data, with the aim of making some predictions into the sea-ice levels in the future.
# 
# First we must start by importing useful python packages, and the data itself, and converting it to a more useful format, luckily the data is already mostly in a good form, so there is little required here;
# 
# [1]: https://nsidc.org

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing
import datetime as dt # date and time processing functions
import itertools
import matplotlib.pyplot as plt # basic plotting 
import matplotlib.dates as mdates # date processing in matplotlib
from matplotlib.offsetbox import AnchoredText
plt.style.use('ggplot') # use ggplot style
get_ipython().run_line_magic('matplotlib', 'inline')

# read in the data from the provided csv file
df = pd.read_csv('../input/seaice.csv')

# drop the 'Source Data' column as it obscures more useful columns and doesn't tell us much
df.drop('Source Data', axis = 1, inplace = True)

# convert the provided 3 column date format to datetime format and set it as the index
df['Date'] = pd.to_datetime(df[['Year','Month','Day']])
df.index = df['Date'].values

# split according to hemisphere, as we are expecting different trends for each
north = df[df['hemisphere'] == 'north']
south = df[df['hemisphere'] == 'south']


# Now that we have separated and cleaned data frames, we can plot all of the data to see what form it takes;

# In[ ]:


plt.figure(figsize=(9,3))
plt.plot(north.index,north['Extent'], label='Northern Hemisphere')
plt.plot(south.index,south['Extent'], label='Southern Hemisphere')

# add plot legend and titles
plt.legend(bbox_to_anchor=(0., -.362, 1., .102), loc=3, ncol=2, 
           mode="expand", borderaxespad=0.)
plt.ylabel('Sea ice extent (10^6 sq km)')
plt.xlabel('Date')
plt.title('Daily sea-ice extent');


# From this data we can see there there are independent maxima and minima for the northern and southern hemisphere data sets, as is to be expected from the change of season. In addition to this, there is on average more sea-ice in the southern hemisphere than the north.
# 
# The long-term trend looks as though there is a relatively constant amount of sea-ice year on year, but it would be more useful to plot the average mean, and see what the case actually is;

# In[ ]:


# resample raw data into annual averages
northyear = north['01-01-1979':'31-12-2016'].resample('12M').mean()
southyear = south['01-01-1979':'31-12-2016'].resample('12M').mean()

# remove the initial and final item as they aer averaged incorrectly (also indexes seem bad)
northyear = northyear[1:-1]
southyear = southyear[1:-1]

plt.figure(figsize=(9,3))
plt.plot(northyear.Year,northyear['Extent'], marker = '.', label='Northern Hemisphere')
plt.plot(southyear.Year,southyear['Extent'], marker = '.', label='Southern Hemisphere')

# add plot legend and titles
plt.legend(bbox_to_anchor=(0., -.362, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
plt.ylabel('Sea ice extent (10^6 sq km)')
plt.xlabel('Date')
plt.title('Annual average sea-ice extent')
plt.xlim(1977, 2016)


# Here, we can observe the general trend that there has been a steady decrease in the extent of sea-ice in the northern hemisphere, the extent of sea-ice in the southern hemisphere was relatively constant until the early 2000s, and since then has been increasing, at first this seemed to be a very strange result, [this has been known for some time][1] and is due to a large number of complicated factors.
# 
#   [1]: https://www.theguardian.com/environment/2014/oct/09/why-is-antarctic-sea-ice-at-record-levels-despite-global-warming

# One of the key trends observed in the rise of global warming is that seasonal phenomena seem to be shifting to earlier and earlier in the year, let's see if we can observe this in the melting and re-freezing of sea-ice in the polar regions;

# In[ ]:


# define date range to plot between
start = 1978
end = dt.datetime.now().year + 1

# define plot
f, axarr = plt.subplots(2, sharex=True, figsize=(9,6))


# organise plot axes (set x axis to months only and cycle colours according to gradient)
month_fmt = mdates.DateFormatter('%b')
axarr[0].xaxis.set_major_formatter(month_fmt)
axarr[0].set_prop_cycle(plt.cycler('color', 
                                   plt.cm.winter(np.linspace(0, 1, len(range(start, end))))))
axarr[1].set_prop_cycle(plt.cycler('color', 
                                   plt.cm.winter(np.linspace(0, 1, len(range(start, end))))))

# add plot legend and titles
axarr[0].set_ylabel('Sea ice extent (10^6 sq km)')
axarr[1].set_ylabel('Sea ice extent (10^6 sq km)')
axarr[1].set_xlabel('Month')
axarr[0].set_title('Annual change in sea-ice extent');
axarr[0].add_artist(AnchoredText('Northern Hemisphere', loc=3))
axarr[1].add_artist(AnchoredText('Southern Hemisphere', loc=2))

# loop for every year between the start year and current
for year in range(start, end):
    # create new dataframe for each year, 
    # and set the year to 1972 so all are plotted on the same axis
    nyeardf = north[['Extent', 'Day', 'Month']][north['Year'] == year]
    nyeardf['Year'] = 1972
    nyeardf['Date'] = pd.to_datetime(nyeardf[['Year','Month','Day']])
    nyeardf.index = nyeardf['Date'].values
    
    syeardf = south[['Extent', 'Day', 'Month']][south['Year'] == year]
    syeardf['Year'] = 1972
    syeardf['Date'] = pd.to_datetime(syeardf[['Year','Month','Day']])
    syeardf.index = syeardf['Date'].values
    
    # plot each year individually
    axarr[0].plot(nyeardf.index,nyeardf['Extent'], label = year)
    axarr[1].plot(syeardf.index,syeardf['Extent'])


# In this figure the older data are coloured green, shifting towards blue in the current data.
# 
# From this data we can clearly see that there is a shift in the positions of the maxima and minima of the sea ice extent in both the northern and southern hemispheres, indicating that the seasons have been gradually shifting over time. 
# 
# In the northern hemisphere data there is a marked difference between the fist and final year data, clearly highlighting the decrease in the extent of sea ice year on year. The southern hemisphere has been much more uniform over the same time period, indicating that the extent of sea-ice melting over this period has been much lower.

# In[ ]:




