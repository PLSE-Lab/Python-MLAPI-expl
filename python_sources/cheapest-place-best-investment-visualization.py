#!/usr/bin/env python
# coding: utf-8

# ### 11-8-2016
# Just trying out some ideas to visualize the data. Two things I have focused thus far: 1) the average listing price over the years for each region, and 2) pricing trend for each region.
# 
# *Will update this script with more graphs in the future and write a more proper introduction.*

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

get_ipython().run_line_magic('matplotlib', 'inline')

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Read the data
df = pd.read_csv('../input/median_price.csv')
df.shape # 67 Regions, 87 Variables


# In[ ]:


# Get the median listing price columns 
prices = df.columns[6:]


# In[ ]:


# Calculate the average listing price and the variance over the time period
df['average_price'] = np.nanmean(df[prices], axis = 1)
df['price_variance'] = np.nanvar(df[prices], axis = 1)
df['size'] = df.shape[0] - df['SizeRank'] # Reverse code the size rank so larger number = larger size


# # Take a quick look at the data
# ### 9 out of the 10 regions with highest listing price over the years are from New York. Anyone surprised?

# In[ ]:


df[['RegionName', 'City', 'average_price']].nlargest(10, 'average_price')


# ### Let's look at the other end of the spectrum, lowest listing price regions over the time period.

# In[ ]:


df[['RegionName', 'City', 'average_price']].nsmallest(10, 'average_price')


# ### Let's take a look at how the average price over the time period is related to size of the region and volatility of the price.

# In[ ]:


colors = np.random.rand(df.shape[0]) #Color code for each region
# Use size of the point to indicate price fluctuations over the years (Larger = more fluctuation)
sizes = df['price_variance'] / 10 #Scale the variance by a factor of 10 so the sizes are more managable

plt.style.use('fivethirtyeight') #Use the fivethirtyeight style

# Scatter plot of the data
plt.scatter(df['size'], df['average_price'], s = sizes, c = colors, alpha = 0.7)
plt.ylim([0, 2500])
plt.xlim([-10, 85])
plt.ylabel('Median Listing Price')
plt.xlabel('Size')
labels = df['RegionName']
plt.text(-10, 2500, 'Size of the dot represents variance in price for the region',
         fontsize = 10,
         color = 'red')
# Find the regions with top 5 variance in price over the time period and label the them in the plot
top5variance = df[['RegionName', 'size', 'average_price', 'price_variance']].nlargest(5, 'price_variance')
for r in top5variance.itertuples(index = False):
    plt.annotate(r[0], xy = (r[1], r[2]),
                 size = 10,
                 xycoords = 'data',
                 xytext = (r[1]+10, r[2] + 20),
                 arrowprops = dict(arrowstyle = '->', color = 'black'))
plt.show()


# As we can see above, prices of the largest regions tend to fluctuate more over the time period. List price is fairly stable for the smllaer region. I am going to actually explore the price trend below.
# 
# # Plotting the price trend for the regions with top 5 price variance

# In[ ]:


top10variances = df.nlargest(5, 'price_variance')

yearsFmt = mdates.DateFormatter('%Y')
for i in range(5):
    plt.plot_date(prices, top10variances.iloc[i][prices], 'o-')
plt.ylim([500, 2300])
plt.ylabel('Median Listing Price')
ax = plt.axes()
ax.xaxis.set_major_formatter(yearsFmt) # Only plot years at x axis
plt.legend(top10variances['RegionName'], loc = 2, prop={'size':10})


# We can see the regions with most price fluctuations all have an upward trend over the years. The exception   is Flatiron District which experienced a price dip in early 2015. The changes does not seem very impressive until you see the graph below.
# 
# # Plotting the price trend for the regions with bottom 5 price variance

# In[ ]:


lowestvariances = df.nsmallest(5, 'price_variance')
for i in range(5):
    plt.plot_date(prices, lowestvariances.iloc[i][prices], 'o-')
plt.ylabel('Median Listing Price')
ax = plt.axes()
ax.xaxis.set_major_formatter(yearsFmt)
plt.legend(lowestvariances['RegionName'], loc = 2, prop={'size':10})


# The price trend is essentially flat for all these 5 regions.
