#!/usr/bin/env python
# coding: utf-8

# Breaking news: Russia is cold... sometimes really cold. 
# 
# Finding something interesting in a jumble of time series.
# 
# Often, directly plotting the data doesn't yield information that is easy to digest. AverageTemperatures is a case in point. Here, I use rolling deviations to reveal something of historical interest.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import statsmodels.api as sm
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
source = pd.read_csv('../input/GlobalLandTemperaturesByMajorCity.csv')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = source.copy()  # I only do this to avoid repeated readings of CSV file.
df.loc[:, 'dt'] = pd.to_datetime(df.loc[:, 'dt'])  # Req'd for decomposition.
df = source.set_index(df.loc[:, 'dt'])  # note: makes multiple rows with same index.

cities = list(source['City'].unique())
df.drop(['Latitude', 'Longitude'], axis=1, inplace=True)


# Since we're looking at time series data that has an obvious seasonal component, we're using statsmodels to remove it. This way, we can examine the trend.
# 
# With the intention of keeping this notebook short, we won't investigate residuals although let's note that we should.

# In[ ]:


dfd= {}
df_trend = pd.DataFrame()
basedt = '1882-01-01'  # somewhat arbitrary starting point.

for city in cities:
    dfd[city] = df[df['City']==city]
    dfd[city].interpolate(inplace=True)  # decompose requires data to be nan-free. Linearly interpolating.
    df_trend[city] = sm.tsa.seasonal_decompose(dfd[city]['AverageTemperature']).trend


# In[ ]:


df_trend.dropna(inplace=True)
rolling = df_trend.rolling(window=36, center=False).std()  # pd.rolling_std() is deprecated.
rolling = rolling.dropna()


# It is difficult to identify something interesting in AverageTemperature of cities by visual inspection. (See the first of the four visualizations below).
# Removing the seasonal component of each series yields only slightly more interpretable results. (See the second of the four visualizations below).
# One way to identify which of the trends are most interesting is to calculate rolling standard deviations and pick the n with the greatest maximum. For comparison, we pick the n with the smallest maximum too. (See the third of the four visualizations below).

# In[ ]:


n = 2
largest = rolling.max().nlargest(n)
smallest = rolling.max().nsmallest(n)


# In[ ]:


pt = pd.pivot_table(df,
                    values='AverageTemperature',
                    index='dt',
                    columns='City',
                    aggfunc=sum,
                    dropna=False)
pt = pt.set_index(pd.to_datetime(pt.index))


# In[ ]:


fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True, figsize=(20, 14))

pt.loc[basedt:].plot(ax=ax1,
                     legend=False,
                     title='AverageTemperature (difficult to learn much)')
df_trend.plot(ax=ax2,
              legend=False,
              title='Trend component of AverageTemperature (still difficult to learn much)')
rolling[largest.index | smallest.index].plot(ax=ax3,
              title='Rolling Deviations of Most and Least Volatile (we can learn something)')
df_trend.loc[basedt:, largest.index | smallest.index].plot(ax=ax4,
              title='AverageTemperature of Most and Least Volatile (we can learn something)')
plt.show()


# A quick google search returned this http://onlinelibrary.wiley.com/doi/10.1256/wea.248.04/pdf. According to this, 1940-1942 was a period of "global climate anomoly."
# 
# Not surprisingly, the pair of countries with the lowest maximum in its rolling deviations (trend) are pretty stable. It seems as though nothing interesting is happening there.

# Taken from http://www.geipelnet.com/war_albums/otto/ow_12.html
# 
# 1942
# The winter comes with full strength, hardly a way left to advance without missing winter equipment. Even the winter clothing is missing.
# ![title](http://www.geipelnet.com/war_albums/otto/jpg/OW12_1.JPG)
# 
# The Panje Horse
# The only reliable means of movement is this time.
# At midnight the temperature dropped to a new reported low point - On January 24, 1942, -56c degrees was measured at our division observation post.
# 
# ![title](http://www.geipelnet.com/war_albums/otto/jpg/OW12_2.JPG)
# 
