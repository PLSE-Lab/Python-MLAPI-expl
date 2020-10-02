#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns

import math
import datetime as datetime

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
uk_housing_prices = pd.read_csv('../input/price_paid_records.csv')
# Any results you write to the current directory are saved as output.


# In[ ]:


uk_housing_prices.columns


# **Hypothesis**
# * Old properties will have less price then new one
# * Terrace and semi detached property will have more price compared to falt one.
# * City property prices will be more
# * Recent property price will be more compared to past.
# * County and District also have impact on property price

# In[ ]:


uk_housing_prices.shape


# In[ ]:


uk_housing_prices['County'].value_counts()


# These data file is huge, so we will examine only BERKSHIRE County. and check our hypothesis to start with our analysis.

# In[ ]:


berkshire_housing_prices = uk_housing_prices.loc[uk_housing_prices['County'] == 'BERKSHIRE']
berkshire_housing_prices.shape


# In[ ]:


berkshire_housing_prices.head(10)


# In[ ]:


berkshire_housing_prices.describe()


# In[ ]:


berkshire_housing_prices.isnull().sum()


# In[ ]:


berkshire_housing_prices['Price'].hist(bins = 50)


# Here we observe that there are few extreme values. This is also the reason why 50 bins are required to depict the distribution clearly.
# 
# Next, we look at box plots to understand the distributions. Box plot for fare can be plotted by:

# In[ ]:


berkshire_housing_prices.boxplot(column='Price')


# In[ ]:


berkshire_housing_prices.boxplot(column='Price', by = 'Old/New')


# We can see there is no substantial difference between the mean Price for 'Old' or 'New' property. 
# however there are
# a higher number of Prices for New property type. which are appearing to be the outliers.

# In[ ]:


berkshire_housing_prices.boxplot(column = 'Price', by = 'Property Type')


# There are a higher number of Prices for 'Detached' Property type. also there are visible outliers for Detached Property type.

# In[ ]:


berkshire_housing_prices.boxplot(column = 'Price', by ='Town/City', figsize = (16,8))


# In[ ]:


berkshire_housing_prices.boxplot(column = 'Price', by = 'Duration')


# In[ ]:


temp1 = pd.crosstab(berkshire_housing_prices['Town/City'], berkshire_housing_prices['Property Type'])
temp1.plot(kind = 'bar', stacked = True, color =['red','blue','green','orange'], grid = False)


# In[ ]:


temp2 = pd.crosstab(berkshire_housing_prices['Duration'], berkshire_housing_prices['Old/New'])
temp2.plot(kind='bar', stacked = True, color =['red','blue'], grid = False)


# In[ ]:


berkshire_housing_prices['Price_log'] = np.log(berkshire_housing_prices['Price'])
berkshire_housing_prices['Price_log'].hist(bins = 50)


# In[ ]:


berkshire_housing_prices['Date of Transfer'] = pd.to_datetime(berkshire_housing_prices['Date of Transfer'], format='%Y-%m-%d %H:%M')


# In[ ]:


berkshire_housing_prices.index = berkshire_housing_prices['Date of Transfer']
df = berkshire_housing_prices.loc[:,['Price']]
ts = df['Price']
plt.figure(figsize=(16,8))
plt.plot(ts, label='House Price')
plt.title('Time series')
plt.xlabel('Time(year-month)')
plt.ylabel('House Price')
plt.legend(loc = 'best')
plt.show()


# Looks like there is no data entry for berkshire house prices after 1998, 
# we can see from tail data, there are hardly 3 entries.

# In[ ]:


uk_housing_prices['Date of Transfer'] = pd.to_datetime(uk_housing_prices['Date of Transfer'], format='%Y-%m-%d %H:%M')


# In[ ]:


uk_housing_prices.dtypes


# Time series plot for huge data shows overflow error so lets try with their mean value to check our hypothesis is right or not?

# In[ ]:


uk_housing_prices.index = uk_housing_prices['Date of Transfer']
uk_housing_prices.index


# In[ ]:


uk_housing_prices['year'] = pd.DatetimeIndex(uk_housing_prices['Date of Transfer']).year
uk_housing_prices['month'] = pd.DatetimeIndex(uk_housing_prices['Date of Transfer']).month


# In[ ]:


uk_housing_prices.head(5)


# In[ ]:


uk_housing_prices.groupby('year')['Price'].mean().plot.bar()


# we can clearly see that housing price is significantly increasing every year from the above graph.

# In[ ]:


county_data = uk_housing_prices.groupby('County')['Price'].mean()
county_data.plot(figsize = (20,8), title = 'UK County House Prices')


# In[ ]:


temp = uk_housing_prices.groupby(['year','month'])['Price'].mean()
temp.plot(figsize=(16,5), title = 'UK Housing Price(Monthwise)', fontsize = 12)


# In[ ]:


temp3 = pd.crosstab(uk_housing_prices['Old/New'], uk_housing_prices['Property Type'])
temp3.plot(kind='bar', stacked=True, grid=False, figsize=(18,8))


# In[ ]:


temp = uk_housing_prices.groupby(['Old/New','Property Type'])['Price'].mean()
temp.plot(figsize=(16,5), title = 'UK Housing Price(Monthwise)', fontsize = 12)


# From berkshire_uousing_prices on time series, we can see there are many data entries are missing after 1998.
# infact hardly 3-4 data entries after 1998. Thus it will be a huge gap if similarly data entries are missing for other countys.
# 1) First of all we must check is this the case with other counties as well?
# 2) Lets check london housing prices as well manchester housing prices on time series in this case and get more accurate
# idea about missing data for other counties probably can have.

# In[ ]:


london_housing_prices = uk_housing_prices.loc[uk_housing_prices['County'] == 'GREATER LONDON']


# In[ ]:


london_housing_prices.shape


# In[ ]:


london_housing_prices.dtypes


# In[ ]:


london_housing_prices.index = london_housing_prices['Date of Transfer']
df1 = london_housing_prices.loc[:,['Price']]
ts1 = df1['Price']
plt.figure(figsize=(16,8))
plt.plot(ts, label='London House Price')
plt.title('Time series')
plt.xlabel('Time(year-month)')
plt.ylabel('London House Price')
plt.legend(loc = 'best')
plt.show()


# In[ ]:


machester_housing_prices = uk_housing_prices.loc[uk_housing_prices['County'] == 'GREATER MANCHESTER']


# In[ ]:


machester_housing_prices.shape


# In[ ]:


machester_housing_prices.index = machester_housing_prices['Date of Transfer']
df2 = machester_housing_prices.loc[:,['Price']]
ts2 = df2['Price']
plt.figure(figsize=(16,8))
plt.plot(ts, label='Machester House Price')
plt.title('Time series')
plt.xlabel('Time(year-month)')
plt.ylabel('Machester House Price')
plt.legend(loc = 'best')
plt.show()


# From above 3 cases 1) either we make a model based on 1995 to 1998 data or 2) we have to check first every year how many entries to deal with missing data**** which we will cover in part2

# In[ ]:




