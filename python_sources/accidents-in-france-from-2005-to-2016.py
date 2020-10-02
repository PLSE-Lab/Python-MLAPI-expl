#!/usr/bin/env python
# coding: utf-8

# # Table of Contents
# 1. Making imports
# 2. Explore the main (caracteristics.csv) dataset
# 3. Find missing values
# 4. Time based analysis
# 5. Spatial analysis [TODO]
# 6. Road condition analysis [TODO]
# 7. Mixed analysis [TODO]
# 8. Try to determine the main features [TODO]
# 9. Determine the most dangerous and the least dangerous conditions about [TODO]
# (10. Build model?!) [TODO] 

# # Making imports
# I import the necessary modules and list the input directory. After that I load the first dataset with pandas.

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime  

import statsmodels.api as sm  
import missingno as msno

import matplotlib.pyplot as plt
plt.style.use('classic')
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from collections import OrderedDict
sns.set()
import bokeh

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # Explore caracteristics dataset
# I do some conversions in time attributes, create pandas datetime to processing time series data easier. After the conversions I exlore the missing values in the dataset. 

# In[45]:


caracteristics = pd.read_csv('../input/caracteristics.csv', encoding = 'latin-1')
caracteristics.describe()


# In[46]:


caracteristics["hour"] = (caracteristics.hrmn - caracteristics.hrmn%100)/100
caracteristics["min"] = caracteristics.hrmn - caracteristics.hour*100;
caracteristics.an = caracteristics.an + 2000
caracteristics['date']= caracteristics.apply(lambda row :
                          datetime.date(row.an,row.mois,row.jour), 
                          axis=1)
caracteristics['date'] = pd.to_datetime(caracteristics['date'])
caracteristics['day_of_week'] = caracteristics['date'].dt.dayofweek
caracteristics['day_of_year'] = caracteristics['date'].dt.dayofyear
caracteristics.head()


# ## Now let's explore the missing values easy with visualization

# In[4]:


msno.matrix(caracteristics)


# ### What can we see here? 
# We have full coverage instead of location features. In location features we have got better coverage in adr than in gps and usually where we have missing adr then we have gps location and vice versa. So this doesn't mean a huge problem, beacuse we can convert adr values to gps and have a great coverage in location features too.
# **So now we can make time based analysis and we don't need to clean or modify further our dataset.**

# # Time based analysis
# Do the followig analysis:
# * Number of accidents in years
# * Number of accidents in months
# * Number of accidents in weekdays
# * Number of accidents in hours

# In[47]:


from pylab import rcParams
rcParams['figure.figsize'] = 8, 8

ax = sns.distplot(caracteristics.an,bins=12,kde=False);
plt.title('Number of accidents in years')
plt.xlabel('years')
plt.ylabel('Number of accidents')


# In[49]:


from pylab import rcParams
rcParams['figure.figsize'] = 8, 8

ax = sns.distplot(caracteristics.day_of_year,bins=365,kde=False);
plt.title('Number of accidents in days')
plt.xlabel('days in the year')
plt.ylabel('Number of accidents')


# In[6]:


from pylab import rcParams
rcParams['figure.figsize'] = 8, 8

ax = sns.distplot(caracteristics.mois,bins=12,kde=False);
plt.title('Number of accidents in months')
plt.xlabel('months')
plt.ylabel('Number of accidents')


# In[7]:


from pylab import rcParams
rcParams['figure.figsize'] = 8, 8

ax = sns.distplot(caracteristics.day_of_week,bins=7,kde=False);
plt.title('Number of accidents in weekdays')
plt.xlabel('weekdays')
plt.ylabel('Number of accidents')


# In[8]:


from pylab import rcParams
rcParams['figure.figsize'] = 8, 8

ax = sns.distplot(caracteristics.hour,bins=24,kde=False);
plt.title('Number of accidents in hours')
plt.xlabel('hours')
plt.ylabel('Number of accidents')


# In[9]:


import matplotlib.dates as mdates
from pylab import rcParams
rcParams['figure.figsize'] = 8, 8

caracteristics.index = caracteristics['date'] 
day_resample = caracteristics.resample('D').count()

day_resample.head()

sns.tsplot(data=day_resample.Num_Acc, time=day_resample.index)
ax = plt.gca()
# get current xtick labels
xticks = ax.get_xticks()
# convert all xtick labels to selected format from ms timestamp
ax.set_xticklabels([pd.to_datetime(tm).strftime('%Y-%m-%d\n') for tm in xticks],rotation=50)

plt.title('Number of accidents in a day')
plt.xlabel('date')
plt.ylabel('Number of accidents')
plt.show()


# ## Evalution of the timeseries plot
# * We can see a slightly non-stationary in timeseries
# * And we see also some outlier with short cycle
# 
# Now I do some time based analysis from another viewpont. I try to separate trend and seasonality from each other. 
# First of all I need to test stacionary of the timeseries and if it isn't stacionary then tranform it for later steps. I use Dickey-Fuller test for stacionary analysis.

# In[15]:


from statsmodels.tsa.stattools import adfuller

def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = timeseries.rolling(12).mean()
    rolstd = timeseries.rolling(12).std()

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Standard deviation')
    plt.legend(loc='best')
    plt.title('Rolling Mean and Standard deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
    
test_stationarity(day_resample.Num_Acc)


# ## Result of Dickey-Fuller test:
# Okay so now we can say the full timeseries of accidents is (almost) stationary (test statistic is smaller than critical value of 1%) with 99% confidence level. 
# 
# ## Separate trend and seasonality with one year period
# 

# In[44]:


full_df_ts = day_resample.Num_Acc
full_df_ts.index = day_resample.index
full_df_ts.loc[full_df_ts.index >= '2006']
res = sm.tsa.seasonal_decompose(full_df_ts, freq=365, model='additive')

plt.rcParams["figure.figsize"] = (20,10)
fig, (ax1,ax2,ax3) = plt.subplots(3,1, figsize=(20,10))
res.trend.plot(ax=ax1)
res.resid.plot(ax=ax3)
res.seasonal.plot(ax=ax2)


# # **Conclusion of the time based analysis**: 
# * The number of accidents have decreased between 2005 and 2013 with about ~30%. 
# * There are two local minimum in 'number of accidents in months' plot. The second minimum probably was caused by summer holiday period. Then lot of people go to holiday and use their car less often. (The first minimum could also caused by hoilday period. )
# * Moreover we can see one other absolute minimum at the end of the year, but its length is shorter than others. It caused that we see it only on "Number of accidents in days" and not on "Number of accidents in months".
# * We can see two local maximum in 'number of accidents in hours' plot. First local maximum is in the morning when people go to work and second maximum is in the afternoun when people go to home from workplace. 

# ## In future create a model based on only time...
