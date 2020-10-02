#!/usr/bin/env python
# coding: utf-8

# # 0. Introduction
# 
# This note gives you a brief introduction of how to deal with time series. I try to write in a succinct and informative language. The material should not be too hard if you decided to click on this notebook (and you did). Have fun and feel free to comment. 

# # I. Timestamp manipulation

# We are going to use two data sets:
# 
# The **min_temp data** has the minimum temperature of every day for 10 years. (1981-1990)
# 
# The **ts data** set has 72 random numbers with an increasing trend and some noises.

# In[ ]:


# min_temp data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image
import os
get_ipython().run_line_magic('matplotlib', 'inline')

temp_data = pd.read_csv("../input/min-temp/min_temp.csv")
temp_data.head().append(temp_data.tail())


# In[ ]:


#ts data
years = pd.date_range('2012-01', periods=72, freq="M")
index = pd.DatetimeIndex(years)

np.random.seed(3456)
sales= np.random.randint(-4, high=4, size=72)
bigger = np.array([0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,3,3,3,3,
                   3,3,3,3,3,3,3,3,7,7,7,7,7,7,7,7,7,7,7,
                   11,11,11,11,11,11,11,11,11,11,18,18,18,
                   18,18,18,18,18,18,26,26,26,26,26,36,36,36,36,36])
data = pd.Series(sales+bigger+6, index=index)
ts=data
ts


# ### Changing the index and resampling the data

# In[ ]:


#Replace the index by Date column
temp_data.Date = pd.to_datetime(temp_data.Date, format='%d/%m/%y') #convert to pandas timestamp type
temp_data.set_index('Date', inplace=True) 

temp_data.head(2)


# In[ ]:


#Group by month and find the average of each month

temp_monthly = temp_data.resample('MS') #Since it reduces the data, this called down sampling
month_mean = temp_monthly.mean()
month_mean.head(5)


# In[ ]:


#Adding addition row for each day and fill it with previous value
temp_bidaily= temp_data.resample('12H').asfreq()#Since it increase the data, this called up sampling

print(temp_bidaily.isnull().sum()) #There are some null here

#Fill data behind it with the following one
temp_bidaily_fill= temp_data.resample('12H').ffill() #forward filling, backward filling (bfill())
temp_bidaily_fill.head()


# In[ ]:


#Selecting and slicing time series data
#Retrieve data after 1985
temp_1985_onwards = temp_data['1985':]
temp_1985_onwards.head(2).append(temp_1985_onwards.tail(2))


# ### Plot

# In[ ]:


temp_data.plot()
plt.show()


# In[ ]:


#Dot plots can prove to be very helpful in identifying outliers and very small patterns 
#which may not be so obvious otherwise

temp_data.plot(style=".b")
plt.show()


# [More trick](https://towardsdatascience.com/basic-time-series-manipulation-with-pandas-4432afee64ea)

# # II. Time series trend
# ### Concept
# A given time series is thought to consist of three systematic components including level, trend, seasonality, and one non-systematic component called noise:
# 
# >**Seasonal**: Patterns that repeat within a fixed period. For example, a website might receive more visits during weekends; this would produce data with the seasonality of 7 days.
# 
# >**Trend**: The underlying trend of the metrics. A website increasing in popularity should show a general trend that goes up.
# 
# >**Level**: The average value in the series.
# 
# >**Noise/Residual**: The random variation in the series. This is what left after removing seasonal and trend
# 
# There are two types of time series:
# 
# * **Stationary**: the oscillation does not go up or down. Same mean over time (**homoscedasticity**)
# 
# * **Non-stationary**: Having some tendency (trend) of going up or down. Different means over time. The tendency it non-stationary time series may have are: linear, exponential, periodic, oscillation get bigger over time

# In[ ]:


Image("../input/pictures/stationary.png")


# ### Trend or not trend 
# It is important to know the trend because the model often works better with non-trend time series. (Just like how model work better with normal distribution for different types of data)
# 
# **1) Rolling statistics**
# 
# For any specific time $t$ we can use a window of length $m$ to capture the values of the time series right before $t$. After getting these values from the window, we can compute the average or variance, depends on the purpose, and estimate the value at $t$. 
# 
# The graph of this method is much smoother than the original graph. This will give us a better indication of the trend of the graph. 
# 
# Pandas has a built-in function called [rolling()](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.rolling.html), which can be called along with `.mean()` and `.std()` to calculate these rolling statistics. Let's take a window size of 8 for this example. 
# 
# (Test different window sizes and see how it affect the rolling graph) 

# In[ ]:


rolmean = ts.rolling(window = 10, center = False).mean()
rolstd = ts.rolling(window = 10, center = False).std()
#Note that it lost a little bit in the beginning since the window use the previous info to check the future

fig = plt.figure(figsize=(12,7))
orig = plt.plot(ts, color='blue',label='Original')
mean = plt.plot(rolmean, color='red', label='Rolling Mean')
std = plt.plot(rolstd, color='black', label = 'Rolling Std')
plt.legend(loc='best')
plt.title('Rolling Mean & Standard Deviation')
plt.show(block=False)


# The red and black lines represent the rolling mean and rolling standard deviations. You can see that the mean is not constant over time, so we can reconfirm our conclusion that the time series is not stationary based on rolling mean and rolling standard error.

# #### 2) The weighted rolling mean
# 
# A drawback of the rolling mean approach is that the window has to be strictly defined. In this case, we can take yearly averages but in complex situations like forecasting a stock price, it may be difficult to come up with an exact number. So we take a "weighted rolling mean" (or weighted moving average, WMA for short) where **more recent values are given a higher weight**. There are several techniques for assigning weights. A popular one is **Exponentially Weighted Moving Average** where weights are assigned to all the previous values with an exponential decay factor. This can be implemented in Pandas with `DataFrame.ewm()` method. Details can be found [here](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.ewm.html).

# In[ ]:


# Use Pandas ewma() to calculate Weighted Moving Average of ts_log
exp_rolmean = data.ewm(halflife = 3).mean() #Here, 3 is 3 month period. Halflife means the decativity rate
exp_rolstd = data.ewm(halflife = 3).std()
# Plot the original data with exp weighted average
fig = plt.figure(figsize=(12,7))
plt.plot(data, color='blue',label='Original')
plt.plot(exp_rolmean, color='red', label='Exponentially Weighted Rolling Mean')
plt.plot(exp_rolstd, color='black', label='Exponentially Weighted Rolling STD')
plt.legend()
plt.title('Exponentially Weighted Rolling Mean & Standard Deviation')
plt.show()


# **3) The Dickey-Fuller Test** 
# 
# The Dickey-Fuller Test is a statistical test for testing stationarity. The Null-hypothesis for the test is that the time series is not stationary. So if the test statistic is less than the critical value, we reject the null
# hypothesis and say that the series is stationary. The Dickey-Fuller test is available in stat tools from the StatsModels module. More details on this can be viewed [here](http://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.adfuller.html).

# In[ ]:


from statsmodels.tsa.stattools import adfuller

dftest = adfuller(ts)

dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

print ('Results of Dickey-Fuller Test:')
print(dfoutput)


# The null-hypothesis here is there is no trend. Since p-value = 1,  we reject the null-hypothesis.

# ### Eliminating the trend
# 
# Recap:
# 
# The reason to eliminate the trend is that the stationarity assumption is required in major time series modeling techniques but few practical time series are stationary.
# 
# 
# >**Trend**: Varying means over time. 
# 
# >**Seasonality**: Certain variations at specific time-frames.
# 
# The underlying principle is to model or estimate the trend and seasonality in the series and remove those from the series to get a stationary series. Statistical modeling techniques can then be implemented in these series. The final step would be to convert the modeled values into the original scale by applying trend and seasonality constraints back.
# 
# There are 3 important keys to eliminate trends:
# - Taking the log transformation (Alternatively, square root, cube root,...)
# - Subtracting the rolling mean
# - Differencing

# **1) Log transformation vs Subtracting the rolling mean**
# 
# By apply the log transformation, we reduce the volatile of the graph. 
# 
# By subtracting the rolling mean, we straighten the trend line and move it toward the horizontal line. 
# 
# In this example, we will compare the original graph with log transformation, subtracting the rolling mean, and subtracting the weighted rolling mean.

# In[ ]:


fig, axs = plt.subplots(4,sharex=True,figsize=(11,7),gridspec_kw={'hspace': 0})

#Original data
axs[0].plot(ts, color='blue')

#Log transform
axs[1].plot(np.log(ts),color='red')

#Subtracting the rolling mean
rolmean = ts.rolling(window = 4).mean()
data_minus_rolmean1 = ts - rolmean #How we define "Subtracting the rolling mean"
axs[2].plot(data_minus_rolmean1,color='green')

#Subtracting the weighted rolling mean
exp_rolmean = data.ewm(halflife = 2).mean()
data_minus_rolmean2 = ts - exp_rolmean
axs[3].plot(data_minus_rolmean2,color='purple')


# Observe that log transformation bend the trend line just a little bit. The subtracting weight method seems to work very effectively. Also, notice that the purple graph did not lose the tip like the green graph thanks to the math behind weight rolling mean!

# ### Differencing
# 
# One of the most common methods of dealing with both trend and seasonality is differencing. In this technique, we take the difference of observation at a particular time instant with that at the previous instant (i.e. a co-called 1-period "lag"). 
# 
# This mostly works pretty well in improving stationarity. First-order differencing can be done in Pandas using the `.diff()` function with periods = 1 (denoting a 1-period lag). Details on `.diff()` can be found [here](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.diff.html).

# In[ ]:


data_diff = data.diff(periods=1)
data_diff.head(10)

fig = plt.figure(figsize=(11,3))
plt.plot(data_diff, color='blue',label='Sales - rolling mean')
plt.legend(loc='best')
plt.title('Differenced sales series')
plt.show()


# This seems to work pretty well if you want to make the series stationary!
# 
# Differencing is a very popular tool to remove seasonal trends from time series as well. Look at the plot below. Here, we differenced our temperature data by taking differences of exactly one year, which removes the cyclical seasonality from the time series data! Pretty magical!

# In[ ]:


fig, axs = plt.subplots(2,sharex=True,figsize=(11,7),gridspec_kw={'hspace': 0})

#Original data temp data
axs[0].plot(temp_data, color='blue', linewidth=1)

#1-period lag
data_diff = temp_data.diff(periods=365)
axs[1].plot(data_diff, color='red', linewidth=1)


# # III. Naive Decomposition

# A time series has 4 components: level, trend, seasonality, and noise where
# 
# Level: The average value in the series.
# Trend: The increasing or decreasing value in the series.
# Seasonality: The repeating short-term cycle in the series. For example, a website might receive more visits during weekends; this would produce data with seasonality of 7 days.
# Noise: The random variation in the series.
# 
# The Naive decomposition will help us decompose these components. Note that because it is naive, it can only deal with simple time series. The temperature data set is too complicated for naive

# Before moving to the code lets break down two types of time series problems: 
# 
# > 1) Additive problems:
# $$y(t) = \text{ Level } + \text{ Trend } + \text{ Seasonality } + \text{ Noise }$$
# Example: For monthly data, an additive model assumes that the difference between the January and July values is approximately the same each year. In other words, the amplitude of the seasonal effect is the same each year.
# 
# The model similarly assumes that the residuals are roughly the same size throughout the series -- they are a random component that adds on to the other components in the same way at all parts of the series.
# 
# > 2) Multiplicative problems: 
# $$y(t) = \text{ Level } \cdot \text{ Trend } \cdot \text{ Seasonality } \cdot \text{ Noise }$$
# In many time series involving quantities (e.g. money, wheat production, ...), the absolute differences in the values are of less interest and importance than the percentage changes. For example, in seasonal data, it might be more useful to model that the July value is the same proportion higher than the January value in each year, rather than assuming that their difference is constant. Assuming that the seasonal and other effects act proportionally on the series is equivalent to a multiplicative model.
# 
# More about it [here](http://www-ist.massey.ac.nz/dstirlin/CAST/CAST/Hmultiplicative/multiplicative1.html).

# ### Code
# The `statsmodels` library provides an implementation of the naive, or classical, decomposition method in a function called `seasonal_decompose()`. It requires that you specify whether the model is additive or multiplicative. By default, it is additive.
# 
# 
# **Recommendation**: This is just for showing what happens when we decompose a time series. The Naive method does not work on complex time series, so I recommend to take a look at something like [Loess or STL decomposition](https://otexts.com/fpp2/stl.html)

# In[ ]:


from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(ts) #model="additive" by default

# Gather the trend, seasonality and noise of decomposed object
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# Plot gathered statistics
fig, axs = plt.subplots(4,sharex=True,figsize=(11,7),gridspec_kw={'hspace': 0})

axs[0].plot(ts, label='Original', color="blue") #Original data
axs[1].plot(trend, label='Trend', color="red") #Trend
axs[2].plot(seasonal,label='Seasonality', color="green") #Season
axs[3].plot(residual, label='Residuals', color="brown") #Residual

plt.show()


# Observe that the data is increasing with respect to time but the variance is at a stable level. Thus, this is an additive model. The trend line is linearly increasing. The third graph so that there is some seasonal pattern in the `ts` data set, and there is some noise in the data. You can scroll to the top to see how the `ts` data set was created.

# This is the end of the first part of this sequel. I hope that you will be able to use these ideas to implement your time series model!

# # IV. References
# 
# A big thanks to [learn.co author](https://github.com/learn-co-students) and [Jason Brownlee](http://www-ist.massey.ac.nz/dstirlin/CAST/CAST/Hmultiplicative/multiplicative1.html) who provided me the idea and sources to write this blog.
