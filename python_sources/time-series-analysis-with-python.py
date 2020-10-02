#!/usr/bin/env python
# coding: utf-8

# # Time-series analysis with Python

# In this ever-growing notebook, I will be doing my best to conduct and showcase various time-series analyses with Python. If you like the notebook, please feel free to give it an upvote. 

# ## 1. Introduction to statsmodels for time-series analysis

# The statsmodels library contains an extensive list of descriptive statistics, statistical tests, plotting functions and result statistics for different types of data and estimators. More information about the statsmodels library can be found here: https://www.statsmodels.org/stable/index.html

# In[ ]:


# Importing necessary libraries
import statsmodels.api as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# For this introduction section, I will be using the macrodata dataset provided by the statsmodels api.

# In[ ]:


# Fetching the data from statsmodel api
df = sm.datasets.macrodata.load_pandas().data
df.head()


# A description of the dataset can be found by running the following code.

# In[ ]:


print(sm.datasets.macrodata.NOTE)


# Now that we are clear with what data we are working on, let us introduce a datetime feature out of the 'year' column to conduct time-series analysis. First, we have to find the starting and ending date range of the data.

# In[ ]:


print("The head of the dataset is:")
df.head()


# In[ ]:


print("The tail of the dataset is:")
df.tail()


# As we can observe, the first starting date is 1959 Quarter 1 and the ending date is 2009 Quarter 3. Therefore, creating a data range from 1959Q1 to 2009Q3 and assigning it to the index of the dataframe.

# In[ ]:


index = pd.Index(sm.tsa.datetools.dates_from_range('1959Q1','2009Q3'))
df.index = index


# Looking at the head of the DataFrame again.

# In[ ]:


print("The head of the dataset is:")
df.head()


# Since we now have a datetime feature, we can perform analysis of multiple columnns using time-series analysis. 
# 
# For an example, let us find the cyclical component and trend of the 'realgdp' column. Visualizing the column's data in respect to the datetime values.

# In[ ]:


df['realgdp'].plot()


# The Hodrick-Prescott filter seperates a time-series into a cyclical component and a trend and can be easily called using the hpfilter() method of statsmodels.

# In[ ]:


gdp_cycle, gdp_trend = sm.tsa.filters.hpfilter(df['realgdp'])


# Visualizing the trend using pandas plot() function.

# In[ ]:


df['trend'] = gdp_trend
df[['realgdp','trend']].plot(figsize=(12,8))


# We can see that the trend is modelling the realgdp value and is upward. Let us have an even detailed look into the difference between the 'realgdp' value and the trend line.

# In[ ]:


df[['realgdp','trend']]['2002-03-31':].plot(figsize=(12,8))


# As we can see the trend line is representative of the data and doesn't fit perfectly to the actual values itself. This is because a trend is a smooth, general, long-term, average tendency of the data to increase or decrease during a period of time.

# ## 2. ETS Models for time-series analysis

# An ETS Model stands for Error-Trend-Seasonality model. It takes each of the three terms (Error-Trend-Seasonality) for smoothing and may add them, multiply them or even just leave some of them out. We can create a model to fit our data based off these key factors. 
# 
# We will be using Time Series Decomposition to break down the time-series data into the three key terms.
# 

# In[ ]:


# Importing necessary library
from statsmodels.tsa.seasonal import seasonal_decompose


# Taking a look at the data again by visualizing it, we can see that it is linearly growing.

# In[ ]:


df['realgdp'].plot(figsize=(12,8))


# Calling seasonal_decompose() function of statsmodels will provide us with the ETS values of the data.

# In[ ]:


result = seasonal_decompose(df['realgdp'], model='additive')


# Plotting the result of the decomposition.

# In[ ]:


fig = result.plot()
fig.set_size_inches(12,8)


# This is how we can perform ETS decomposition and look at the Error-Trend-Seasonality of our data.

# ## 3. SMA and EWMA Models for time-series analysis

# Calculating the Simple Moving Average (SMA) of a time-series data can allow us to create a simple model that describes some trend level behavior of it. However, some of the offcomings of the SMA model are the following:
# 
# - Smaller windows will lead to more noise rather than signal.
# - It will always lag by the size of the window.
# - It will never reach to full peak or valley of the data due to averaging.
# - It does not inform about possible future behaviors and it only describes the trend in the data.
# - Extreme historically values can skew SMA significantly.
# 
# We can improve on the basic SMA and fix some of these issue by using an Exponentially Weighted Moving Average (EWMA) model.
# 
# EWMA will allow us to reduce the lag effect from SMA and it will put more weight on values that occurred more recently (by applying more recent values). The amount of weight applied to the most recent values will depend on the actual parameters used in the EWMA and the number of periods given a window size.
# 
# Now, saying all this, let us first move to calculating the Simple Moving Average and then, the Exponentially Weighted Moving Average.

# Calcualting Simple Moving Averages for 12 months.

# In[ ]:


df['12-month-SMA'] = df['realgdp'].rolling(window=12).mean()


# Plotting the data.

# In[ ]:


df[['realgdp','12-month-SMA']].plot(figsize=(12,8))


# As we can see there is a lag at the start of the SMA. However, we can fix that with Exponential Weighted Moving Averages.
# 
# Now, calculating the EWMA with a span of 12 months.

# In[ ]:


df['EWMA-12'] =  df['realgdp'].ewm(span=12).mean()


# In[ ]:


df[['realgdp','EWMA-12']].plot(figsize=(12,8))


# Looking at the plot, we can see that the we do not have any lag at the start of the data in comparison to SMA.

# ## 4. ARIMA Model for time-series analysis

# The ARIMA model is one of the most common time series models and it stands for AutoRegressive Integrated Moving Average.
# 
# It is a generalization of the autoregressive moving average (ARMA) model. Both of these models are fitted to time series data either to better understand the data or to predict future points in the series, i.e., forecasting. The two types of ARIMA models are Non-seasonal ARIMA and Seasonal ARIMA.
# 
# ARIMA models are applied in some cases where data show evidence of non-stationarity, where an initial differencing step (corresponding to the 'integrated' part of the model) can be applied one or more times to eliminate the non-stationarity.
# 
# Non-seasonal ARIMA models are generally denoted as ARIMA(p,d,q) where parameters p, d and q are non-negative integers.
# 
# The major components of non-seasonal ARIMA are:
# 
# - **p (AR - Autoregression):** A regression model that utilizes the dependent relationship between a current observation and observations over a previous period.
# 
# - **d (I - Integrated):** Differencing of observations (subtracting an obervation from an observation at the previous time step) in order to make the time series stationary.
# 
# - **q (MA - Moving Average):** A model that uses the dependency between an observation and a residual error from a moving average model applied to lagged observations.
# 
# Stationary time-series data has constant mean, covariance and variance over time. We can use the Augmented Dickey-Fuller test to test if a time-series data is stationary or not.
# 
# Also, the p, d and q terms can be choosed by looking at **AutoCorrelation Plots** and **Partial AutoCorrelation Plots**.
# 
# ### Autocorrelation Plots and Partial AutoCorrelation Plots
# 
# An autocorrelation plot shows the correlation of the timeseries with itself, lagged by x time units. So, the y-axis of the plot is the correlation and the x-axis of the plot is the number of time units of lags. The two general autocorrelation plots are Gradual Decline and Sharp Drop-off.
# 
# If the autocorrelation plot shows positive autocorrelation at the first lag (lag-1), then it suggests to use the AR terms in relation to the lag.
# 
# If the autocorrelation plot shows negative autocorrelation at the first lag (lag-1), then it suggests to use the MA terms.
# 
# - p: The number of lag observations included in the model.
# - d: The number of times that the raw observations are differenced.
# - q: The size of the moving average window, also called the order of moving average.
# 
# In general, a partial correlation is a conditional correlation. It is the correlation between two variables under the assumption that we know and take into account the values of some other set of variables.
# 
# Typically, a sharp drop after lag 'k' suggests an AR-k model should be used. If there is a gradual decline, it suggests an MA model.
# 
# Key points:
# 
# - Identification of an AR model is often best done with the PACF(unction).
# - Identification of an MA model is often best done with the ACF(unction) rather than the PACF(unction).
# 

# ### Coding an ARIMA Model
# 
# Now, onto the coding. This is our process:
# 
# - Visualize Time Series Data
# - Make the time series data stationary
# - Plot the correlation and autocorrelation charts
# - Construct the ARIMA model
# - Use the model to make predictions

# Plotting the timeseries data.

# In[ ]:


df['realgdp'].plot()


# The data looks seasonal and looks to have an upward trend. Calculating the 12 month SMA and plotting it.

# In[ ]:


time_series = df['realgdp']
time_series.rolling(12).mean().plot(label = '12 Month Rolling Mean')
time_series.rolling(12).std().plot(label = '12 Month Rolling STD')
time_series.plot()
plt.legend()


# Decomposing the time series data into ETS model.

# In[ ]:


decomp = seasonal_decompose(time_series)
fig = decomp.plot()
fig.set_size_inches(12,8)


# Testing if the data is stationary or not using the Augmented Dickey-Fuller test. If the p-value is less than 0.05 then, the time series data is stationary. Thus, our null hypothesis states that the data is non-stationary.

# In[ ]:


# Importing necessary library
from statsmodels.tsa.stattools import adfuller


# In[ ]:


result = adfuller(df['realgdp'])


# In[ ]:


def adf_check(time_series):
    result = adfuller(time_series)
    print("Augmented Dicky-Fuller Test")
    labels = ['ADF Test Statistic', 'p-value', '# of lags','# of observations used']
    
    for value, label in zip(result, labels):
        print(label + " : " + str(value))
        
    if result[1] <= 0.05:
        print("Strong evidence against null hypothesis.\nReject Null Hypothesis.\nData has no unit root and is stationary.")
    else:
        print("Weak evidence against null hypothesis.\nFail to reject Null Hypothesis.\nData has a unit root and is non-stationary.")


# In[ ]:


adf_check(df['realgdp'])


# Taking the first difference and performing ADF test.

# In[ ]:


df['First Difference'] = df['realgdp'] - df['realgdp'].shift(1)
df['First Difference'].plot()


# In[ ]:


adf_check(df['First Difference'].dropna())


# The data is now stationary. Just in case if you want to perform a second difference, here is how you do it.

# In[ ]:


df['Second Difference'] = df['First Difference'] - df['First Difference'].shift(1)
df['Second Difference'].plot()


# In[ ]:


adf_check(df['Second Difference'].dropna())


# The data is still stationary even if we take a second difference.

# Now, plotting ACF and PACF for the two differences that we have took.

# In[ ]:


# Importing necessary libary
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# In[ ]:


fig_first = plot_acf(df['First Difference'].dropna())


# In[ ]:


fig_first_pacf = plot_pacf(df['First Difference'].dropna())


# In[ ]:


fig_second = plot_acf(df['Second Difference'].dropna())


# In[ ]:


fig_second_pacf = plot_pacf(df['Second Difference'].dropna())


# A sharp drop in all of our plots suggests that we should use an AR model.

# Now, time for modelling. Since the data is not seasonal, we should be using the non-seasonal ARIMA model.

# In[ ]:


# Importing necessary library
from statsmodels.tsa.arima_model import ARIMA


# In[ ]:


# Initializing the model
model = sm.tsa.ARIMA(df['realgdp'], order = (1, 0, 0))


# In[ ]:


# Fitting the model
results = model.fit()


# In[ ]:


# Summary of the fit
print(results.summary())


# In[ ]:


# Plotting the residual values
results.resid.plot()


# In[ ]:


# Plotting the residual values KDE
results.resid.plot(kind='kde')


# In[ ]:


# Predicting using the ARIMA model
df['forecast'] = results.predict(start = 160, end = 203)
df[['realgdp','forecast']].plot(figsize = (12, 8))


# Pretty close! 
# 
# Now, finally let's predict the values for the future. We will first have to add datetime values for the future dates that we want to predict.

# In[ ]:


# Importing necessary library
from pandas.tseries.offsets import DateOffset


# In[ ]:


# Since we are working with quarterly data, keeping step in range as 3
future_dates = [df.index[-1] + DateOffset(months = x) for x in range(3,28,3)]


# In[ ]:


# Concatenating to the dataframe
future_df = pd.DataFrame(index=future_dates, columns = df.columns)
final_df = pd.concat([df,future_df])


# In[ ]:


final_df.shape


# In[ ]:


final_df['forecast'] = results.predict(start = 168, end = 212)
final_df[['realgdp','forecast']].plot(figsize = (12, 8))


# And this is how you implement the ARIMA model.

# Note: I will be updating this notebook frequently, so please feel free to check back from time to time. If you like the kernel, hope you give it an upvote.
