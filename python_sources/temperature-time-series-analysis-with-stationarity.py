#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose

get_ipython().run_line_magic('matplotlib', 'inline')


# # Global Temperature Time Series Data

# This data contains average land temperature data by country and state measured monthly since 1750. I received this dataset from Kaggle.
# https://www.kaggle.com/berkeleyearth/climate-change-earth-surface-temperature-data
# 
# In this post, I will be carrying out some exploratory data analysis and exploring different ways to make a time series stationary.

# # Preparing the Data

# Steps:
#     1. Import the dataset
#     2. Review what information is provided 
#     3. Check for any null values
#     4. Clean the data 

# Step 1: Import the data

# In[7]:


temp = pd.read_csv('../input/GlobalLandTemperaturesByState.csv')


# Step 2: Review what information is in the dataset

# In[8]:


print (temp.head())
print('')
print ('Dtypes')
print (temp.dtypes)
print('')
print ('Shape')
print (temp.shape)


# Step 3: Check for null values

# In[9]:


# Counts the number of null values in each column
temp.isnull().sum()


# Because there are many null values, I don't want to just delete them. Instead I'm going to create dummy variables that indicate if we have temperature data for that row or not.

# In[10]:


# Creating a new column indicating if we have a null value in the Avg_temp column (1) or not (0)
temp['Have_temp_data'] = temp['AverageTemperature'].apply(lambda x: 1 if not pd.isnull(x) else 0)
temp.head()


# In[11]:


# Verifying all null values have 0 
temp['Have_temp_data'].value_counts()


# In[12]:


# Null values by country
temp['Have_temp_data'].groupby(temp['Country']).value_counts()


# Step 4: Cleaning the data

# In[13]:


# I want to rename the column names so they are more concise
temp.rename(columns={'dt':'Date', 'AverageTemperature':'Avg_temp', 'AverageTemperatureUncertainty':'Temp_confidence_interval'}, inplace=True)
temp.head()


# In[14]:


# Convert the date column to datetime series
temp['Date'] = pd.to_datetime(temp['Date'])
temp.set_index('Date', inplace=True)
temp.index


# In[15]:


# I want to extract just the year from the date column and create it's own column.
temp['Year'] = temp.index.year
temp.head()


# In[16]:


# Reviewing if there's null values in the last years of the data grouped by Year
temp['Have_temp_data'].groupby(temp.index.year).value_counts().tail(45)


# It seems that most of the null values are not found towards the end of the time series data (except in 2013 where there are a handful). I'm going to analyze the data based on the years 1970-2013.

# # Analyzing the Data

# Steps:
#     1. Structure the data
#     2. Visualize the data
#     3. Complete analysis of the data

# Step 1: Structure the data

# In[17]:


# Statistical information by column
temp.describe()


# In[18]:


# Filtering by years 1970-2013 because these didn't have many null values
recent_temp = temp.loc['1970':'2013']
recent_temp.head()


# In[19]:


#  Statistical information by country
recent_temp.groupby('Country').describe()


# In[20]:


# Shows the average temperature by country in descending order
recent_temp[['Country','Avg_temp']].groupby(['Country']).mean().sort_values('Avg_temp',ascending=False)


# Step 2: Visualize the time series

# In[21]:


recent_temp[['Avg_temp']].plot(kind='line',title='Temperature Changes from 1970-2013',figsize=(12,6))


# Step 3: Complete analysis of the time series

# Here I'm going to review how to determine if a time seriees is stationary and if it isn't, then list a few ways to make the series stationary.
# 
# Steps:
# 
#     a) Resample the time series to create a more distinct line
# 
#     b) Test stationary with the Dickey-Fuller Test 
# 
#     c) Transform the data to make it stationary, if needed 
# 
#     d) Review SARIMA with ACF and PACF

# Step 3a: Resample the time series

# In[22]:


# Resampling annual averages 
temp_resamp = recent_temp[['Avg_temp']].resample('A').mean()

# Temperature graph 
temp_resamp.plot(title='Temperature Changes from 1970-2013',figsize=(8,5))
plt.ylabel('Temperature',fontsize=12)
plt.xlabel('Year',fontsize=12)
plt.legend()

plt.tight_layout()


# Step 3b: Dickey-Fuller Test

# The Dickey-Fuller test determines how stationary a time series is. If trends affect the time series (i.e. the mean or variance is not constant), then it is not stationary. This test will show stationary with the Test Statistic less than the critical value.
# 
# The null hypothesis is that the time series is not stationary and is affected by trends.

# In[23]:


# Dickey-Fuller test
from statsmodels.tsa.stattools import adfuller

print ('Dickey-Fuller Test Results:')
dftest = adfuller(temp_resamp.iloc[:,0].values, autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print (dfoutput)


# The Test Statstic is greater than the critical value. Therefore, we have failed to reject the null hypothesis at this point. 
# 
# We can visualize that the time series isn't stationary yet by separating the trend component.

# In[24]:


# Decomposing the data
temp_decomp = seasonal_decompose(temp_resamp, freq=3)  

# Extracting the components
trend = temp_decomp.trend
seasonal = temp_decomp.seasonal
residual = temp_decomp.resid

# Plotting the original time series
plt.subplot(411)
plt.plot(temp_resamp)
plt.xlabel('Original')
plt.figure(figsize=(6,4))

# Plotting the trend component
plt.subplot(412)
plt.plot(trend)
plt.xlabel('Trend')
plt.figure(figsize=(6,4))

# Plotting the seasonal component
plt.subplot(413)
plt.plot(seasonal)
plt.xlabel('Seasonal')
plt.figure(figsize=(6,4))

# Plotting the residual component
plt.subplot(414)
plt.plot(residual)
plt.xlabel('Residual')
plt.figure(figsize=(6,4))

plt.tight_layout()


# In[25]:


# Graphing just the trend line 
trend.plot(title='Temperature Trend Line',figsize=(8,4)) 

# Graph labels
plt.xlabel('Year',fontsize=12)
plt.ylabel('Temperature',fontsize=12)

plt.tight_layout()


# The increasing trend line in the decomposition proves that this data currently isn't stationary. We want the trend line to be constant over time.

# Step 3c: Transformation

# There are a few ways to use transformation to make the data stationary:
#     1. Moving average 
#     2. Exponential smoothing 
#     3. Shifting  
#     4. Discomposing the residuals

# I'm going to first compare the transformation using both the moving average (rolling mean) and exponential smoothing. The rolling mean will take a window of "k" values and average them. The exponentially weighted mean uses "exponential decay" which decreases the weight of the previous means over time. 

# In[26]:


# Rolling mean 
temp_rol_mean = temp_resamp.rolling(window=3, center=True).mean()

# Exponentially weighted mean 
temp_ewm = temp_resamp.ewm(span=3).mean()

# Rolling standard deviation 
temp_rol_std = temp_resamp.rolling(window=3, center=True).std()

# Creating subplots next to each other
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))

# Temperature graph with rolling mean and exponentially weighted mean
ax1.plot(temp_resamp,label='Original')
ax1.plot(temp_rol_mean,label='Rolling Mean')
ax1.plot(temp_ewm, label='Exponentially Weighted Mean')
ax1.set_title('Temperature Changes from 1970-2013',fontsize=14)
ax1.set_ylabel('Temperature',fontsize=12)
ax1.set_xlabel('Year',fontsize=12)
ax1.legend()

# Temperature graph with rolling STD 
ax2.plot(temp_rol_std,label='Rolling STD')
ax2.set_title('Temperature Changes from 1970-2013',fontsize=14)
ax2.set_ylabel('Temperature',fontsize=12)
ax2.set_xlabel('Year',fontsize=12)
ax2.legend()

plt.tight_layout()
plt.show()


# In[27]:


# Dickey-Fuller test 
temp_rol_mean.dropna(inplace=True)
temp_ewm.dropna(inplace=True)
print ('Dickey-Fuller Test for the Rolling Mean:')
dftest = adfuller(temp_rol_mean.iloc[:,0].values, autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print (dfoutput)
print ('')
print ('Dickey-Fuller Test for the Exponentially Weighted Mean:')
dftest = adfuller(temp_ewm.iloc[:,0].values, autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print (dfoutput)


# Here both test statistics are greater than the critical values, so we have failed to reject the null hypothesis.

# We can use differencing to remove the moving average or exponentially weighted mean from the original time series. We can then drop any rows that are N/A.

# In[28]:


# Difference between the original and the rolling mean 
diff_rol_mean = temp_resamp - temp_rol_mean
diff_rol_mean.dropna(inplace=True)
diff_rol_mean.head()


# In[29]:


# Difference between the original and the exponentially weighted mean
diff_ewm = temp_resamp - temp_ewm
diff_ewm.dropna(inplace=True)
diff_ewm.head()


# In[31]:


# Rolling mean of the difference
temp_rol_mean_diff = diff_rol_mean.rolling(window=3, center=True).mean()

# Expotentially weighted mean of the difference
temp_ewm_diff = diff_ewm.ewm(span=3).mean()

# Creating subplots next to each other
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))

# Difference graph with the rolling mean
ax1.plot(diff_rol_mean,label='Original')
ax1.plot(temp_rol_mean_diff,label='Rolling Mean')
ax1.set_title('Temperature Changes from 1970-2013',fontsize=14)
ax1.set_ylabel('Temperature',fontsize=12)
ax1.set_xlabel('Year',fontsize=12)
ax1.legend()

# Difference graph with the exponentially weighted mean
ax2.plot(diff_ewm,label='Original')
ax2.plot(temp_ewm_diff,label='Exponentially Weighted Mean')
ax2.set_title('Temperature Changes from 1970-2013',fontsize=14)
ax2.set_ylabel('Temperature',fontsize=12)
ax2.set_xlabel('Year',fontsize=12)
ax2.legend()

plt.tight_layout()


# In[33]:


# Dickey-Fuller test 
print ('Dickey-Fuller Test for the Difference between the Original and Rolling Mean:')
dftest = adfuller(diff_rol_mean.iloc[:,0].values, autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print (dfoutput)
print ('')
print ('Dickey-Fuller Test for the Difference between the Original and Exponentially Weighted Mean:')
dftest = adfuller(diff_ewm.iloc[:,0].values, autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print (dfoutput)


# The test statistic is less than the critial value for both the rolling mean and exponentially weighted mean, indicating that we can reject the null hypothesis. We can be 99% confident that this data is stationary.

# A third way to remove the trend is by shifting values. 

# In[34]:


# Shifting forward by 1 year
temp_shift1 = temp_resamp.shift(1)
temp_shift1.head()


# In[36]:


# Difference between the original and time series shifted by 1 year 
shift1_diff = temp_resamp - temp_shift1
shift1_diff.dropna(inplace=True)

# Rolling mean 
temp_shift1_diff_rol_mean = shift1_diff.rolling(window=3, center=True).mean()

# Exponentially weighted mean 
temp_shift1_diff_ewm = shift1_diff.ewm(span=3).mean()

# Rolling standard deviation 
temp_shift1_diff_rol_std = shift1_diff.rolling(window=3, center=True).std()

# Creating subplots next to each other
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))

# Temperature graph 
ax1.plot(shift1_diff,label='Original')
ax1.plot(temp_shift1_diff_rol_mean,label='Rolling Mean')
ax1.plot(temp_shift1_diff_ewm,label='Exponentially Weighted Mean')
ax1.set_title('Shifted By 1 Year Temperature Changes from 1970-2013',fontsize=14)
ax1.set_ylabel('Temperature',fontsize=12)
ax1.set_xlabel('Year',fontsize=12)
ax1.legend()

# Temperature Rolling STD graph
ax2.plot(temp_shift1_diff_rol_std)
ax2.set_title('Shifted By 1 Year Rolling Standard Deviation',fontsize=14)
ax2.set_ylabel('Temperature',fontsize=12)
ax2.set_xlabel('Year',fontsize=12)

plt.tight_layout()
plt.show()


# In[38]:


# Dickey-Fuller test 
print ('Dickey-Fuller Test for Difference between the Original and Shifted by 1 Year:')
dftest = adfuller(shift1_diff.iloc[:,0].values, autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print (dfoutput)


# We can reject the null hypothesis because the test statistic is less than the critical value. We can be 99% confident that this data is stationary.

# A fourth way is to decompose the time series to extract just the residual component (and remove the trend and seasonality components). I had already decomposed the time series in step 3b, so I will graph just the residuals.

# In[39]:


# Drop N/A values
residual.dropna(inplace=True)

# Residuals rolling mean
resid_rol_mean = residual.rolling(window=3).mean()

# Residuals exponentially weighted mean
resid_ewm = residual.ewm(span=3).mean()

# Residuals rolling standard deviation 
resid_rol_std = residual.rolling(window=3, center=True).std()

# Creating subplots next to each other
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))

# Temperature graph with residual rolling mean and exponentially weighted mean
ax1.plot(residual,label='Original')
ax1.plot(resid_rol_mean,label='Rolling Mean')
ax1.plot(resid_ewm, label='Exponentially Weighted Mean')
ax1.set_title('Residuals',fontsize=14)
ax1.set_ylabel('Temperature',fontsize=12)
ax1.set_xlabel('Year',fontsize=12)
ax1.legend()

# Temperature graph with residual rolling STD 
ax2.plot(resid_rol_std,label='Rolling STD')
ax2.set_title('Residuals Rolling STD',fontsize=14)
ax2.set_ylabel('Temperature',fontsize=12)
ax2.set_xlabel('Year',fontsize=12)
ax2.legend()

plt.tight_layout()
plt.show()


# In[40]:


# Dickey-Fuller test 
print ('Dickey-Fuller Test for the Residuals:')
dftest = adfuller(residual.iloc[:,0].values, autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print (dfoutput)


# We can see that the test statistic is less than the critical values, therefore we can reject the null hypothesis.

# Step 3d: SARIMA 

# Here I'll review how you can find the parameters for the Seasonal Autoregressive Integrated Moving Average (SARIMA) which forecasts a time series similarly to linear regression and has parameters (p,d,q) (P,D,Q). The first set of parameters (p,d,q) is applied to the non-seasonal portion of the time series while the second set (P,D,Q) is applied to the seasonal portion of the time series.

# The parameters are broken down as:    
# 1. Autoregressive (p): looks at the past values. I.e., if the temperature increased over the past 3 years, it's most likely to increase next year as well.
# 2. Integrated (d): looks at the difference between the past values and the current value. I.e., if the temperature differed little over the past 3 years, then it's most likely to be the same temperature next year.
# 3. Moving average (q): uses a linear combination of errors in past values for the predicted error of the model (errors being the difference between the moving average and actual values). I.e., for q=5, x(t) is e(t-1)...e(t-5).

# To determine the p and q parameters, we can use autocorrelation function (ACF) and partial autocorrelation function (PACF).

# In[41]:


from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from matplotlib import pyplot

# Plotting the autocorrelation and partial autocorrelation graphs
pyplot.figure(figsize=(10,5))
pyplot.subplot(211)
plot_acf(temp_resamp, ax=pyplot.gca())
pyplot.subplot(212)
plot_pacf(temp_resamp, ax=pyplot.gca())
pyplot.show()


# The p parameter will be the first value on the Partial Autocorrelation graph that is signficantly different from the previous values. Here, p could be 28.
# 
# The q parameter is the largest lag on the Autocorrelation plot that is significantly differeent from the previous values. Here, q could be 0 as there are no signficant lags.
# 
# Next, the d parameter will be the number of differences taken to make the time series stationary. For example, d could be 1 when using one of the transformations above.
# 
# You can then apply this logic to the seasonality parameters (P,D,Q)!

# # Conclusion

# We showed how to brielfy explore some temperature time series data and then broke down how we can identify if the series is stationary or not by using the Dickey-Fuller Test.

# If a time series data is not stationary, then some ways we can make it stationary is with:
#     1. Moving average
#     2. Exponential smoothing
#     3. Shifting
#     4. Decomposing the residuals

# We can then use SARIMA to forecast the time series data.

# In[ ]:




