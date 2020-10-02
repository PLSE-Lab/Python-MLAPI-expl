#!/usr/bin/env python
# coding: utf-8

# Removing the default index column

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import seaborn as sns

df = pd.read_csv('/kaggle/input/irish-weather-hourly-data/hourly_irish_weather.csv').iloc[:,1:]


# Changing the column name 'date' to 'date_stamp' as it contains both date and time values

# In[ ]:


cols = df.columns.tolist()
cols[0] = 'date_stamp'
df.columns = cols

df.isna().sum()


# Seperating the date and time stamp into date, time, year, month, day and hour columns

# In[ ]:


df[['date','timestamp']] = df['date_stamp'].str.split(' ',expand=True)

df[['year','month','day']] = df['date'].str.split('-',expand=True)

df['hour'] = df['timestamp'].str[0:2]


# In[ ]:


df.head(5)


# Correlation matrix between the variables in the dataframe

# In[ ]:


corr_matrx =df.corr()
corr_matrx


# Filtering only the strongly correlated variables

# In[ ]:


corr_matrx[(corr_matrx > 0.5) | (corr_matrx < - 0.5)]


# High correlation between 1) Temp - Wet bulb temp - dew point temp - vapour pressure
#                          2) Cloud Ceiling Height - clamtCloud Amount

# Subsetting only Galway weather data

# In[ ]:


galway_df = df[(df['county'] == 'Galway') ]
galway_df.count()


# In[ ]:


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
get_ipython().run_line_magic('matplotlib', 'inline')


# Grouping the data by year and month and then aggregating the mean data for Temperature variable

# In[ ]:


tmp_gpby =galway_df.groupby(['year','month']).agg({'temp':np.mean})

tmp_gpby[tmp_gpby == np.nan].sum()


# Filtering indexes of rows with null value for temperature

# In[ ]:


rolling_index_diff_df = pd.DataFrame(pd.Series(galway_df[galway_df['temp'].isna()].index),columns=['index'])


# User defined function to fill the missing values in Temperature variable
# 
# The missing values for a particular day and time are filled by that particular month's average temperature 

# In[ ]:



check_list = galway_df['temp'].isna().index.tolist()

def impute():
    
    op = []
    
    for i in galway_df[galway_df['temp'].isna()].index:
        [year,month] = galway_df.loc[i,['year','month']].tolist()
        op.append(tmp_gpby.loc[(year,month)][0])
    return op

op = impute()

galway_df.loc[galway_df['temp'].isna(),'temp'] = op


# Setting 'date_stamp' column as DateTime Index

# In[ ]:


galway_df.index = pd.DatetimeIndex(galway_df['date_stamp'])
#galway_df.drop(columns=['date_stamp','rolling_temp','rolling_sd'],inplace=True)


# In[ ]:


mean = galway_df['temp'].rolling(window = 24*30).mean()
sd =  galway_df['temp'].rolling(window = 24*30).std()

dummy = galway_df.copy()
dummy['rolling_temp'] = mean
dummy['rolling_sd'] =  sd

galway_df = dummy.copy()


# In[ ]:


galway_df


# 1 Grouping the county galway weather data using date column and collecting the mean aggregate for the Temperature     column
# 
# 2 Dropping the variables which has no data captured

# In[ ]:


galway_daywise_df = galway_df.groupby(['date']).agg({'temp':np.mean})
galway_daywise_df.index = pd.DatetimeIndex(galway_daywise_df.index)
galway_df.drop(columns=['w','ww','sun','vis','clht','clamt'],inplace=True) #no captures for this variables [**GALWAY***]


# 1) Plotting the day wise weather data for galway 
# 2) Configuring the plotting parameters

# In[ ]:


import matplotlib.dates as mdates



ax = galway_daywise_df.plot(figsize=(20,10))

ax.xaxis.set_major_locator(mdates.YearLocator(1))
# set formatter
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

ax


# Performing Augmented Dickey Fuller Test - A Statistical Test for checking the stationarity of the time series data
# 
# Null Hypothesis : The data is not stationary
# Alternate Hpothesis: The data is stationary

# In[ ]:


from statsmodels.tsa.stattools import adfuller
adfuller(galway_daywise_df['temp'])

#p-value is less than 0.05 (95% CI), so rejecting the null hypothesis, therefore at 95% confidence, we have enough evidence to
#support the claim that the data is stationary


# Plotting Lag plot for correlation between current temperature and lagged observations and Auto Correlation Function (ACF) and Partial Autocorrelation Function (PACF) Plots for getting optimal 'p' and 'q' parameters of AR (Auto Regression) and MA (Moving Average) models

# In[ ]:


#Checking for the relationship between the current temperature observations and the lagged observations - should indicate either strong 
#positive or strong negative correlation - for the data to be stationary
from pandas.plotting import lag_plot
lag_plot(galway_daywise_df['temp'])


# In[ ]:


from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

p = plot_acf(galway_daywise_df['temp'],lags=30)

p.set_size_inches(15, 8)


# In[ ]:


p = plot_pacf(galway_daywise_df['temp'],lags=30)

p.set_size_inches(15, 8)


# Decomposing the galway daywise weather time series data into Trend,Seasonality and Residuals 
# 
# The frequency is set to year

# In[ ]:


from statsmodels.tsa.seasonal import seasonal_decompose

seasonal_decompose(galway_daywise_df['temp'],freq=30*12).plot();


# Modelling the galway county time series data using SARIMAX (Seasonal Auto Regression Integrating Moving Average with Exogeneous Variable) model 

# ### SARIMAX - MULTIVARIATE

# In[ ]:


galway_multivariate_df = galway_df.copy()
galway_multivariate_df.drop(columns=['station','county','longitude','latitude'],inplace=True)
galway_multivariate_df = galway_multivariate_df.iloc[:,0:14]
galway_multivariate_df.isna().sum()


# Grouping the data by year and month and finding the mean for all the columns

# In[ ]:


multivariate_gpy = galway_multivariate_df.groupby(['year','month']).agg({'rain':'mean', 'temp':'mean', 'wetb':'mean', 'dewpt':'mean', 'vappr':'mean', 'rhum':'mean', 'msl':'mean', 'wdsp':'mean', 'wddir':'mean'})


# 'Interpolating the multivariate time series data using time interpolation

# In[ ]:


galway_multivariate_df = galway_multivariate_df.interpolate(method='time')


# Ensuring no missing values after time interpolation

# In[ ]:


galway_multivariate_df.isna().sum()


# In[ ]:


galway_multivariate_daywise_df = galway_multivariate_df.groupby(['date']).agg({'rain':'mean', 'temp':'mean', 'wetb':'mean', 'dewpt':'mean', 'vappr':'mean', 'rhum':'mean', 'msl':'mean', 'wdsp':'mean', 'wddir':'mean'})
galway_multivariate_daywise_df.index = pd.DatetimeIndex(galway_multivariate_daywise_df.index)


# Splitting the data into train set and test set
# 
# Train set is used for modelling and the test set is used for validating the model
# 
# Train set - Except last year data
# Test set - Last year data

# In[ ]:


train = galway_multivariate_daywise_df.iloc[:4525,:]
test = galway_multivariate_daywise_df.iloc[4525:,:]

train.index  = pd.DatetimeIndex(train.index).to_period('D')


# Exogenous variables are the variables which probably has relationship with dependent variable and influential in predicting the dependent variable 

# Training the data with SARIMAX model

# In[ ]:


from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(train['temp'],exog=train[['wdsp', 'vappr', 'wetb', 'dewpt', 'rhum', 'msl', 'rain', 'wddir']],order=(3, 0, 2), seasonal_order=(0,0,0,12))

model_fit = model.fit()


# Forecasting the next 1 year data using the model built above and comparing the results with the test set 'temperature' values

# In[ ]:


yhat = model_fit.predict(4525, 4525+364,exog=test[['wdsp', 'vappr', 'wetb', 'dewpt', 'rhum', 'msl', 'rain', 'wddir']])
print(yhat)


# Comparing the test set temperatures and forecasted temperatures by plotting them

# In[ ]:


prediction_df = yhat.to_frame()
prediction_df['actual'] = test['temp'].values

prediction_df.rename(columns={0:'pred'},inplace=True)
prediction_df.index =  test.index


prediction_df.plot(figsize=(20,12))


# Getting the SARIMAX model summary 

# In[ ]:


model_fit.summary()


# Checking the variance in the forecasted values and the actual test values by getting the MSE (Mean Squared Error) and MAE (Mean Absolute Error)

# In[ ]:


from sklearn.metrics import mean_squared_error,mean_absolute_error

print(mean_squared_error(prediction_df['actual'],prediction_df['pred']))
print(mean_absolute_error(prediction_df['actual'],prediction_df['pred']))

