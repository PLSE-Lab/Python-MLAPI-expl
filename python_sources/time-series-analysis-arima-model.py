#!/usr/bin/env python
# coding: utf-8

# # <center> Time series anlysis on Air passengers data - ARIMA algorithm</center>

# In[ ]:


import numpy as np
import pandas as pd

import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 10, 6
get_ipython().run_line_magic('matplotlib', 'inline')

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA

from datetime import datetime

import warnings
warnings.filterwarnings('ignore')


# ### Path to the data

# In[ ]:


PATH = '../input/'


# ### Filename

# In[ ]:


filename = 'AirPassengers.csv'


# ### Read data

# In[ ]:


data = pd.read_csv(PATH + filename)
data.head()


# In[ ]:


data.tail()


# ### Describe data

# In[ ]:


data.describe()


# ### Data types

# In[ ]:


data.dtypes


# ### Convert type of column 'Month' from 'object' to 'datetime'

# In[ ]:


data['Month'] = pd.to_datetime(data['Month'])
data.dtypes


# ### Convert column 'Month' as index

# In[ ]:


indexedData = data.set_index('Month')
indexedData.head()


# ### Plot indexed data (Date vs Number of passengers)

# In[ ]:


plt.plot(indexedData, color='blue')
plt.xlabel('Date')
plt.ylabel('Number of passengers')


# ### What is stationarity?
# - A time series is said to be stationary if its statistical properties such as mean, variance remain constant over time.
# - The basic assumption before applying stochastic models is that the time series should be stationary.

# - As we see above plot, the data is not stationary. The trend is increasing. The mean is not constant.
# - So, our first goal is to make the time series into stationary.

# - Next step is to find seasonality (s).
# - Let's plot Moving average for 4, 6, 8 and 12 months.

# ### 4- Months Moving Average

# In[ ]:


four_months_moving_average = indexedData.rolling(window=4).mean()
plt.plot(indexedData, color='blue', label='Original')
plt.plot(four_months_moving_average, color='red', label='Rolling Mean')
plt.legend(loc='best')
plt.title('4 Months Moving Average')


# ### 6-Months Moving Average

# In[ ]:


six_months_moving_average = indexedData.rolling(window=6).mean()
plt.plot(indexedData, color='blue', label='Original')
plt.plot(six_months_moving_average, color='red', label='Rolling Mean')
plt.legend(loc='best')
plt.title('6 Months Moving Average')


# ### 8-Months Moving Average

# In[ ]:


eight_months_moving_average = indexedData.rolling(window=8).mean()
plt.plot(indexedData, color='blue', label='Original')
plt.plot(eight_months_moving_average, color='red', label='Rolling Mean')
plt.legend(loc='best')
plt.title('8 Months Moving Average')


# ### 12-Months Moving Average

# In[ ]:


twelve_months_moving_average = indexedData.rolling(window=12).mean()
plt.plot(indexedData, color='blue', label='Original')
plt.plot(twelve_months_moving_average, color='red', label='Rolling Mean')
plt.legend(loc='best')
plt.title('12 Months Moving Average')


# - As we observe the above four moving averages, the moving average with window '12' is smooth compared to others. So, we can confirm that the seasonality(s) is 12.

# ### There are two tests to check whether a time series is stationary or not.
# - Rolling statistics (Visual test)
# - Dickey Fuller test

# ## Rolling statistics - A visual test
# - From the above observations, we can chose 's' as 12. (s=12)

# In[ ]:


rolmean = indexedData.rolling(window=12).mean()


# In[ ]:


plt.plot(rolmean, 'blue')
plt.title('Mean')


# In[ ]:


rolstd = indexedData.rolling(window=12).std()


# In[ ]:


plt.plot(rolstd, 'blue')
plt.title('Standard Deviation')


# - As the data is at monthly level (12), we used window as 12. So, we get 'NaN' for the first 11 months.

# ### Plot rolling statistics

# In[ ]:


plt.plot(indexedData, color='blue', label='Original')
plt.plot(rolmean, color='red', label='Rolling Mean')
plt.plot(rolstd, color='black', label='Rolling Std')

plt.legend(loc='best')
plt.title('Rolling Mean and Standard deviation\n')


# - As we can see, mean and standard deviation are not stationary. So, the time seires is not stationary.

# ## Dickey Fuller test
# -  Let's run the Dicky Fuller Test on the timeseries and verify the null hypothesis that the TS is non-stationary.

# In[ ]:


dftest = adfuller(indexedData['#Passengers'], autolag='AIC')


# In[ ]:


dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', 'No. of Lags used', 'Number of observations used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)' %key] = value


# In[ ]:


dfoutput


# - p-value is high. It should be very less. We fail to reject the null hypothesis as p-value is high. So, the time series is non-stationary.
# - If p-value is less, then we can say that the time series is stationary.

# ## Function to perform both the tests
# - Let's write a function to perform both the tests for us at a time.

# In[ ]:


def test_stationary(timeseries):
    
    # Rolling statistics
    movingAverage = timeseries.rolling(window=12).mean()
    movingSTD = timeseries.rolling(window=12).std()
    
    # Plot rolling statistics
    plt.plot(timeseries, color='blue', label='Original')
    plt.plot(movingAverage, color='red', label='Rolling Mean')
    plt.plot(movingSTD, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation\n')
    plt.show(block=False)
    
    # Dickey Fuller test
    print('Results of Dickey Fuller Test:\n')
    dftest = adfuller(timeseries['#Passengers'], autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', 'No. of Lags used', 'Number of observations used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)' %key] = value
    print(dfoutput)


# In[ ]:


test_stationary(indexedData)


# ### Estimating trend
# - Apply log transform on the indexed_data.

# In[ ]:


indexedData_logScale= np.log(indexedData)


# In[ ]:


plt.plot(indexedData_logScale, 'blue')


# - Trend remains same. The values of the y-axis got changed.

# In[ ]:


movingAverage = indexedData_logScale.rolling(window=12).mean()
movingSTD = indexedData_logScale.rolling(window=12).std()


# In[ ]:


plt.plot(indexedData_logScale, color='blue')
plt.plot(movingAverage, color='red')


# In[ ]:


test_stationary(indexedData_logScale)


# - The time series is not stationary. We can tell just by seeing the above graph. The mean is not constant.

# ### Another transformation.

# In[ ]:


dataLogScaleMinusMovingAverage = indexedData_logScale - movingAverage
dataLogScaleMinusMovingAverage.dropna(inplace=True)
dataLogScaleMinusMovingAverage.head()


# In[ ]:


test_stationary(dataLogScaleMinusMovingAverage)


# - As p-value is less, null hypothesis is rejected. So, it is stationary.

# ### Another transformation

# In[ ]:


exponentialDecayWeightedAverage = indexedData_logScale.ewm(halflife=12, min_periods=0, adjust=True).mean()
plt.plot(indexedData_logScale, 'blue')
plt.plot(exponentialDecayWeightedAverage, 'red')


# In[ ]:


dataLogScaleMinusMovingExponentialDecayAverage = indexedData_logScale - exponentialDecayWeightedAverage
test_stationary(dataLogScaleMinusMovingExponentialDecayAverage)


# - As p-value is less, null hypothesis is rejected. So, it is stationary.

# ### Another way of making the time series stationary is by differencing.
# - Let's difference the log transformed data.

# In[ ]:


dataLogDiffShifting = indexedData_logScale - indexedData_logScale.shift()
plt.plot(dataLogDiffShifting, color='blue')


# In[ ]:


dataLogDiffShifting.dropna(inplace=True)


# In[ ]:


test_stationary(dataLogDiffShifting)


# - Null hypothesis is rejected. Therefore, the time series is stationary now after differencing once.

# ## Components of time series

# In[ ]:


decomposition = seasonal_decompose(indexedData_logScale)


# In[ ]:


# Just for reference
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid


# In[ ]:


fig = decomposition.plot()
fig.set_figwidth(12)
fig.set_figheight(8)
plt.suptitle('Decomposition of multiplicative time series')


# In[ ]:


decomposedLogData = residual
decomposedLogData.dropna(inplace=True)
test_stationary(decomposedLogData)


# The Dickey-Fuller test statistic is significantly lower than the 1% critical value. So this time series is very close to stationary.

# ### ACF and PACF plots
# - While fitting an arima model, we need to find correct 'p', 'd' and 'q'.
# - We find 'd' by differencing the data number of times till it becomes stationary.
# - ACF and PACF plots are very useful in determining the values of p and q.

# In[ ]:


fig, axes = plt.subplots(1, 2, sharey=False, sharex=False)
fig.set_figwidth(12)
fig.set_figheight(4)
plot_acf(dataLogDiffShifting, lags=20, ax=axes[0], alpha=0.5)
plot_pacf(dataLogDiffShifting, lags=20, ax=axes[1], alpha=0.5)
plt.tight_layout()


# - The lag value where the PACF graph crosses the upper confidence interval for the first time. If you notice closely, in this case p=2.
# - The lag value where the ACF graph crosses the upper confidence interval for the first time. If you notice closely, in this case q=2.

# ### Let's fit ARIMA model with (2,1,2)
# - p = 2, d = 1, q = 2

# In[ ]:


model = ARIMA(indexedData_logScale, order=(2,1,2))
results_ARIMA = model.fit(disp=-1)
plt.plot(dataLogDiffShifting, color='blue')
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %4f'% sum((results_ARIMA.fittedvalues - dataLogDiffShifting['#Passengers'])**2))
print('Plotting ARIMA model')


# ### ARIMA model with (2,1,0)
# - p = 2, d = 1, q = 0

# In[ ]:


model = ARIMA(indexedData_logScale, order=(2,1,0))
results_AR = model.fit()
plt.plot(dataLogDiffShifting)
plt.plot(results_AR.fittedvalues, color='red')
plt.title('RSS: %4f'% sum((results_AR.fittedvalues - dataLogDiffShifting['#Passengers'])**2))
print('Plotting ARIMA model')


# ### ARIMA model with (0,1,2)
# - p = 0, d = 1, q = 2

# In[ ]:


model = ARIMA(indexedData_logScale, order=(0,1,2))
results_MA = model.fit()
plt.plot(dataLogDiffShifting)
plt.plot(results_MA.fittedvalues, color='red')
plt.title('RSS: %4f'% sum((results_MA.fittedvalues - dataLogDiffShifting['#Passengers'])**2))
print('Plotting ARIMA model')


# In[ ]:


predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
predictions_ARIMA_diff.head()


# In[ ]:


plt.plot(predictions_ARIMA_diff)


# - As we compare the above models, the model with parameters (2,1,2) has less RSS score.

# ### Back to original scale
# - The way to convert the differencing to log scale is to add these differences consecutively to the base number.

# In[ ]:


predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
predictions_ARIMA_diff_cumsum.head()


# In[ ]:


predictions_ARIMA_log = pd.Series(indexedData_logScale['#Passengers'].ix[0], index=indexedData_logScale.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)
predictions_ARIMA_log.head()


# In[ ]:


predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(indexedData)
plt.plot(predictions_ARIMA)


# - As we see, the model is not too bad. It's okay. Try with different parameters and different transformations to build a better model.
# - We can apply SARIMAX(which considers seasonality into account - P, D, Q) to get more accurate results.

# ### References:
# 1. Time series Analysis by Box and Jenkins - Textbook.
# 2. Analytics Vidya - Time series forecasting.

# Don't forget to Upvote the kernel if you find it useful.

# In[ ]:




