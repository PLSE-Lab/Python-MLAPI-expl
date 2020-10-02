#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import warnings
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
# First we will read the csv file into a pandas dataframe
poll =  pd.read_csv('pollution.csv')

# We are only interested in the AirQualityIndex values and not the absolute values. There is also a lot of location data available
# that we are not interested in, we are going to group by the state and remove the redundent data so that our processing is faster
# Lets drop the columns for the unnecessary data from the Pandes dataframe
poll.head()

## Prepare all 4 AQIs against state and date 
pollSt = poll[['State','Date Local','NO2 AQI','O3 AQI','SO2 AQI','CO AQI']]

# We delete a row with a blank value, we could populate it with the mean but since our dataset is big enough, it will not have
# a strong effect on our analysis
pollSt = pollSt.dropna(axis='rows') 

# We are only interested in the US data for this analysis
pollSt = pollSt[pollSt.State!='Country Of Mexico'] 

# Lets format the date string to an actual date value so that we can act on ranges later
pollSt['Date Local'] = pd.to_datetime(pollSt['Date Local'],format='%Y-%m-%d')

# Group by the State and Date, if there are any duplicate values (Some dates have multiple samples for AQI),
# We will take the mean of thos duplicate values
pollSt = pollSt.groupby(['Date Local']).mean()

yn = pollSt['NO2 AQI'].resample('M').mean()
yo = pollSt['O3 AQI'].resample('M').mean()
yc = pollSt['CO AQI'].resample('M').mean()
ys = pollSt['SO2 AQI'].resample('M').mean()

plt.plot(yc)

from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=12)
    rolstd = pd.rolling_std(timeseries, window=12)

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)
    


test_stationarity(yc)
#take a log
yc_log=np.log(yc)
plt.plot(yc_log)
test_stationarity(yc_log)


# Moving average
moving_avg = pd.rolling_mean(yc_log,12)
plt.plot(yc_log)
plt.plot(moving_avg, color='red')
yc_log_moving_avg_diff = yc_log - moving_avg
yc_log_moving_avg_diff.dropna(inplace=True)
test_stationarity(yc_log_moving_avg_diff)
# exponentially weighted moving average
expwighted_avg = pd.ewma(yc_log, halflife=12)
plt.plot(yc_log)
plt.plot(expwighted_avg, color='red')
yc_log_ewma_diff = yc_log - expwighted_avg
test_stationarity(yc_log_ewma_diff)

# Eliminating Trend and Seasonality
#first order differencing 
yc_log_diff = yc_log - yc_log.shift()
plt.plot(yc_log_diff)
yc_log_diff.dropna(inplace=True)
test_stationarity(yc_log_diff)

# Decomposing
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(yc_log)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(yc_log, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()
yc_log_decompose = residual
yc_log_decompose.dropna(inplace=True)
test_stationarity(yc_log_decompose)

from statsmodels.tsa.stattools import acf, pacf
lag_acf = acf(yc_log_diff, nlags=20)
lag_pacf = pacf(yc_log_diff, nlags=20, method='ols')
#Plot ACF: 

plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(yc_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(yc_log_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')
#Plot PACF:

plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(yc_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(yc_log_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()


p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))

# Generate all different combinations of seasonal p, q and q triplets
seasonal_pdq = [(x[0], x[1], x[2],12) for x in list(itertools.product(p, d, q))]

print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

warnings.filterwarnings("ignore") # specify to ignore warning messages

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(yc_log,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()

            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue
        
 mod = sm.tsa.statespace.SARIMAX(yc_log,
                                order=(1,0,1),
                                seasonal_order=(1, 0, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results = mod.fit()

print(results.summary().tables[1])

results.plot_diagnostics(figsize=(15, 12))
plt.show()
        
pred = results.get_prediction(start=pd.to_datetime('2015-06-30 00:00:00'), dynamic=False)
pred_ci = pred.conf_int()
ax = yc_log['2000':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7)

ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)

ax.set_xlabel('Date')
ax.set_ylabel('CO AQI Levels')
plt.legend()

plt.show()

yc_forecasted = np.exp(pred.predicted_mean)
yc_truth =np.exp( yc_log['2015-06-30 00:00:00':])

# Compute the mean square error
mse =( (yc_forecasted - yc_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

pred_dynamic = results.get_prediction(start=pd.to_datetime('2015-06-30 00:00:00'), dynamic=True, full_results=True)
pred_dynamic_ci = pred_dynamic.conf_int()

ax = yc_log['2000':].plot(label='observed', figsize=(20, 15))
pred_dynamic.predicted_mean.plot(label='Dynamic Forecast', ax=ax)

ax.fill_between(pred_dynamic_ci.index,
                pred_dynamic_ci.iloc[:, 0],
                pred_dynamic_ci.iloc[:, 1], color='k', alpha=.25)

ax.fill_betweenx(ax.get_ylim(), pd.to_datetime('2015-06-30 00:00:00'), yc_log.index[-1],
                 alpha=.1, zorder=-1)

ax.set_xlabel('Date')
ax.set_ylabel('CO AQI')

plt.legend()
plt.show()

# Extract the predicted and true values of our time series
yc_forecasted = np.exp(pred_dynamic.predicted_mean)
yc_truth = np.exp(yc_log['1998-01-01':])

# Compute the mean square error
mse = ((yc_forecasted - yc_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

# Get forecast 500 steps ahead in future
pred_uc = results.get_forecast(steps=12)

# Get confidence intervals of forecasts
pred_ci = pred_uc.conf_int()

ax = yc_log.plot(label='observed', figsize=(20, 15))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('CO AQI')

plt.legend()
plt.show()


np.exp(pred_uc.predicted_mean)











# In[ ]:




