#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import matplotlib.pyplot as plt

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # Time series data pre-processing
# 
# The price time series is only available from calendar.csv, so let us concern with that particular data for now.

# In[ ]:


calendar = pd.read_csv('../input/calendar.csv')
calendar.head()


# Let us convert date from string/object to pandas datetime for easy handling later.

# In[ ]:


calendar['date']=pd.to_datetime(calendar['date'])
calendar.head()


# Price is also still in string/object, hence let us change it into numeric. It is also useful to set the date as index, and later clean up the columns not needed.
# 
# For now, we only take the price when it is available.

# In[ ]:


calendar_available = calendar[calendar['available']=='t']
calendar_available['price_cleaned'] = calendar_available['price'].str.replace('$','').apply(pd.to_numeric, errors='coerce')
calendar_available.index = calendar_available['date']
calendar_available = calendar_available.drop(columns=['date','available','price'])


# There are many possible ways to have a good representation of the price data in any given day. Let us try with average/mean from all listings.

# In[ ]:


mean_price = calendar_available.groupby(calendar_available.index).mean().drop(columns='listing_id')
mean_price.plot()
ax = plt.ylabel('mean price ($)')


# # Time series stationarity
# 
# It is obvious that there is a seasonal trend. However, let us look at the decomposed time series to be clear.

# In[ ]:


from dateutil.relativedelta import relativedelta
import statsmodels.api as sm  
from statsmodels.tsa.stattools import acf  
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(mean_price) 
decomposition.plot()
plt.show()


# It is apparent that the time series is not stationary. A more formal way rather than visual inspection is by using statistics test, e.g. Dickey-Fuller test, to check whether is it appropriate to reject null hypothesis of the data is not stationary.

# In[ ]:


from statsmodels.tsa.stattools import adfuller

def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = timeseries.rolling(window=7).mean()
    rolstd = timeseries.rolling(window=7).std()

    #Plot rolling statistics:
    fig = plt.figure(figsize=(12, 8))
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)


# In[ ]:


test_stationarity(mean_price.price_cleaned)


# From above test, the test statistics is larger than the critical value. Hence, we can't reject the null hypotheses. In other words, the time series is not stationary.
# 
# One way to make the data stationary is to difference the time series.

# In[ ]:


price_diff = mean_price.diff().dropna()
test_stationarity(price_diff.price_cleaned)


# Here the test statistics is less than critical values. So, first order differencing is actually sufficient for the data.
# 
# # Time Series Forecasting
# 
# To do forecasting, we would like to use SARIMA model. This is available in statsmodel package as sarimax. However, It is quite tedious to find the optimum model parameters manually, since there are 6 + 1 parameters needed. We can write the parameter space as (p, d, q) x (P, D, Q, s).
# 
# To do it in a more automated fashion, let us test using Akaike Information Criterion (AIC) as the objective function. In other words, the lower the AIC value, the better the parameters because the model fits better to the data. AIC is preferred over measure such as root mean square error (RMSE) because AIC has penalty for model complexity. The more complex the data, the higher the AIC value. With this approach, we hope that the best bias vs variance tradeoff can be achieved.
# 
# Note that we have checked that at least first order differencing is necessary for the data to be stationary. Since SARIMA assume time series stationarity, we can set d parameter search only from 1, not from zero.

# In[ ]:


import itertools
import warnings

def parameter_search_sarimax(timeseries,s):
    lowest_aic = None
    lowest_parm = None
    lowest_param_seasonal = None
    
    p = D = q = range(0, 3)
    d = [1, 2]
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], s) for x in list(itertools.product(p, D, q))]

    warnings.filterwarnings('ignore') # specify to ignore warning messages

    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(timeseries,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)

                results = mod.fit()
                
                # Store results
                current_aic = results.aic
                # Set baseline for aic
                if (lowest_aic == None):
                    lowest_aic = results.aic
                # Compare results
                if (current_aic <= lowest_aic):
                    lowest_aic = current_aic
                    lowest_parm = param
                    lowest_param_seasonal = param_seasonal

                print('SARIMA{}x{} - AIC:{}'.format(param, param_seasonal, results.aic))
            except:
                continue
            
    print('The best model is: SARIMA{}x{} - AIC:{}'.format(lowest_parm, lowest_param_seasonal, lowest_aic))


# For s, or seasonality, parameter we can give a test for different values. Let us try with 12 as many other examples used.

# In[ ]:


parameter_search_sarimax(mean_price,12)


# After we got the "optimum" parameters, it is necessary to validate the model. Here, we will predict part of the time series starting from mid December 2016. We also check if the assumption made by SARIMA model were met for the particular parameter.

# In[ ]:


def model_run_validate(parameter):

    mod = sm.tsa.statespace.SARIMAX(mean_price,
                                    order=(parameter[0], parameter[1], parameter[2]),
                                    seasonal_order=(parameter[3], parameter[4], parameter[5], parameter[6]),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)

    results = mod.fit(maxiter=200)

    print(results.summary().tables[1])

    results.plot_diagnostics(figsize=(15, 12))

    pred_start_date = '2016-12-20'
    pred_dynamic = results.get_prediction(start=pd.to_datetime(pred_start_date), dynamic=True, full_results=True)
    pred_dynamic_ci = pred_dynamic.conf_int()

    ax = mean_price.plot(label='Observed', figsize=(20, 15))
    pred_dynamic.predicted_mean.plot(label='Dynamic Forecast', ax=ax)

    ax.fill_between(pred_dynamic_ci.index,
                    pred_dynamic_ci.iloc[:, 0],
                    pred_dynamic_ci.iloc[:, 1], color='k', alpha=.25)

    ax.fill_betweenx(ax.get_ylim(), pd.to_datetime(pred_start_date), mean_price.index[-1],
                     alpha=.1, zorder=-1)

    ax.set_xlabel('Date')
    ax.set_ylabel('Mean Price ($)')
    ax.legend()


# In[ ]:


parameter = [2,1,2,2,1,1,12]
model_run_validate(parameter)


# Not too bad. The fit is good. Residual is almost normally distributed, too. However, from correlogram, it is apparent that the autocorrelation reveals that the residual is still quite correlated. This invalidates one of the assumption for SARIMA model.
# 
# Time lag of 7 days shows the strongest correlation. It is suggestive that the period of seasonality is 7 days. Hence, let us try with s=7.

# In[ ]:


parameter_search_sarimax(mean_price,7)


# In[ ]:


parameter = [0,1,2,2,1,2,7]
model_run_validate(parameter)


# It appears that by choosing s as 7, the residual is not much correlated anymore. However, the model may be too simple since it use p=0. It would be interesting to try the parameters with the second lowest AIC.
# 
# SARIMA(2, 1, 1)x(2, 1, 2, 7) - AIC:841.5884052299125

# In[ ]:


parameter = [2,1,1,2,1,2,7]
model_run_validate(parameter)


# Let us forecast the next 2 months (60 steps) based on the model parameter.

# In[ ]:


def run_model_forecast(parameter):
    
    mod = sm.tsa.statespace.SARIMAX(mean_price,
                                    order=(parameter[0], parameter[1], parameter[2]),
                                    seasonal_order=(parameter[3], parameter[4], parameter[5], parameter[6]),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)

    results = mod.fit(maxiter=200)
    
    pred_uc = results.get_forecast(steps=60)
    pred_ci = pred_uc.conf_int()
    
    ax = mean_price.price_cleaned.plot(label='Observed', figsize=(20, 15))
    pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.25)
    ax.set_xlabel('Date')
    ax.set_ylabel('Mean Price ($)')
    ax.legend()


# First try with the last retrieved parameter, and second try with the previously simpler model.

# In[ ]:


parameter = [2,1,1,2,1,2,7]
run_model_forecast(parameter)


# In[ ]:


parameter = [0,1,2,2,1,2,7]
run_model_forecast(parameter)


# Both seems plausible. 
# 
# It would be interesting to check integer multiplication of this weekly seasonality.
# We can run again the parameter_search_sarimax using monthly seasonality, i.e. s=28. It is also interesting to check 3monthly, s= 84. To save on running[](http://) time, we only do validation and forecast using the retrieved parameter values and skip the parameter search from this kernel.

# In[ ]:


parameter = [2,1,2,1,1,2,28]
model_run_validate(parameter)


# Not too bad fit. Using s=28 also takes care the 7 days seasonality, since the autocorrelation is still low. Let us see how the forecast goes.

# In[ ]:


run_model_forecast(parameter)


# Not bad. Still plausible. How about using s=84?

# In[ ]:


parameter = [2,1,0,2,2,0,84]
model_run_validate(parameter)


# Unfortunately, the trend during the validated part is a mismatch of the observed trend. The residual is also correlated quite strongly now.
# 
# The forecast below also does not look good.

# In[ ]:


run_model_forecast(parameter)


# So it seems monthly seasonality, s=28, works well and includes also weekly, s=7, seasonality. But 3-monthly is not.
# 
# However, we noted that the data is limited to only one year. Hence, year by year trend is not available to be learnt from the data. We should take the forecasting result with a grain of salt and make the model relearn if a more complete data is available. If at least 2 years data is available, we can try annual seasonality.

# In[ ]:




