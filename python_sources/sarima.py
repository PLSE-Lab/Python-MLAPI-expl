#!/usr/bin/env python
# coding: utf-8

# # Store Item Demand Forecasting Challenge

# ## Seasonal Autoregressive Integrated Moving Average (SARIMA)
# 
# <a href="https://www.kaggle.com/c/demand-forecasting-kernels-only">Link to competition on Kaggle.</a>
# 
# SARIMA is a variant on the <a href="https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average">ARIMA</a> model for datasets with a suspected seasonal effect.

# In[ ]:


import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

pd.options.display.max_columns = 99
plt.rcParams['figure.figsize'] = (12, 8)


# ## Load Data

# In[ ]:


df_train = pd.read_csv('../input/train.csv', parse_dates=['date'], index_col=['date'])
df_test = pd.read_csv('../input/test.csv', parse_dates=['date'], index_col=['date'])
df_train.shape, df_test.shape


# In[ ]:


df_train.head()


# In[ ]:


num_stores = len(df_train['store'].unique())
fig, axes = plt.subplots(num_stores, figsize=(8, 16))

for s in df_train['store'].unique():
    t = df_train.loc[df_train['store'] == s, 'sales'].resample('W').sum()
    ax = t.plot(ax=axes[s-1])
    ax.grid()
    ax.set_xlabel('')
    ax.set_ylabel('sales')
fig.tight_layout();


# All stores appear to show identical trends and seasonality; they just differ in scale.

# ## SARIMA
# 
# We will build a SARIMA model for a single store and item, and then retrain it and generate predictions for all time series in the dataset.

# ### Example store and item

# In[ ]:


s1i1 = df_train.loc[(df_train['store'] == 1) & (df_train['item'] == 1)]
s1i1.head()


# In[ ]:


s1i1['sales'].plot();


# ### Time Series Decomposition
# 
# Decompose the example time series into trend, seasonal, and residual components.
# 

# In[ ]:


fig = seasonal_decompose(s1i1['sales'], model='additive', freq=365).plot()


# There is clearly yearly seasonality and a non-stationary, upward trend. We can run a Dickey-Fuller test to examine the stationarity.

# In[ ]:


dftest = adfuller(s1i1['sales'], autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
dfoutput


# The Dickey-Fuller test p-value is lower than I would have expected, but the time series is not considered stationary using a 1% Critical Value and we can see visually that there is an upwards trend.

# ### Apply a seasonal difference
# 
# We should start by seeing if we can remove the trend by taking a seasonal difference.

# In[ ]:


diff_7 = s1i1['sales'].diff(7)
diff_7.dropna(inplace=True)
fig = seasonal_decompose(diff_7, model='additive', freq=365).plot()


# In[ ]:


dftest = adfuller(diff_7, autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
dfoutput


# The Dickey-Fuller test suggests that the trend has been removed, although we can still see visually that there appears to be a trend. We can try and eliminate this by applying a further first difference.

# ### Take first differences

# In[ ]:


diff_1_7 = diff_7.diff(1)
diff_1_7.dropna(inplace=True)
fig = seasonal_decompose(diff_1_7, model='additive', freq=365).plot()


# In[ ]:


dftest = adfuller(diff_1_7, autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
dfoutput


# Clearly the trend has now been eliminated.

# ### Plot ACF and PACF
# 
# The <a href="https://en.wikipedia.org/wiki/Autocorrelation">Autocorrelation Function</a> (ACF) is the correlation of a signal with a delayed copy of itself as a function of delay.
# 
# The <a href="https://en.wikipedia.org/wiki/Partial_autocorrelation_function">Partial Autocorrelation Function</a> (PACF) is the partial correlation of a signal with a delayed copy of itself, controlling for the values of the time series at all shorter delays, as a function of delay.
# 
# Using the ACF and PACF, and some <a href="http://people.duke.edu/~rnau/arimrule.htm">simple heuristics</a>, the approriate Autoregression (AR) and Moving Average (MA) values can be identified for the SARIMA model. These are explained below.

# In[ ]:


fig, ax = plt.subplots(2)
ax[0] = sm.graphics.tsa.plot_acf(diff_1_7, lags=50, ax=ax[0])
ax[1] = sm.graphics.tsa.plot_pacf(diff_1_7, lags=50, ax=ax[1])


# From the ACF, one non-seasonal MA term looks sensible. From the PACF, six non-seasonal AR terms looks sensible.
# 
# From the ACF, one seasonal MA term looks sensible. From the PACF, we could use multiple seasonal AR terms, but will use one to keep the training time reasonable.

# ### Build Model
# 
# From the above analysis, we have identified the following parameters for our seasonal ARIMA(p,d,q)(P,D,Q)m model:
# - <b>p</b>: 6
# - <b>d</b>: 1
# - <b>q</b>: 1
# - <b>P</b>: 1
# - <b>D</b>: 1
# - <b>Q</b>: 1
# - <b>m</b>: 7

# In[ ]:


sarima = sm.tsa.statespace.SARIMAX(s1i1['sales'], trend='n', freq='D', enforce_invertibility=False,
                                   order=(6, 1, 1), seasonal_order=(1, 1, 1, 7))
results = sarima.fit()
print(results.summary())


# #### Example forecast

# In[ ]:


s1i1['fcst'] = results.predict(start='2017-10-01', end='2017-12-31', dynamic=True)
s1i1[['sales', 'fcst']].loc['2017-10-01':].plot();


# ## Make Predictions

# In[ ]:


sarima_results = df_test.reset_index()
sarima_results['sales'] = 0


# In[ ]:


tic = time.time()

for s in sarima_results['store'].unique():
    for i in sarima_results['item'].unique():
        si = df_train.loc[(df_train['store'] == s) & (df_train['item'] == i), 'sales']
        sarima = sm.tsa.statespace.SARIMAX(si, trend='n', freq='D', enforce_invertibility=False,
                                           order=(6, 1, 1), seasonal_order=(1, 1, 1, 7))
        results = sarima.fit()
        fcst = results.predict(start='2017-12-31', end='2018-03-31', dynamic=True)
        sarima_results.loc[(sarima_results['store'] == s) & (sarima_results['item'] == i), 'sales'] = fcst.values[1:]
        
        toc = time.time()
        if i % 10 == 0:
            print("Completed store {} item {}. Cumulative time: {:.1f}s".format(s, i, toc-tic))


# In[ ]:


sarima_results.drop(['date', 'store', 'item'], axis=1, inplace=True)
sarima_results.head()


# In[ ]:


sarima_results.to_csv('sarima_results.csv', index=False)


# In[ ]:




