#!/usr/bin/env python
# coding: utf-8

# <center>
# <img src="https://habrastorage.org/files/fd4/502/43d/fd450243dd604b81b9713213a247aa20.jpg">
# ## Open Machine Learning Course
# <center>Author: [Yury Kashnitsky](http://yorko.github.io). All content is distributed under the [Creative Commons CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.

# # <center>Topic 9. Time series analysis with Python</center>
# ## <center>Analyzing accidental deaths in US with ARIMA</center>

# [Introduction to ARIMA](https://www.youtube.com/watch?v=Y2khrpVo6qI).
# 
# 
# We know monthly numbers of accidental deaths in the US from January 1973 till December 1978. Let's build predictions for next 2 years.

# In[ ]:


import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = 12, 10
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
from itertools import product
import numpy as np


# **Read and plot data. We can clearly notice seasonality.**

# In[ ]:


deaths = pd.read_csv('../input/mlcourse/accidental-deaths-in-usa-monthly.csv',
                   index_col=['Month'], parse_dates=['Month'])
deaths.rename(columns={'Accidental deaths in USA: monthly, 1973 ? 1978': 'num_deaths'}, inplace=True)
deaths['num_deaths'].plot()
plt.ylabel('Accidental deaths');
deaths.head()


# In[ ]:


power = pd.read_csv('../input/powerdata/power_data.csv',
                    parse_dates=['Date'])
#deaths.rename(columns={'Accidental deaths in USA: monthly, 1973 ? 1978': 'num_deaths'}, inplace=True)
df = power[["Date", "Hour", "Toronto"]]
df['DateTime'] = pd.to_datetime(df.Date) + pd.to_timedelta(df.Hour, unit='h')
df['TimeStamp'] = df.DateTime.values.astype(np.int64) // 10**9
df.drop(['Date','Hour','TimeStamp'],axis=1,inplace=True)
df.set_index('DateTime', inplace=True, drop=True)
df['Toronto'].plot()
plt.ylabel('Demand');
df.head()


# 1. **Checking stationarity and performing STL decomposition ([Seasonal and Trend decomposition using Loess](https://otexts.org/fpp2/stl.html))**

# In[ ]:


sm.tsa.seasonal_decompose(deaths['num_deaths']).plot()
print("Dickey-Fuller criterion: p=%f" % 
      sm.tsa.stattools.adfuller(deaths['num_deaths'])[1])


# In[ ]:


sm.tsa.seasonal_decompose(df['Toronto']).plot()
print("Dickey-Fuller criterion: p=%f" % 
      sm.tsa.stattools.adfuller(df['Toronto'])[1])


# ### Stationarity

# Dickey-Fuller criteriom does not reject the non-stationarity null-hypothesis, but we still see a trend. Let's perform seasonal differentiation, then check again stationarity and perform STL decomposition:

# In[ ]:


deaths['num_deaths_diff'] = deaths['num_deaths'] -                             deaths['num_deaths'].shift(12)
sm.tsa.seasonal_decompose(deaths['num_deaths_diff'][12:]).plot()
print("Dickey-Fuller criterion: p=%f" % 
      sm.tsa.stattools.adfuller(deaths['num_deaths_diff'][12:])[1])


# Dickey-Fuller criteriom does now rejects the non-stationarity null-hypothesis, but we still see a trend. Let's now perform one more differentiation.

# In[ ]:


deaths['num_deaths_diff2'] = deaths['num_deaths_diff'] -                              deaths['num_deaths_diff'].shift(1)
sm.tsa.seasonal_decompose(deaths['num_deaths_diff2'][13:]).plot()
print("Dickey-Fuller criterion: p=%f" % sm.tsa.stattools.adfuller(deaths['num_deaths_diff2'][13:])[1])


# Non-stationarity hypothesis is now rejected, and graphs look all right, no trend anymore.

# ## Model selection

# Let's build ACF and PACF for our time series (seems that there's a bug in PACF, it can't be >1):

# In[ ]:


plt.figure(figsize=(12, 8))
ax = plt.subplot(211)
sm.graphics.tsa.plot_acf(deaths['num_deaths_diff2'][13:].values.squeeze(), 
                         lags=30, ax=ax)
ax = plt.subplot(212)
sm.graphics.tsa.plot_pacf(deaths['num_deaths_diff2'][13:].values.squeeze(), 
                          lags=30, ax=ax);


# Initial values: Q=2, q=1, P=2, p=2.
# Setting these is not obligatory, but if we do so, we'll perform less computations tuning hyperparams.

# In[ ]:


ps = range(0, 3)
d=1
qs = range(0, 1)
Ps = range(0, 3)
D=1
Qs = range(0, 3)


# In[ ]:


parameters = product(ps, qs, Ps, Qs)
parameters_list = list(parameters)
len(parameters_list)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'results = []\nbest_aic = float("inf")\n\nwarnings.filterwarnings(\'ignore\')\n\nfor param in parameters_list:\n    #try except is needed because some parameter combinations are not valid\n    try:\n        model=sm.tsa.statespace.SARIMAX(deaths[\'num_deaths\'], order=(param[0], d, param[1]), \n                                        seasonal_order=(param[2], D, \n                                                        param[3], 12)).fit(disp=-1)\n    except ValueError:\n        print(\'wrong parameters:\', param)\n        continue\n    aic = model.aic\n    # save best model, it\'s AIC and params\n    if aic < best_aic:\n        best_model = model\n        best_aic = aic\n        best_param = param\n    results.append([param, model.aic])\n    \nwarnings.filterwarnings(\'default\')')


# In[ ]:


result_table = pd.DataFrame(results)
result_table.columns = ['parameters', 'aic']
print(result_table.sort_values(by = 'aic', ascending=True).head())


# Best model:

# In[ ]:


print(best_model.summary())


# Its residuals:

# In[ ]:


plt.subplot(211)
best_model.resid[13:].plot()
plt.ylabel(u'Residuals')

ax = plt.subplot(212)
sm.graphics.tsa.plot_acf(best_model.resid[13:].values.squeeze(), lags=48, ax=ax)

print("Student's criterion: p=%f" % stats.ttest_1samp(best_model.resid[13:], 0)[1])
print("Dickey-Fuller criterion: p=%f" % sm.tsa.stattools.adfuller(best_model.resid[13:])[1])


# Residuals are not biased (confirmed by the Student's criterion), are stationary (confirmed by the Dickey-Fuller criterion) and not auto-correlated (confirmed by the Ljung-Box criterion and correlogram).
# Let's see how well the model fits data:

# In[ ]:


plt.figure(figsize=(8, 6))
deaths['model'] = best_model.fittedvalues
deaths['num_deaths'].plot(label='actual')
deaths['model'][13:].plot(color='r', label='forecast')
plt.ylabel('Accidental deaths')
plt.legend();


# ### Forecast

# In[ ]:


from dateutil.relativedelta import relativedelta
deaths2 = deaths[['num_deaths']]
date_list = [pd.datetime.strptime("1979-01-01", "%Y-%m-%d") + relativedelta(months=x) for x in range(0,24)]
future = pd.DataFrame(index=date_list, columns=deaths2.columns)
deaths2 = pd.concat([deaths2, future])
deaths2['forecast'] = best_model.predict(start=72, end=100)

deaths2['num_deaths'].plot(color='b', label='actual')
deaths2['forecast'].plot(color='r', label='forecast')
plt.ylabel('Accidental deaths')
plt.legend();


# In[ ]:




