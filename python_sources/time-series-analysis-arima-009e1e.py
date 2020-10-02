#!/usr/bin/env python
# coding: utf-8

# **Basic Time Series Analysis on Cryptocurrency Historical Prices**

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


# Import Packages
import warnings
warnings.filterwarnings('ignore')
import numpy as np 
import pandas as pd 
from pandas import DataFrame
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
import statsmodels.tsa.api as smt
import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('../input/bitcoin_price.csv', parse_dates=['Date'])
df.head(3)


# In[ ]:


print (df.describe())


# In[ ]:


print (df.dtypes)


# In[ ]:


df1 = df[['Date','Close']]
df1.head(3)


# In[ ]:


# Setting the Date as Index
df_ts = df1.set_index('Date')
df_ts.sort_index(inplace=True)
print (type(df_ts))
print (df_ts.head(3))
print (df_ts.tail(3))


# In[ ]:


#plot
df_ts.plot()


# In[ ]:


# Dickey Fuller Test Function
def test_stationarity(timeseries):
    # Perform Dickey-Fuller test:
    from statsmodels.tsa.stattools import adfuller
    print('Results of Dickey-Fuller Test:')
       
    dftest = adfuller(timeseries, autolag='AIC')
    
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#lags Used', 'Number of Observations Used'])
    
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    
    print(dfoutput)


# In[ ]:


# Stationarity with Dickey Fuller Test 
# Convert the DF to series first
ts = df_ts['Close']
test_stationarity(ts)


# **Conclusion**
# 
# The Test Statistics value is Much higher than critical value. So we can't reject the Null Hypothesis.
# 
# Hence from the plot the Time series is Non-Stationary

# In[ ]:


# plot the 12-Month Moving Rolling Mean, Variance 
# Rolling Statistics
rolmean = ts.rolling(window=12).mean()
rolvar = ts.rolling(window=12).std()

plt.plot(ts, label='Original')
plt.plot(rolmean, label='Rolling Mean')
plt.plot(rolvar, label='Rolling Standard Variance')
plt.legend(loc='best')
plt.title('Rolling Mean & Standard Deviation')
plt.show(block=False)


# In[ ]:


# vanila decomposition to see any trend seasonality in the ts
decomposition = sm.tsa.seasonal_decompose(ts, model='multiplicative')

fig = decomposition.plot()
fig.set_figwidth(12)
fig.set_figheight(8)
fig.suptitle('Decomposition of multiplicative time series')
plt.show()


# **Conslusion**
# 
# Seasonal graph is too stacked . This was obvious by the way as we are analyzing daily data so trying out a Monthly approach

# In[ ]:


#data by Month and analyze again
df_ts_m = df_ts.resample('M').mean()
print (type(df_ts_m))
print (df_ts_m.head(3))


# In[ ]:


tsm = df_ts_m['Close']
print (type(tsm))


# In[ ]:


# Stationarity Check
test_stationarity(tsm)


# In[ ]:


# A quick vanila decomposition 
decomposition = sm.tsa.seasonal_decompose(tsm, model='multiplicative')

fig = decomposition.plot()
fig.set_figwidth(12)
fig.set_figheight(8)
fig.suptitle('Decomposition of multiplicative time series')
plt.show()


# **Conclusion**
# 
# Much more clear seasonality graph

# In[ ]:


# "tsm" Stationary

tsmlog = np.log10(tsm)
tsmlog.dropna(inplace=True)

tsmlogdiff = tsmlog.diff(periods=1)
tsmlogdiff.dropna(inplace=True)
# Stationarity Check
test_stationarity(tsmlogdiff)


# **Conclusion**
# 
# The time series is Stationary now. We can use it now in Forecasting Techniques like ARIMA

# In[ ]:


# plot ACF & PACF graphs to visualize AR & MA components
fig, axes = plt.subplots(1, 2)
fig.set_figwidth(12)
fig.set_figheight(4)
smt.graphics.plot_acf(tsmlogdiff, lags=30, ax=axes[0], alpha=0.5)
smt.graphics.plot_pacf(tsmlogdiff, lags=30, ax=axes[1], alpha=0.5)
plt.tight_layout()

