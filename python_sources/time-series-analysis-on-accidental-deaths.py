#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pyplot import rcParams
rcParams['figure.figsize'] = 15,10


# ## Parsing the data as a datatime object

# In[ ]:


dateparser = lambda dates: pd.datetime.strptime(dates,'%Y-%m')
data = pd.read_csv('../input/accidental-deaths-in-usa-monthly.csv',header = 0, names = ['Month', 'Deaths'], parse_dates = ['Month'],index_col='Month',date_parser=dateparser)
data = pd.DataFrame(data)
data.head(10)


# In[ ]:


plt.plot(data)


# ## Converting Dataframe into a Series

# In[ ]:


ts = data["Deaths"]
ts.head(10)


# In[ ]:


plt.plot(ts)


# ### A method to test the stationarity of the time series

# In[ ]:


from statsmodels.tsa.stattools import adfuller
def test_stationarity(ts):
    rolling_mean = ts.rolling(12).mean()
    rolling_std = ts.rolling(12).std()
    
    plt.plot(ts,color="blue", label="Original")
    plt.plot(rolling_mean,color="red", label="Rolling Mean")
    plt.plot(rolling_std,color="green", label = "Rolling Std")
    plt.legend(loc="best")
    plt.title('Rolling Mean and Standard Deviation')
    plt.show(block=False)
    
    print("Results of Dickey Fuller Test")
    dfresult = adfuller(ts,autolag = "AIC")
    dfoutput=pd.Series(dfresult[0:4],index=['Test Statistics','p-value','#lags used','#obv used'])
    for key,value in dfresult[4].items():
        dfoutput["Critical Value (%s)"%key] = value
    print(dfoutput)


# In[ ]:


test_stationarity(ts)


# ## Logarithmic Transform to smoothen the data

# In[ ]:


ts_log=np.log(ts)
plt.plot(ts_log)


# ## Second-order differencing

# In[ ]:


ts_log_diff = ts_log - ts_log.shift()
plt.plot(ts_log_diff)


# In[ ]:


ts_log_diff.dropna(inplace=True)
test_stationarity(ts_log_diff)


# In[ ]:


ts_log_diff = ts_log_diff - ts_log_diff.shift()
plt.plot(ts_log_diff)


# In[ ]:


ts_log_diff.dropna(inplace=True)
test_stationarity(ts_log_diff)


# Test statistic is less tha 1% critical value - 99% confidence

# ## PACF and ACF graphs

# In[ ]:


from statsmodels.tsa.stattools import acf,pacf

lag_acf=acf(ts_log_diff,nlags=20)
lag_pacf=pacf(ts_log_diff,nlags=20,method="ols")

plt.subplot("121")
plt.plot(lag_acf)
plt.axhline(y=0,color="gray",linestyle="--")
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),color="gray",linestyle="--")
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),color="gray",linestyle="--")
plt.xticks(np.arange(0,21,1.0))
plt.title("Autocorrelation Function")

plt.subplot("122")
plt.plot(lag_pacf)
plt.axhline(y=0,color="gray",linestyle="--")
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),color="gray",linestyle="--")
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),color="gray",linestyle="--")
plt.xticks(np.arange(0,21,1.0))
plt.title("Partial Autocorrelation Function")

plt.tight_layout()


# ### Determining the values of p and q from the ACF and PACF graphs. d is the order of differencing = 2

# In[ ]:


p,d,q = 1,2,1


# ## Fitting ARIMA model

# In[ ]:


from statsmodels.tsa.arima_model import ARIMA

model = ARIMA(ts_log, order=(p,d,q))
results_ARIMA = model.fit(disp=-1)
plt.plot(ts_log_diff)
plt.plot(results_ARIMA.fittedvalues,color="red")
plt.title("RSS = %.4f"% sum((results_ARIMA.fittedvalues-ts_log_diff)**2))
#plt.title('RMSE: %.4f'% np.sqrt(((results_ARIMA.fittedvalues-ts_log_diff)**2).mean()))


# In[ ]:


predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
print(predictions_ARIMA_diff.head(10))


# In[ ]:


predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
print(predictions_ARIMA_diff_cumsum.head(10))


# In[ ]:


predictions_ARIMA_log = pd.Series(ts_log.iloc[0], index=ts_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
print(predictions_ARIMA_log.head(10))


# In[ ]:


predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(ts)
plt.plot(predictions_ARIMA)
plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-ts)**2)/len(ts)))


# In[ ]:




