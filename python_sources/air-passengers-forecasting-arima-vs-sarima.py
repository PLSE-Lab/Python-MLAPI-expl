#!/usr/bin/env python
# coding: utf-8

# # ARIMA vs SARIMA 

# ## IMPORT THE NECESSARY LIBRARIES

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from datetime import datetime
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from itertools import combinations

from datetime import datetime
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA as ARIMA
import statsmodels.api as sm
import statsmodels.tsa.api as smt
pd.options.display.float_format = '{:.2f}'.format


# ## IMPORT THE DATASET

# In[ ]:


data = pd.read_csv('../input/air-passengers/AirPassengers.csv')
data.head()


# ## CHECK FOR MISSING VALUES AND BASIC INFO

# In[ ]:


data.isnull().sum()


# In[ ]:


data.info()


# - NO MISSING VALUES
# - CONVERT THE MONTH COLUMN TO DATETIME DATATYPE AND ASSIGN IT AS INDEX 

# In[ ]:


data['Date'] = pd.to_datetime(data['Month'])
data = data.drop(columns = 'Month')
data = data.set_index('Date')
data = data.rename(columns = {'#Passengers':'Passengers'})
data.head()


# ## FUNCTIONS FOR TIMESERIES ANALYSIS

# In[ ]:


def test_stationarity(timeseries):
    #Determing rolling statistics
    MA = timeseries.rolling(window=12).mean()
    MSTD = timeseries.rolling(window=12).std()

    #Plot rolling statistics:
    plt.figure(figsize=(15,5))
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(MA, color='red', label='Rolling Mean')
    std = plt.plot(MSTD, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)

    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)


# In[ ]:


def tsplot(y, lags=None, figsize=(12, 7), style='bmh'):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
        
    with plt.style.context(style):    
        fig = plt.figure(figsize=figsize)
        layout = (2, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        
        y.plot(ax=ts_ax)
        p_value = sm.tsa.stattools.adfuller(y)[1]
        ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_value))
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
        plt.tight_layout()


# ### DATA 

# In[ ]:


test_stationarity(data['Passengers'])


# In[ ]:


dec = sm.tsa.seasonal_decompose(data['Passengers'],period = 12).plot()
plt.show()


# In[ ]:


sns.distplot(data['Passengers'])


# - DATA IS NOT STATIONARY AS THE TEST STATISTIC VALUE IS MORE THAN ANY OF THE CRITICAL VALUE
# - ALSO THE P-Value IS NOT LESS THAN 0.05
# - DATA HAS AN INCREASING TREND
# - DATA IS ALSO SEASONAL WITH A PATTERN OF 1 YEAR

# ### LOG DATA

# In[ ]:


log_data = np.log(data)
log_data.head()


# In[ ]:


test_stationarity(log_data['Passengers'])


# In[ ]:


sns.distplot(log_data['Passengers'])


# - LOG DATA ALSO HAS THE SAME ATTRIBUTES AS THAT OF DATA
# - ONLY THE DATA DISTRIBUTION IS SLIGHTLY BETTER THAN PREVIOUS

# ## DIFFERENCING

# 1] DATA

# In[ ]:


data_diff = data['Passengers'].diff()
data_diff = data_diff.dropna()
dec = sm.tsa.seasonal_decompose(data_diff,period = 12).plot()
plt.show()


# In[ ]:


test_stationarity(data_diff)


# - TREND HAS DIED DOWN AND IS CONSTANT
# - TEST STATISTIC < CRITICAL VALUE(10%) --> DATA IS 90% SURELY STATIONARY
# - P-Value = 0.05
# - ROLLING IS ALSO CONSTANT
# - HENCE DATA IS STATIONARY
# - HOWEVER SEASONALITY IS STILL PRESENT

# 2] LOG DATA

# In[ ]:


log_data_diff = log_data['Passengers'].diff()
log_data_diff = log_data_diff.dropna()
dec = sm.tsa.seasonal_decompose(log_data_diff,period = 12)
dec.plot()
plt.show()


# In[ ]:


test_stationarity(log_data_diff)


# - TREND HAS DIED DOWN AND IS CONSTANT
# - TEST STATISTIC < CRITICAL VALUE(10%) --> DATA IS 90% SURELY STATIONARY
# - P-Value = 0.05
# - ROLLING IS ALSO CONSTANT
# - HENCE DATA IS STATIONARY
# - HOWEVER SEASONALITY IS STILL PRESENT

# ## FROM THE ABOVE TESTS, WE CAN CHOOSE ANY OF THE DATA ABOVE FOR SELECTING THE ORDER OF ARIMA

# # ARIMA [p,d,q]

# 1] DATA 

# In[ ]:


tsplot(data_diff)


# - ARIMA MODEL ORDER [p,d,q]
# - p = PARTIAL AUTOCORRELATION PLOT = LAG VALUE AT WHICH THE LINE TOUCHES THE CONFIDENCE INTERVAL FIRST
# - d = DIFFERENCING ORDER
# - q = AUTOCORRELATION PLOT =  LAG VALUE AT WHICH THE LINE TOUCHES THE CONFIDENCE INTERVAL FIRST

# - FOR OUR MODEL :- 
# - p = [1-2]
# - d = 1
# - q = [1-2]
# - SELECT THE ORDER OF ARIMA MODEL WITH THE LOWEST AIC VALUE

# In[ ]:


model = ARIMA(data['Passengers'],order = (2,1,2))
model_fit = model.fit()
print(model_fit.summary())


# In[ ]:


data['FORECAST'] = model_fit.predict(start = 120,end = 144,dynamic = True)
data[['Passengers','FORECAST']].plot(figsize = (10,6))


# In[ ]:


exp = [data.iloc[i,0] for i in range(120,len(data))]
pred = [data.iloc[i,1] for i in range(120,len(data))]
data = data.drop(columns = 'FORECAST')
print(mean_absolute_error(exp,pred))


# - THE PREDICTION PLOTS ARE NOT GOOD AT ALL 
# - THE MEAN ABSOLUTE ERROR VALUE IS ALSO HIGH
# - THIS IS BECAUSE OF THE ISSUE OF SEASONALITY
# - HENCE WE REJECT ARIMA MODEL AND MOVE ONTO SARIMA WHICH HANDLES SEASONALITY

# # SARIMA [(p,d,q)x(P,D,Q,s)]

# 1] DATA

# In[ ]:


data_diff_seas = data_diff.diff(12)
data_diff_seas = data_diff_seas.dropna()
dec = sm.tsa.seasonal_decompose(data_diff_seas,period = 12)
dec.plot()
plt.show()


# - SEASONAL DIFFERENCE WITH A SEASONAL PERIOD(s) OF 12
# - SINCE OUR DATA IS MONTHLY DATA AND FROM THE PLOTS,WE OBSERVE THAT A YEARLY PATTERN IS PRESENT
# - WE USE THIS OPERATION ON THE PREVIOUSLY DIFFERENCED DATA SO THAT WE DO  NOT HAVE TO DEAL WITH TREND & STATIONARITY AGAIN

# In[ ]:


tsplot(data_diff_seas)


# - SARIMA MODEL ORDER [(p,d,q)x(P,D,Q,s)]
# - (p,d,q) = THIS ORDER IS INHERITED FROM OUR ABOVE ARIMA MODEL
# - (P,D,Q,s) = THIS IS ORDER IS SELECTED USING THE SAME TECHNIQUE USED FOR ARIMA
# - s = SEASONAL ORDER = ONLY ADDITIONAL PARAMETER 
# - WE AGAIN SELECT THE MODEL WITH LEAST AIC SCORE

# In[ ]:


model = sm.tsa.statespace.SARIMAX(data['Passengers'],order = (2,1,2),seasonal_order = (1,1,2,12))
results = model.fit()
print(results.summary())


# In[ ]:


data['FORECAST'] = results.predict(start = 120,end = 144,dynamic = True)
data[['Passengers','FORECAST']].plot(figsize = (12,8))


# In[ ]:


exp = [data.iloc[i,0] for i in range(120,len(data))]
pred = [data.iloc[i,1] for i in range(120,len(data))]
data = data.drop(columns = 'FORECAST')
print(mean_absolute_error(exp,pred))


# - PREDICTED PLOTS ARE GREAT 
# - ERROR HAS ALSO REDUCED ALOT
# - HENCE WE ACCEPT THIS MODEL AND FORECAST FOR 2 MORE YEARS

# # FORECASTING

# ### ADD DATES TO OUR DATAFRAME FOR OUR FORECASTING PURPOSE

# In[ ]:


from pandas.tseries.offsets import DateOffset
future_dates = [data.index[-1] + DateOffset(months = x)for x in range(0,25)]
df = pd.DataFrame(index = future_dates[1:],columns = data.columns)


# ## FINAL PLOT

# In[ ]:


forecast = pd.concat([data,df])
forecast['FORECAST'] = results.predict(start = 144,end = 168,dynamic = True)
forecast[['Passengers','FORECAST']].plot(figsize = (12,8))


# # THANK YOU
