#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


pwd


# In[ ]:


df=pd.read_csv('../input/air-passengers/AirPassengers.csv')


# In[ ]:


df.head()


# In[ ]:


df.tail()


# In[ ]:


df.shape


# In[ ]:


df.info()


# In[ ]:


df.isnull().sum()


# # DATASET LOADING

# In[ ]:


df=pd.read_csv('../input/air-passengers/AirPassengers.csv',index_col='Month',parse_dates=True)


# In[ ]:


df.head()


# In[ ]:


df.index


# In[ ]:


df.index.freq='MS'


# In[ ]:


df.index


# In[ ]:


df.plot(figsize=(8,5))


# ### TIME RESAMPLING

# In[ ]:


df['#Passengers'].resample('Y').mean().plot(kind='bar')


# In[ ]:


df['#Passengers'].iloc[:12].mean()


# ### ROLLING WINDOWS

# In[ ]:


df['#Passengers'].plot(figsize=(8,5),legend=True)
df['#Passengers'].rolling(window=7).mean().plot(legend=True)


# In[ ]:


df['#Passengers'].plot(figsize=(8,5),legend=True)
df['#Passengers'].rolling(window=14).mean().plot(legend=True)


# ### EXPANDING WINDOWS

# In[ ]:


df['#Passengers'].plot(figsize=(8,5),legend=True)
df['#Passengers'].expanding().mean().plot(legend=True)


#  ### Hodrick-Prescott Filter

# In[ ]:


from statsmodels.tsa.filters.hp_filter import hpfilter


# In[ ]:


pas_cycle,pas_trend=hpfilter(df['#Passengers'],lamb=1600)


# In[ ]:


df1=df.copy()


# In[ ]:


df1['trend']=pas_trend


# In[ ]:


df1[['#Passengers','trend']].plot(figsize=(12,10))


# ### ETS DECOMPOSITION

# In[ ]:


from statsmodels.tsa.seasonal import seasonal_decompose
from pylab import rcParams
rcParams['figure.figsize']=12,5


# In[ ]:


result=seasonal_decompose(df['#Passengers'],model='additive')


# In[ ]:


result.plot();


# In[ ]:


result=seasonal_decompose(df['#Passengers'],model='multiplicative')


# In[ ]:


result.plot();


# ### EWMA MODELS

# In[ ]:


df1['6 month-SMA']=df1['#Passengers'].rolling(window=6).mean()
df1['12 month-SMA']=df1['#Passengers'].rolling(window=12).mean()
df1['EWMA-6']=df1['#Passengers'].ewm(span=6,adjust=False).mean()
df1['EWMA-12']=df1['#Passengers'].ewm(span=12,adjust=False).mean()


# In[ ]:


df1[['#Passengers','6 month-SMA','12 month-SMA','EWMA-6','EWMA-12']]['1959-01-01':'1961-01-01'].plot(figsize=(12,10))


# ### HOLT-WINTERS METHOD

# #### Simple Exponential Smoothing

# In[ ]:


from statsmodels.tsa.holtwinters import SimpleExpSmoothing


# In[ ]:


span=12
alpha=2/(span+1)


# #### method 1

# In[ ]:


df1['EWMA12']=df1['#Passengers'].ewm(alpha=alpha,adjust=False).mean()
df1.head()


# ##### method 2 :using statsmodels

# In[ ]:


model=SimpleExpSmoothing(df['#Passengers']).fit(smoothing_level=alpha,optimized=False)
#fitted_model=model.fit(df['#Passengers'])
df1['SES12']=model.fittedvalues.shift(-1)
#model.


# In[ ]:


df1.head()


# ### DoubleExpSmoothing

# In[ ]:


from statsmodels.tsa.holtwinters import ExponentialSmoothing


# In[ ]:


df1['DES_add_12']=ExponentialSmoothing(df1['#Passengers'],trend='add').fit().fittedvalues.shift(-1)


# In[ ]:


df1[['#Passengers','SES12','DES_add_12']].plot(figsize=(12,10))


# ### Triple Exponential Smoothing

# In[ ]:


df1['TES_mul_12']=ExponentialSmoothing(df1['#Passengers'],trend='mul',seasonal='mul',seasonal_periods=12).fit().fittedvalues


# In[ ]:


df1[['#Passengers','SES12','TES_mul_12']].plot(figsize=(12,10))


# In[ ]:


# DES is performing better


# ## GENERAL FORECASTING METHODS:

# ## using holt-winters method to forecast for the future 

#  ### TES method

# In[ ]:


train=df.iloc[:109]
test=df.iloc[108:]


# In[ ]:


from statsmodels.tsa.holtwinters import ExponentialSmoothing


# In[ ]:


fitted_model=ExponentialSmoothing(train['#Passengers'],trend='mul',seasonal='mul').fit()


# In[ ]:


test_pred=fitted_model.forecast(36)


# In[ ]:


test_pred.tail()


# In[ ]:


train['#Passengers'].plot(legend=True,figsize=(12,10),label='TRAIN')
test['#Passengers'].plot(legend=True,label='TEST')
test_pred.plot(legend=True,label='TES Predictions')
plt.show()


# In[ ]:


#using holt-winters (DES) method to forecast for the future 


# In[ ]:


fitted_modeld=ExponentialSmoothing(train['#Passengers'],trend='add',seasonal='add').fit() #.fittedvalues.shift(-1)


# In[ ]:


test_predd=fitted_modeld.forecast(36)


# In[ ]:


train['#Passengers'].plot(legend=True,figsize=(12,10),label='TRAIN')
test['#Passengers'].plot(legend=True,label='TEST')
test_predd.plot(legend=True,label='DES Predictions')
plt.show()


# In[ ]:


# EVALUATION 


# In[ ]:


from sklearn.metrics import mean_squared_error


# In[ ]:


mean_squared_error(test['#Passengers'],test_pred) #TES


# In[ ]:


mean_squared_error(test['#Passengers'],test_predd) #DES


# In[ ]:


# DES method worked well!


# In[ ]:


# Now forecasting into future


# In[ ]:


final_modeld=ExponentialSmoothing(df['#Passengers'],trend='add',seasonal='add').fit() #.fittedvalues.shift(-1)


# In[ ]:


final_predd=final_modeld.forecast(36)


# In[ ]:


train['#Passengers'].plot(legend=True,figsize=(12,10),label='TRAIN')
test['#Passengers'].plot(legend=True,label='TEST')
final_predd.plot(legend=True,label='DES Predictions')
plt.show()


# ## Checking for stationarity :

# In[ ]:


# Time series data is stationary when there is no trend and seasonality
#method 1: by differencing
#method 2: by dickey fuller test
    


# In[ ]:


from statsmodels.tsa.statespace.tools import diff


# In[ ]:


df1['d2']=diff(df['#Passengers'],k_diff=2) #.plot()


# In[ ]:


from statsmodels.tsa.stattools import adfuller


# In[ ]:


def adf_test(series,title=''):
    print(f'Augmentted Dickey Fuller Test : {title}')
    result=adfuller(series.dropna(),autolag='AIC')
    labels=['ADF test statistic','p-value','# lags used','# observations']
    out=pd.Series(result[0:4],index=labels)
    for key,val in result[4].items():
        out[f'critical value({key})']=val
    print(out.to_string())
    
    if result[1]<=0.05:
        print('strong evidence against the null hypothesis')
        print('reject the null hypothesis')
        print('data has no unit roots and is stationary')

    else:
        print('weak evidence against the null hypothesis')
        print('fail to reject the null hypothesis')
        print('data has a unit root and is non-stationary')

        


# In[ ]:


adf_test(df1['#Passengers'])


# In[ ]:


adf_test(df1['d2'])


# ## ACF & PACF 

# In[ ]:


# ACF PLOT:


# In[ ]:


from statsmodels.graphics.tsaplots import plot_acf,plot_pacf


# In[ ]:


title='Autocorrelation: No.of Air Passengers'
lags=40
plot_acf(df['#Passengers'],title=title,lags=lags);


# In[ ]:


# This plot indicates non stationary data as there are large number of lags before ACF Values drop off.


# In[ ]:


# PACF plot:


# In[ ]:


# PACF works best with stationary data.Hence apply differencing and make data stationary


# In[ ]:


df1['d1']=diff(df1['#Passengers'],k_diff=1)


# In[ ]:


plot_pacf(df1['d1'].dropna(),title=title,lags=np.arange(lags));


# ## AUTOREGRESSION with Statsmodels

# In[ ]:


from statsmodels.tsa.ar_model import AR,ARResults


# In[ ]:


model=AR(train['#Passengers'])
AR1fit=model.fit(maxlag=1)


# In[ ]:


AR1fit.params


# In[ ]:


start=len(train)
end=len(train)+len(test)-1


# In[ ]:


pred1=AR1fit.predict(start=start,end=end).rename('AR1 Predictions')


# In[ ]:


test.plot(figsize=(8,5),legend=True)
pred1.plot(legend=True)


# In[ ]:


model=AR(df1['#Passengers'])
AR2fit=model.fit(maxlag=2)
pred2=AR2fit.predict(start,end).rename('AR2 Predictions')


# In[ ]:


test.plot(figsize=(8,5),legend=True)
pred2.plot(legend=True)


# In[ ]:


model=AR(df1['#Passengers'])
ARfit=model.fit(ic='t-stat')  # we can choose order p for no.of lags using statsmodels


# In[ ]:


ARfit.params  # 13 lags


# In[ ]:


pred13=ARfit.predict(start,end).rename('AR13 Predictions')


# In[ ]:


test.plot(figsize=(8,5),legend=True)
pred1.plot(legend=True)
pred2.plot(legend=True)
pred13.plot(legend=True)


# In[ ]:


labels=['AR1','AR2','AR13']
preds=[pred1,pred2,pred13]


# In[ ]:


import numpy as np
for i in range(3):
    error=np.sqrt(mean_squared_error(test['#Passengers'],preds[i]))
    print(f'{labels[i]} MSE was :{error}')


# In[ ]:


# AR13 performed well lets build final model and forecast


# In[ ]:


model=AR(df1['#Passengers'])
ARfit=model.fit(maxlag=None)
forecasted_values=ARfit.predict(start=len(df1),end=len(df1)+36).rename('Forecast')


# In[ ]:


df1['#Passengers'].plot(legend=True,figsize=(12,10))
forecasted_values.plot(legend=True)


# ## AUTO-ARIMA

# In[ ]:


#!pip install pmdarima


# In[ ]:


from pmdarima import auto_arima
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


stepwise_fit=auto_arima(df1['#Passengers'],start_p=0,start_q=0,max_p=6,max_q=3,seasonal=True,trace=True,m=12)


# In[ ]:


stepwise_fit.summary()


# In[ ]:


# building model:
from statsmodels.tsa.arima_model import ARMA,ARIMA,ARMAResults,ARIMAResults
model=ARIMA(train['#Passengers'],order=(1,1,1))
results=model.fit()
results.summary()


# In[ ]:


# predictions
predictions=results.predict(start,end,typ='levels').rename('ARIMA(1,1,1) Predictions')


# In[ ]:


test['#Passengers'].plot(legend=True)
predictions.plot(legend=True)


# In[ ]:


from statsmodels.tools.eval_measures import rmse


# In[ ]:


error=rmse(test['#Passengers'],predictions)


# In[ ]:


error


# In[ ]:


test['#Passengers'].mean()


# In[ ]:


# Forecast into future
model=ARIMA(df1['#Passengers'],order=(1,1,1))
results=model.fit()
fcast=results.predict(start=len(df1),end=len(df1)+36,typ='levels').rename('ARIMA(1,1,1) forecast')


# In[ ]:


df1['#Passengers'].plot(legend=True,figsize=(8,5))
fcast.plot(legend=True)


# # SARIMA

# In[ ]:


from statsmodels.tsa.statespace.sarimax import SARIMAX


# In[ ]:


model=SARIMAX(train['#Passengers'],order=(1,1,1),seasonal_order=(1,0,0,12))
results=model.fit()
results.summary()


# In[ ]:


predictions1=results.predict(start,end,typ='levels').rename('SARIMA Predictions')


# In[ ]:


test['#Passengers'].plot(legend=True,figsize=(8,5))
predictions1.plot(legend=True)


# In[ ]:


error1=rmse(test['#Passengers'],predictions1)
error1


# In[ ]:


# Forecast into future
model=SARIMAX(df1['#Passengers'],order=(1,1,1),seasonal_order=(1,0,0,12))
results=model.fit()
fcast=results.predict(start=len(df1),end=len(df1)+36,typ='levels').rename('SARIMA Forecast')


# In[ ]:


df1['#Passengers'].plot(legend=True,figsize=(8,5))
fcast.plot(legend=True)

