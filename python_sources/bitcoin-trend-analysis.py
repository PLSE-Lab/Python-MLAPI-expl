#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd 
import matplotlib.dates as md
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import datetime as dt
import datetime,pytz

from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
from xgboost import plot_importance, plot_tree

from scipy import stats
from scipy.stats import boxcox

import statsmodels.api as sm
from itertools import product

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from numpy.linalg import LinAlgError


# In[ ]:


#define a conversion function for the native timestamps in the csv file
def dateparse (time_in_secs):    
    return pytz.utc.localize(datetime.datetime.fromtimestamp(float(time_in_secs)))


# In[ ]:


data=pd.read_csv('../input/bitcoin-historical-data/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv',parse_dates=[0], date_parser=dateparse)


# In[ ]:


data.head()


# In[ ]:


data.shape


# In[ ]:


data.isnull().sum()


# In[ ]:


data.info()


# In[ ]:


data.describe().T


# In[ ]:


#Preprocessing the data
# First thing is to fix the data for bars where there are no trades. 
# Volume/trades are a single event so fill na's with zeroes for relevant fields
data['Volume_(BTC)'].fillna(value=0, inplace=True)
data['Volume_(Currency)'].fillna(value=0, inplace=True)
data['Weighted_Price'].fillna(value=0, inplace=True)

# next we need to fix the Open,high,low,close data which is a continuous timeseries so forwardfill those values...
data['Open'].fillna(method='ffill', inplace=True)
data['High'].fillna(method='ffill', inplace=True)
data['Low'].fillna(method='ffill', inplace=True)
data['Close'].fillna(method='ffill', inplace=True)

data.head()


# In[ ]:


corr=data.corr()
f,ax = plt.subplots(figsize=(12, 12))
sns.heatmap(corr,annot=True,linewidths=.5, fmt= '.1f',ax=ax)


# In[ ]:


# ScatterPlot  
# x = open, y = close
data.plot(kind='scatter', x='Volume_(BTC)', y='Volume_(Currency)',alpha = 0.5)
plt.xlabel('BTC Volume')            
plt.ylabel('Currency Volume')
plt.title('BTC-Currency Scatter Plot') 
plt.show()


# In[ ]:


#!pip install prophet


# In[ ]:


plt.figure(figsize=(16,4))
data.Open.plot(kind='line', color='r', label='Open', alpha=0.5, linewidth=5, grid=True, linestyle=':')
data.High.plot(color='g', label='High', linewidth=1, alpha=0.5, grid=True, linestyle='-.')
plt.legend(loc='upper right') #legend put label into plot
plt.xlabel('Time')
plt.ylabel('price at the start of the time window')
plt.title('Line plot')
plt.show()


# In[ ]:


#open and high values have same trend


# # XGBoost

# In[ ]:


data = pd.read_csv('../input/bitcoin-historical-data/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv',parse_dates=[0], date_parser=dateparse) 
data['Timestamp'] = data['Timestamp'].dt.tz_localize(None)
data = data.groupby([pd.Grouper(key='Timestamp', freq='H')]).first().reset_index()
data = data.set_index('Timestamp')
data = data[['Weighted_Price']]
data['Weighted_Price'].fillna(method='ffill', inplace=True)


# In[ ]:


data.plot(style='', figsize=(15,5), title='BTC Weighted_Price Price (USD) by Hours')


# In[ ]:


split_date = '25-Jun-2018'
data_train = data.loc[data.index <= split_date].copy()
data_test = data.loc[data.index > split_date].copy()


# In[ ]:


_ = data_test     .rename(columns={'Weighted_Price': 'Test Set'})     .join(data_train.rename(columns={'Weighted_Price': 'Training Set'}), how='outer')     .plot(figsize=(15,5), title='BTC Weighted_Price Price (USD) by Hours', style='')


# In[ ]:


def create_features(df, label=None):
    """
    Creates time series features from datetime index
    """
    df['date'] = df.index
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.weekofyear
    
    X = df[['hour','dayofweek','quarter','month','year',
           'dayofyear','dayofmonth','weekofyear']]
    if label:
        y = df[label]
        return X, y
    return X


# In[ ]:


#data split
X_train, y_train = create_features(data_train, label='Weighted_Price')
X_test, y_test = create_features(data_test, label='Weighted_Price')


# In[ ]:


model =  xgb.XGBRegressor(objective ='reg:linear',min_child_weight=10, booster='gbtree', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 100)
model.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        early_stopping_rounds=50,
       verbose=False) 


# In[ ]:


data_test['Weighted_Price_Prediction'] = model.predict(X_test)
data_all = pd.concat([data_test, data_train], sort=False)


# In[ ]:


_ = data_all[['Weighted_Price','Weighted_Price_Prediction']].plot(figsize=(15, 5))


# In[ ]:


# Plot the forecast with the actuals
f, ax = plt.subplots(1)
f.set_figheight(5)
f.set_figwidth(15)
_ = data_all[['Weighted_Price_Prediction','Weighted_Price']].plot(ax=ax,
                                              style=['-','.'])
ax.set_xbound(lower='08-01-2018', upper='09-01-2018')
ax.set_ylim(0, 10000)
plot = plt.suptitle('August 2018 Forecast vs Actuals')


# In[ ]:


# Plot the forecast with the actuals
f, ax = plt.subplots(1)
f.set_figheight(5)
f.set_figwidth(15)
_ = data_all[['Weighted_Price_Prediction','Weighted_Price']].plot(ax=ax,
                                              style=['-','.'])
ax.set_xbound(lower='08-01-2018', upper='08-08-2018')
ax.set_ylim(0, 10000)
plot = plt.suptitle('First Week of August 2018 Forecast vs Actuals')


# In[ ]:


print("MSE of XGBoost: ",mean_squared_error(y_true=data_test['Weighted_Price'],
                   y_pred=data_test['Weighted_Price_Prediction']))


# In[ ]:


print("MAE of XGBoost: ",mean_absolute_error(y_true=data_test['Weighted_Price'],
                   y_pred=data_test['Weighted_Price_Prediction']))


# # ARIMA Model

# In[ ]:


data = pd.read_csv('../input/bitcoin-historical-data/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv',parse_dates=[0], date_parser=dateparse) 


# In[ ]:


data['Volume_(BTC)'].fillna(value=0, inplace=True)
data['Volume_(Currency)'].fillna(value=0, inplace=True)
data['Weighted_Price'].fillna(value=0, inplace=True)

# next we need to fix the OHLC (open high low close) data which is a continuous timeseries so
# lets fill forwards those values...
data['Open'].fillna(method='ffill', inplace=True)
data['High'].fillna(method='ffill', inplace=True)
data['Low'].fillna(method='ffill', inplace=True)
data['Close'].fillna(method='ffill', inplace=True)


# In[ ]:


#seasonality by monthly
decomposition = sm.tsa.seasonal_decompose(data.Weighted_Price,freq=3)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

fig = plt.figure(figsize=(20,8))

plt.subplot(411)
plt.plot(data.Weighted_Price, label='Original')
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

fig.suptitle('Decomposition of Prices Data')
plt.show()


# # Trial

# In[ ]:


data = pd.read_csv('../input/bitcoin-historical-data/bitstampUSD_1-min_data_2012-01-01_to_2019-08-12.csv',parse_dates=[0], date_parser=dateparse) 


# In[ ]:


data.head()


# In[ ]:


data.Timestamp = pd.to_datetime(data.Timestamp, unit = 's')
data.index = data.Timestamp
data = data.resample('M').mean()


# In[ ]:


data.head()


# In[ ]:


data.isnull().sum()


# In[ ]:


price=data["Weighted_Price"]
plt.figure(figsize = (14,6))
sns.lineplot(x = data.index, y = price)


# using ARIMA model for forecasting,
# before that checking our TS is having stationary or not? ,by Seasonal Trend Decomposition and DF Test

# In[ ]:


def decompose(series):
    plt.figure(figsize = (14,7))
    seasonal_decompose(series).plot()
    plt.show()
    
def DFTest(series):
    testdf = adfuller(series)
    print("DF test p-value : %.16f" %testdf[1] )
    
    
def plots(series):
    plt.figure(figsize = (10,6))
    sns.lineplot(data = series, color = 'blue', label = 'observed line plot')
    sns.lineplot(data = series.rolling(window = 12).mean(), color = 'green', label = 'rolling mean, window -12')
    sns.lineplot(data = series.rolling(window = 12).std(), color = 'black', label = 'std deviation, window -12')


# In[ ]:


print("DF Test->")
#running tests
DFTest(price)
decompose(price)
plots(price)


# With the visualisation and the p value of the DF test (>0.05) we can state that the series is not stationary and hence, we can't apply ARIMA model to it just yet. we will now transform our TS to fit ARIMA

# In[ ]:


#1. Log Transformation


# In[ ]:


prices_log = np.log(price)

#running tests
DFTest(prices_log)
decompose(prices_log)
plots(prices_log)


# In[ ]:


#prices_log with regular shift transform
prices_log_r = prices_log - prices_log.shift(1)
prices_log_r.dropna(inplace = True)

DFTest(prices_log_r)
decompose(prices_log_r)
plots(prices_log_r)


# In[ ]:


#2. Box Cox Transformation


# In[ ]:


prices_box_cox_, lambda_ = boxcox(price)
prices_box_cox = pd.Series(data = prices_box_cox_, index = data.index) #decompose functions requires a pandas object that has a timestamp index.

decompose(prices_box_cox) 
DFTest(prices_box_cox)
print('lambda value:', lambda_)
plots(prices_box_cox)


# In[ ]:


#Regular Time shift added to Box Cox transformation


# In[ ]:


prices_box_cox_r = prices_box_cox - prices_box_cox.shift(1)
prices_box_cox_r.dropna(inplace = True)

decompose(prices_box_cox_r) 
DFTest(prices_box_cox_r)
plots(prices_box_cox_r)


# ACF and PACF

# In[ ]:


plt.figure(figsize = (14,7)) 
a = acf(prices_log_r)
p = pacf(prices_log_r)

plt.subplot(221)
sns.lineplot(data = a)
plt.axhline(y=0, linestyle='--', color='gray')

plt.subplot(222)
sns.lineplot(data = p)
plt.axhline(y=0, linestyle='--', color='gray')


# In[ ]:


#We infer from the plot that The ACF and PACF gets close to zero while lag approaches 1.


# In[ ]:


data.head()


# In[ ]:


a = [[1,2,3], [1],[1,2,3]]
params = list(product(*a))

results = []   
min_aic = float('inf')
best_param = []

# checking different set of params for best fit
for param in params:
    try:
        model = ARIMA(prices_log, order = param).fit(disp = -1)
    except LinAlgError:
        print('Rejected Parameters:', param)
        continue
    except ValueError:
        print('Rejected Parameters:', param)
        continue
    if(min_aic > model.aic):
        min_aic = model.aic
        best_param = param
        best_model = model
        
    results.append([param, model.aic])

print(best_param,min_aic)
print(results)

print(best_model.fittedvalues)

plt.figure(figsize=(16,8))
sns.lineplot(data = prices_log_r, color = 'blue')
sns.lineplot(data = best_model.fittedvalues, color = 'red') 


# In[ ]:


fitted_values = best_model.fittedvalues
fitted_values = fitted_values.cumsum()

fitted_values = fitted_values + prices_log[0]

final_values = np.exp(fitted_values)

d = {'prices' : price, 'prices_log' : prices_log, 'price_log_r' : prices_log_r, 'fitted_values' : fitted_values, 'final_values' : final_values}
summaryDF = pd.DataFrame(data = d)
sns.lineplot(data = summaryDF['prices'], color = 'green')
sns.lineplot(data = summaryDF['final_values'], color = 'red')


# In[ ]:


predicted_values = np.exp((best_model.predict(start = 1, end = 99).cumsum()) + prices_log[0])
sns.lineplot(data = price, label  = 'recorded')
sns.lineplot(data = predicted_values, label = 'predicted')

