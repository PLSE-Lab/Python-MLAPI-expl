#!/usr/bin/env python
# coding: utf-8

# **This notebook tries to take you through a detailed time series analysis using the Bitcoin Historical data for its Price Prediction.  **
# 
# We Will be covering various techniques to make the most out of the data available and try to achieve best results in forecasting values.
# Let's go.
# 

# In[ ]:


import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA

from scipy.stats import boxcox

from itertools import product
from numpy.linalg import LinAlgError


# In[ ]:


df = pd.read_csv('../input/bitstampUSD_1-min_data_2012-01-01_to_2019-03-13.csv')  

#just a few basic steps
print(df.head())
print(df.shape)
print(df.describe())
print(df.isnull().any())


# The timestamp column of the dataset records the price features with a stamp every 60 s, i.e a minute.
# Lets resample our data to have the timestamp with a stamp every month.

# In[ ]:


df.Timestamp = pd.to_datetime(df.Timestamp, unit = 's')
df.index = df.Timestamp
df = df.resample('M').mean()


# Now we have a timestamp index and the stamps are a month apart. We will continue with this time series analysis. We will try to predict monthly prices since the data we are using is also monthly sampled. Let's quickly describe our data once.

# In[ ]:


print(df.describe())
df.isnull().any()
prices = df.Weighted_Price


# we have no missing values in our data. Let's go ahead with plotting our data now.

# In[ ]:


plt.figure(figsize = (14,6))
sns.lineplot(x = df.index, y = prices)


# Let us now use the concepts of time series analysis to make some meaning out of the given time series.
# We define a decompose function that uses seasonal decompose to analyse the componenets of the time series. 
# **We use ARIMA model to analyse our time series first. We know that for it, we need our series to be stationary. So we use the below described techniques to realise if our series is stationary **
# 
# **1.Seasonal Trend Decomposition**
# We use seasonal decomposition to visualise the seasonal and trend components of the time series. We aim to get a residual that is free of trends and seasonality.
# 
# **2.Dicky Fuller test**
# Dicky Fuller test considers the null hypothesis that the time series under consideration is non-stationary. If p-value is sufficiently low ( less than 0.05) while hypothesis testing, then only we reject the null hypothesis and consider the series to be stationary. The DF test provides us with the p-value which we use to determine the stationarity of our series.
# 
# Let us write functions for our above tests.

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
    


# ** Let us run the above tests for our original prices series.**
# 

# In[ ]:


print("DF Test->")
#running tests
DFTest(prices)
decompose(prices)
plots(prices)


# With the visualisation and the p value of the DF test (>0.05) we can state that the series is not stationary and hence, we can't apply ARIMA model to it just yet.
# Now We consider a few transformations for our series, and we go on to check which makes our series suitable for ARIMA modelling.
# 

# **Transformations**
# 
# **1. Log Transformation**

# In[ ]:


prices_log = np.log(prices)

#running tests
DFTest(prices_log)
decompose(prices_log)
plots(prices_log)


# **2. Regular time shift applied to Log transformed prices**

# In[ ]:


#prices_log with regular shift transform
prices_log_r = prices_log - prices_log.shift(1)
prices_log_r.dropna(inplace = True)

DFTest(prices_log_r)
decompose(prices_log_r)
plots(prices_log_r)


# **3.Box_Cox power transform**

# In[ ]:


prices_box_cox_, lambda_ = boxcox(prices)
prices_box_cox = pd.Series(data = prices_box_cox_, index = df.index) #decompose functions requires a pandas object that has a timestamp index.

decompose(prices_box_cox) 
DFTest(prices_box_cox)
print('lambda value:', lambda_)
plots(prices_box_cox)


# **4. Regular time shift applied on Box Cox Transformed prices** 

# In[ ]:


prices_box_cox_r = prices_box_cox - prices_box_cox.shift(1)
prices_box_cox_r.dropna(inplace = True)

decompose(prices_box_cox_r) 
DFTest(prices_box_cox_r)
plots(prices_box_cox_r)


# Gives a nice result on DF test.
# 
# Let us now plot the **ACF and PACF** of the selected transformed function. We get an idea of the parameters to be used for using in ARIMA.
# 
# 

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


# We infer from the plot that The ACF and PACF gets close to zero while lag approaches 1.
# You can learn making sense out of the ACF and PACF plots [here](https://people.duke.edu/~rnau/411arim.htm). 
# 
# As per the plots let us try different values of p and q. D = 1
# 

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


# Now we will try to predict some values and get 

# In[ ]:


fitted_values = best_model.fittedvalues
fitted_values = fitted_values.cumsum()

fitted_values = fitted_values + prices_log[0]

final_values = np.exp(fitted_values)

d = {'prices' : prices, 'prices_log' : prices_log, 'price_log_r' : prices_log_r, 'fitted_values' : fitted_values, 'final_values' : final_values}
summaryDF = pd.DataFrame(data = d)
sns.lineplot(data = summaryDF['prices'], color = 'blue')
sns.lineplot(data = summaryDF['final_values'], color = 'red')


# In[ ]:


predicted_values = np.exp((best_model.predict(start = 1, end = 99).cumsum()) + prices_log[0])
sns.lineplot(data = prices, label  = 'recorded')
sns.lineplot(data = predicted_values, label = 'predicted')

