#!/usr/bin/env python
# coding: utf-8

# # Introduction
#  This is the second part of Time Series Stock Price Predictions. You find the part 1 [here](https://www.kaggle.com/viswanathanc/time-series-stock-price-predictions-part-1).. The First part contains simple models for time series prediction. This notebook contains some classical model for time series predicttions.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
from warnings import filterwarnings
from sklearn.metrics import mean_squared_error as mse
filterwarnings("ignore")


# The data to be used is the closing stock price of Infosys in NSE. EDA of the data was done in the 'Adjustment for Split-up' section in Part 1.

# In[ ]:


df = pd.read_csv("../input/national-stock-exchange-time-series/infy_stock.csv",
                 usecols=['Date', 'Close'], parse_dates=['Date'],index_col='Date')
stock_price = pd.concat([df.Close[:'2015-06-12']/2,df.Close['2015-06-15':]]) # adjustment
stock_price = stock_price.squeeze()
plt.figure(figsize=(17,5))
plt.plot(stock_price)
plt.title("Closing Price Adjusted",fontsize=20)
plt.show()


# In[ ]:


y_train = stock_price.iloc[:190]
y_test = stock_price.iloc[190:]

def plot_pred(pred,title):
    plt.figure(figsize=(17,5))
    plt.plot(y_train,label='Train')
    plt.plot(y_test,label='Actual')
    plt.plot(pred,label='Predicted')
    plt.ylabel("Stock prices")
    plt.title(title,fontsize=20)
    plt.legend()
    plt.show()


# In[ ]:


print("start:",y_test.index.min())
print("end:",y_test.index.max())


# In[ ]:


y_test.shape


# # Auto Regression
# The prediction will be a linear combination of the past values. 

# In[ ]:


from statsmodels.tsa.ar_model import AR
ar_model = AR(y_train).fit()
y_ar = ar_model.predict(190,247)
y_ar = y_ar.reset_index(drop=True)
y_ar.index = y_test.index
mse(y_ar,y_test)


# In[ ]:


plot_pred(y_ar,"Autoregression")


# # Moving Average
#   Moving Average is not the predictions based on the moving average of the previous values, instead it is the moving average of the residuals of the previous values.
# 

# In[ ]:


from statsmodels.tsa.arima_model import ARMA
ma = ARMA(y_train, order=(0, 1)).fit()
y_mam = ma.predict(190,247)
y_mam.index = y_test.index
mse(y_mam,y_test)


# In[ ]:


plot_pred(y_mam,"Moving Average(MA)")


# # ARMA
# This is a combination of Auto Regression and Moving Average.

# In[ ]:


arma = ARMA(y_train,order=(1,1)).fit()
y_arma =arma.predict(190,247)
y_arma.index = y_test.index
mse(y_arma,y_test)


# In[ ]:


plot_pred(y_arma,"ARMA Model")


# # ARIMA
#   This is an extension of ARMA wherein the 'Integration' term finds the difference of the values in the series. This will reduce the trend. In ARIMA(1,1,1) we will predict the value based on the difference of the previous two values and the moving average of the same two values.
#   
#    But this unsuitable for Series with Seasonality. We have a seasonality in our data, but we do not have a trend. Lets construct an ARIMA model for this.

# In[ ]:


from statsmodels.tsa.arima_model import ARIMA
arima = ARIMA(y_train,order=(1,1,1)).fit(disp=False)
y_arima = arima.predict(190,247,typ='levels')
y_arima.index = y_test.index
mse(y_arima,y_test)


# In[ ]:


plot_pred(y_arima,"ARIMA Model")


# We donot have a good prediction as there is no significant trend in the data!

# # SARIMA

# Seasonal ARIMA is applicable to time series with trend and seasonality. The results are predicted as the combination of ARIMA model of the entire series and the seasonal component of the time series. Seasonality can be determined from Auto correlation Plots.

# In[ ]:


from statsmodels.tsa.statespace.sarimax import SARIMAX
sarima = SARIMAX(y_train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 45),mle_regression=True).fit(disp=False)
y_sarima = sarima.predict(190,247,typ='levels')
y_sarima.index = y_test.index
mse(y_sarima,y_test)
# Here we can assume a max seasonality of 45 since the function requires atleat 4 seasons. 


#   This also does not make a good prediction because there are only one seasonality (7 months) seen in the training data. We can better predict if we have four seasons in the training data.

# So this is the broad view of univariate time series analysis of Stock Price.

# Ref: https://machinelearningmastery.com/time-series-forecasting-methods-in-python-cheat-sheet/
