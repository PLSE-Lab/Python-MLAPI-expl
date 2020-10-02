#!/usr/bin/env python
# coding: utf-8

# ARIMA - a basic time series modeling method
# ==============
# I studied and tried a basic time series modelling method ARIMA.  
# I hope it will be a reference for learning to beginners in ARIMA.
# 
# 
# - Main topics
#   - Using MA(75) series as ARIMA train data
#   - Using auto ARIMA order(p, q) estimation function
#   - Change the period of train data from all periods to the latest period

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import copy
from pandas import DataFrame
from pandas import concat
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


# Read input data, calculate difference series, 75-day moving average MA75D

# In[ ]:


#covid_data = pd.read_csv("../input/ntt-data-global-ai-challenge-06-2020/COVID-19_and_Price_dataset.csv")
ts = pd.read_csv("../input/ntt-data-global-ai-challenge-06-2020/Crude_oil_trend_From1986-01-02_To2020-06-15.csv")

ts["Diff"] = ts["Price"].diff().dropna()
ts["MA75D"] = ts["Price"].rolling(window=75).mean()
ts["Diff_MA75D"] = ts["MA75D"].diff().dropna()

print(ts.shape)
ts.tail()


# 
# 
# 

# In[ ]:


plt.plot(ts["Price"])
plt.show()
plt.plot(ts["Diff"])
plt.show()
plt.plot(ts["MA75D"])
plt.show()
plt.plot(ts["Diff_MA75D"])
plt.show()


# # ARIMA model order estimation
# ARIMA is reprented as ARIMA(p,d,q). I'll estimate p, q.
# 
# 
# **Characteristics of AR process**
# - Autocorrelation gradually diminishes.
# - The partial autocorrelation has a partial autocorrelation up to the time point p, and becomes 0 after the time point p.
# 
# **Characteristics of MA process**
# - Autocorrelation has autocorrelation up to time point q, and becomes 0 after time point q
# - Partial autocorrelation gradually decreases.

# In[ ]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Autocorrelation
plot_acf(ts["Diff_MA75D"][75:],lags=100)
plt.xlabel("Lag")
plt.ylabel("ACF")
plt.show()

# Partial Autocorrelation
plot_pacf(ts["Diff_MA75D"][75:], lags=100)
plt.xlabel("Lag")
plt.ylabel("PACF")
plt.show()


# We can know p (AR model order) is around 2 by the PACF plot.  
# But, we cannot know q (MA model order) from the ACF Plot.

# In[ ]:


#Split into learning data and test data. Test data uses the last 28 steps
train, test = ts["Diff_MA75D"][75:-28], ts["Diff_MA75D"][-28:]


# In[ ]:


#Execute automatic ARMA estimation function to estimate p and q of ARIMA(p,d,q)
resDiff = sm.tsa.arma_order_select_ic(train, ic='aic', trend='nc')
resDiff

#We can know (p, q)=(4,2) from the result 'aic_min_order': (4, 2)


# # Model fit and predict

# Create a model using the obtained values of (p,d,q).

# In[ ]:


model = ARIMA(train.to_numpy(), order=(4,1,2))

model_fit = model.fit(disp=0)
print(model_fit.summary())


# In[ ]:





# In[ ]:


#forecast() returns only one step forecast.
#if you want future(out of sample) forecasting, set the argument "steps".
predict = model_fit.forecast(steps=28)
yhat_arima = predict[0]
yhat_arima


# # Residual analysis  
# - Regarding the time stationarity of the residuals, the residuals are almost constant and the error does not seem to increase over time. (Excluding the last sudden change)
# - For the distribution of residuals, kernel density estimation reveals that the model predictions are unbiased because the mean is near zero and symmetric.

# In[ ]:


# plot residual errors
residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
plt.show()
residuals.plot(kind='kde')
plt.show()
print(residuals.describe())


# # Evaluation

# In[ ]:


#change differencial series to original series
def get_price(preds):
    base_test = ts["MA75D"].tolist()[-28]
    oil_preds = []
    for i in range(len(preds)):
        oil_preds.append(round(base_test + sum(preds[0:i]),6))
    return oil_preds


# In[ ]:


plt.plot(get_price(test), color="black", label="Actual Oil Price")
plt.plot(get_price(yhat_arima), color="blue", label="ARIMA Prediction")
plt.legend()
plt.show()


# In[ ]:


# calculate RMSE
rmse = np.sqrt(mean_squared_error(get_price(test), get_price(yhat_arima)))

#rmse = np.sqrt(mean_squared_error(test, yhat_arima)) # differencial RMSE

print('Test RMSE: %.3f' % rmse)


# # Create a model by limiting the range of training data to the nearest prediction period
# * Although the generalization performance is lost, the most recent movement is too rapid, so try limiting the range of the learning data to the most recent and making it overlearning.
# * RMSE become bad

# In[ ]:


#Limit learning data to the latest
train, test = ts["Diff_MA75D"][-200:-28], ts["Diff_MA75D"][-28:]
#Run automatic ARMA estimation function on difference series
resDiff = sm.tsa.arma_order_select_ic(train, ic='aic', trend='nc')
resDiff


# In[ ]:


model = ARIMA(train.to_numpy(), order=(2,1,1))
model_fit = model.fit(disp=0)
print(model_fit.summary())


# In[ ]:


#forecast() returns only one step forecast.
#if you want future(out of sample) forecasting, set the argument "steps".
predict = model_fit.forecast(steps=28)
yhat_arima = predict[0]
yhat_arima


# In[ ]:


plt.plot(get_price(test), color="black", label="Actual Oil Price")
plt.plot(get_price(yhat_arima), color="blue", label="ARIMA Prediction")
plt.legend()
plt.show()


# In[ ]:


# calculate RMSE
rmse = np.sqrt(mean_squared_error(get_price(test), get_price(yhat_arima)))
print('Test RMSE: %.3f' % rmse)


# Performance got worse. It is meaningless to fit training data of latest period.
