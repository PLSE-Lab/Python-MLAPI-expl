#!/usr/bin/env python
# coding: utf-8

# # Hourly Time Series Forecasting using Facebook's Prophet

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from fbprophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
plt.style.use('fivethirtyeight')


# # Data
# The data we will be using is hourly power consumption data from PJM. Energy consumtion has some unique charachteristics. It will be interesting to see how prophet picks them up.
# 
# Pulling the `PJM East` which has data from 2002-2018 for the entire east region.

# In[ ]:


pjme = pd.read_csv('../input/PJME_hourly.csv', index_col=[0], parse_dates=[0])


# In[ ]:


color_pal = ["#F8766D", "#D39200", "#93AA00", "#00BA38", "#00C19F", "#00B9E3", "#619CFF", "#DB72FB"]
_ = pjme.plot(style='.', figsize=(15,5), color=color_pal[0], title='PJM East')


# In[ ]:


pjme.index


# # Train/Test Split
# Cut off the data after 2015 to use as our validation set.

# In[ ]:


split_date = '01-Jan-2015'
pjme_train = pjme.loc[pjme.index <= split_date].copy()
pjme_test = pjme.loc[pjme.index > split_date].copy()


# In[ ]:


_ = pjme_test     .rename(columns={'PJME_MW': 'TEST SET'})     .join(pjme_train.rename(columns={'PJME_MW': 'TRAINING SET'}), how='outer')     .plot(figsize=(15,5), title='PJM East', style='.')


# # Simple Prophet Model

# In[ ]:


# Format data for prophet model using ds and y
pjme_train.reset_index().rename(columns={'Datetime':'ds', 'PJME_MW':'y'}).head()


# In[ ]:


# Setup and train model
model = Prophet()
model.fit(pjme_train.reset_index().rename(columns={'Datetime':'ds', 'PJME_MW':'y'}))


# In[ ]:


# Predict on training set with model
pjme_test_fcst = model.predict(df=pjme_test.reset_index().rename(columns={'Datetime':'ds'}))


# In[ ]:


# Plot the forecast
f, ax = plt.subplots(1)
f.set_figheight(5)
f.set_figwidth(15)
fig = model.plot(pjme_test_fcst, ax=ax)


# In[ ]:


# Plot the components
fig = model.plot_components(pjme_test_fcst)


# # Compare Forecast to Actuals

# In[ ]:


# Plot the forecast with the actuals
f, ax = plt.subplots(1)
f.set_figheight(5)
f.set_figwidth(15)
ax.scatter(pjme_test.index, pjme_test['PJME_MW'], color='r')
fig = model.plot(pjme_test_fcst, ax=ax)


# # Look at first month of predictions

# In[ ]:


# Plot the forecast with the actuals
f, ax = plt.subplots(1)
f.set_figheight(5)
f.set_figwidth(15)
ax.scatter(pjme_test.index, pjme_test['PJME_MW'], color='r')
fig = model.plot(pjme_test_fcst, ax=ax)
ax.set_xbound(lower='01-01-2015', upper='02-01-2015')
ax.set_ylim(0, 60000)
plot = plt.suptitle('January 2015 Forecast vs Actuals')


# # Single Week of Predictions

# In[ ]:


# Plot the forecast with the actuals
f, ax = plt.subplots(1)
f.set_figheight(5)
f.set_figwidth(15)
ax.scatter(pjme_test.index, pjme_test['PJME_MW'], color='r')
fig = model.plot(pjme_test_fcst, ax=ax)
ax.set_xbound(lower='01-01-2015', upper='01-08-2015')
ax.set_ylim(0, 60000)
plot = plt.suptitle('First Week of January Forecast vs Actuals')


# # Error Metrics
# 
# Our RMSE error is 43761675  
# Our MAE error is 5181.78  
# Our MAPE error is 16.5%
# 
# by comparison in the XGBoost model our errors were significantly less (8.9% MAPE):
# [Check that out here](https://www.kaggle.com/robikscube/hourly-time-series-forecasting-with-xgboost/)

# In[ ]:


mean_squared_error(y_true=pjme_test['PJME_MW'],
                   y_pred=pjme_test_fcst['yhat'])


# In[ ]:


mean_absolute_error(y_true=pjme_test['PJME_MW'],
                   y_pred=pjme_test_fcst['yhat'])


# In[ ]:


def mean_absolute_percentage_error(y_true, y_pred): 
    """Calculates MAPE given y_true and y_pred"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mean_absolute_percentage_error(y_true=pjme_test['PJME_MW'],
                   y_pred=pjme_test_fcst['yhat'])


# In[ ]:




