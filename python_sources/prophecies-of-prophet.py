#!/usr/bin/env python
# coding: utf-8

# # To the existing fork i have added future forcasting

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from fbprophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
plt.style.use('fivethirtyeight') 


# In[ ]:


pjme = pd.read_csv('../input/PJME_hourly.csv', index_col=[0], parse_dates=[0]) # We set the index column and know it has dates


# In[ ]:


split_date = '01-Jan-2015'
pjme_train = pjme.loc[pjme.index <= split_date].copy()
pjme_test = pjme.loc[pjme.index > split_date].copy()


# In[ ]:


pjme_test     .rename(columns={'PJME_MW': 'TEST SET'})     .join(pjme_train.rename(columns={'PJME_MW': 'TRAINING SET'}),
          how='outer') \
    .plot(figsize=(15,5), title='PJM East', style='.')
plt.show()


# In[ ]:


model = Prophet()
model.fit(pjme_train.reset_index()               .rename(columns={'Datetime':'ds',
                               'PJME_MW':'y'}))


# In[ ]:


pjme_test_fcst = model.predict(df=pjme_test.reset_index()                                    .rename(columns={'Datetime':'ds'}))


# In[ ]:


pjme_test_fcst.head()


# In[ ]:


import datetime as dt
# Plot the forecast with the actuals
f, ax = plt.subplots(1)
f.set_figheight(5)
f.set_figwidth(15)
ax.scatter(pjme_test.index, pjme_test['PJME_MW'], color='r')
fig = model.plot(pjme_test_fcst, ax=ax)
l = dt.datetime.strptime('01-01-2017', '%m-%d-%Y').date()
u = dt.datetime.strptime('01-30-2017', '%m-%d-%Y').date()
ax.set_xbound(lower=l, upper=u)
ax.set_ylim(0, 60000)
plot = plt.suptitle('Jan 2017  Forecast vs Actuals non-Holiday Model')


# In[ ]:


# Plot the forecast with the actuals
f, ax = plt.subplots(1)
f.set_figheight(5)
f.set_figwidth(15)
ax.scatter(pjme_test.index, pjme_test['PJME_MW'], color='r')
fig = model.plot(pjme_test_fcst, ax=ax)
l = dt.datetime.strptime('07-01-2015', '%m-%d-%Y').date()
u = dt.datetime.strptime('07-07-2015', '%m-%d-%Y').date()
ax.set_xbound(lower=l, upper=u)
ax.set_ylim(0, 60000)
plot = plt.suptitle('Week of July Forecast vs Actuals non-Holiday Model')


# In[ ]:


future=model.make_future_dataframe(50000, freq='H')


# In[ ]:


forecast=model.predict(future)


# In[ ]:


forecast.tail()


# In[ ]:


# Plot the forecast with the actuals
f, ax = plt.subplots(1)
f.set_figheight(5)
f.set_figwidth(15)
fig = model.plot(forecast, ax=ax)
l = dt.datetime.strptime('01-01-2019', '%m-%d-%Y').date()
u = dt.datetime.strptime('02-01-2019', '%m-%d-%Y').date()
ax.set_xbound(lower=l, upper=u)
ax.set_ylim(0, 60000)
plot = plt.suptitle('Jan 2019  Forecast')


# In[ ]:


def mean_absolute_percentage_error(y_true, y_pred): 
    """Calculates MAPE given y_true and y_pred"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mean_absolute_percentage_error(y_true=pjme_test['PJME_MW'],
                   y_pred=pjme_test_fcst['yhat'])


# In[ ]:




