#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from fbprophet import Prophet
get_ipython().run_line_magic('matplotlib', 'inline')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


df = pd.read_csv('/kaggle/input/for-simple-exercises-time-series-forecasting/Miles_Traveled.csv')


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


# Renaming the columns of dataframe as per the convention of Prophet library
df.columns = ['ds', 'y']
df.head()


# In[ ]:


# Conveting the type of ds column to datetime format
df['ds'] = pd.to_datetime(df['ds'])
df.head()


# In[ ]:


# Plotting to see the actual behaviour of datasets
df.plot(x='ds', y='y')


# In[ ]:


# finding the number of rows in the dataset
len(df)


# In[ ]:


# Splitting the dataset into train and test (test is for one year(12 months -> 12 rows))
train = df.iloc[:576]
test = df.iloc[576:]


# In[ ]:


# Create an instance of Prophet
m = Prophet()
# Fit the training data
m.fit(train)
# Create a future dataframe 
future = m.make_future_dataframe(periods=12, freq='MS') # for daily data no need to specify freq
# making predictions
forecast = m.predict(future)


# In[ ]:


forecast.shape


# In[ ]:


forecast.iloc[-12:,]


# In[ ]:


forecast.tail()


# In[ ]:


# Plotting the predicted values against the original value
ax = forecast.plot(x='ds', y='yhat', label='Predictions', legend=True, figsize=(12,8))
test.plot(x='ds', y='y', label='True Test Data', legend=True, ax=ax, xlim=('2018-01-01', '2019-01-01'))


# In[ ]:


# Calculate the errors
from statsmodels.tools.eval_measures import rmse


# In[ ]:


predictions = forecast.iloc[-12:]['yhat']


# In[ ]:


predictions


# In[ ]:


test['y']


# In[ ]:


# Calculating rmse values
rmse(predictions, test['y'])


# In[ ]:


test.mean()


# In[ ]:


from fbprophet.diagnostics import cross_validation, performance_metrics
from fbprophet.plot import plot_cross_validation_metric


# In[ ]:


# initial training period
initial = 5 * 365
initial = str(initial) + ' days' # as Prophet requires string code

# period length for which we are gonna perform cross validation
period = 5 * 365
period = str(period) + ' days' # as Prophet requires string code

# horizon of prediction for each fold
# we'll forecast one year ahead
horizon = 365
horizon = str(horizon) + ' days'


# In[ ]:


df_cv = cross_validation(m, initial=initial, period=period, horizon=horizon)


# In[ ]:


df_cv.head()


# In[ ]:


len(df_cv)


# In[ ]:


performance_metrics(df_cv)


# In[ ]:


df.head(2)


# In[ ]:


df.tail(2)


# In[ ]:


plot_cross_validation_metric(df_cv, metric='rmse');


# In[ ]:




