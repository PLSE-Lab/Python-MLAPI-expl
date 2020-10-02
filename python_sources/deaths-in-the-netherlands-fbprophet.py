#!/usr/bin/env python
# coding: utf-8

# # Import modules

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error as mae
from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation, performance_metrics
from fbprophet.plot import plot_cross_validation_metric
register_matplotlib_converters()


# # Data

# In[ ]:


path_data = '/kaggle/input/weekly-deaths-in-the-netherlands/deaths_NL.csv'
df = pd.read_csv(path_data)

df = df[df["Week"].str.len()<20]
df['Week'] = df['Week'].str.replace('*', '')
df["Week"] = pd.to_datetime(df["Week"]+'0', format="%Y week %U%w")

df.head()


# In[ ]:


plt.figure(figsize=(15,5))
plt.plot(df.Week, df['All ages: both sexes'], label='All ages: both sexes')
plt.legend()
plt.show()


# In[ ]:


num_week_test_set = 50
num_week_to_pred = 100


# In[ ]:


df_both_sexes = df[['Week', 'All ages: both sexes']]
df_both_sexes.columns = ['ds', 'y']

train = df_both_sexes[:-num_week_test_set]
test = df_both_sexes[-num_week_test_set:]

train_log = train.copy()
train_log['y'] = np.log1p(train_log['y'])


# In[ ]:


plt.figure(figsize=(15,5))
plt.title('Train and test sets')
plt.plot(train.ds, train.y, label='train')
plt.plot(test.ds, test.y, label='test')
plt.legend()
plt.show()


# # Train and cross validation

# > ## Model train

# In[ ]:


m = Prophet(changepoint_prior_scale=0.003, weekly_seasonality=True)
m.fit(train)
future = m.make_future_dataframe(periods=num_week_to_pred, freq='W', include_history=False)
forecast = m.predict(future)
# forecast['yhat'] = np.expm1(forcast.yhat)
forecast.head()


# In[ ]:


def plot_forecast(test, forecast):
    score = np.round(mae(test.y, forecast[:num_week_test_set].yhat), 2)
    plt.figure(figsize=(15,5))
    plt.title("Forecast last {} points.\nProphet.\nScore: {}".format(num_week_to_pred, score))
    plt.plot(test.ds, test.y, 'o-', label='test')
    plt.plot(forecast.ds, forecast.yhat, 'o-', label='forecast')
    plt.legend()
    plt.show()


# In[ ]:


m.plot_components(forecast);


# ## Validation

# In[ ]:


len(train)


# In[ ]:


cv = cross_validation(m, initial='800 days', period='10 days', horizon='10 days')


# In[ ]:


performance_metrics(cv)


# In[ ]:


plot_cross_validation_metric(cv, 'mae');


# # Prediction

# In[ ]:


plot_forecast(test, forecast)


# In[ ]:




