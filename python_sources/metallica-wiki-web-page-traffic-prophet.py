#!/usr/bin/env python
# coding: utf-8

# # Metallica Spain Tour [Prophet]
# ### Based on Wikipedia Web Page Traffic

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy import stats
import statsmodels.api as sm

import matplotlib.pyplot as plt # plotting
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns # plotting

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## Data

# In[ ]:


base_url = '/kaggle/input/web-traffic-time-series-forecasting/'

train_1 = pd.read_csv(base_url+'train_1.csv')
train_2 = pd.read_csv(base_url+'train_2.csv')


# In[ ]:


train_1.shape


# ## Data description

# #### Train Data Content - 145.063 rows representing different Wikipedia URL pages, 551 columns
# #### first column is the URL page and then each column represents a value of the number of visits to the page in that day
# #### dates from 2015-07-01 to 2016-12-31 (1.5 year, total of 550 days)

# In[ ]:


train_1.head()


# ## Creating Matallica ES and basic plots

# In[ ]:


trainT = train_1.drop('Page', axis=1).T
trainT.columns = train_1.Page.values
trainT.head()


# In[ ]:


metallica = pd.DataFrame(trainT['Metallica_es.wikipedia.org_all-access_all-agents'])
metallica.head()


# In[ ]:


print (metallica.shape)


# In[ ]:


print (metallica.isnull().sum())


# In[ ]:


plt.figure(figsize=(24, 12))
metallica.plot();


# In[ ]:


def plotMovingAverage(series, window, plot_intervals=False, scale=1.96, plot_anomalies=False):

    """
        series - dataframe with timeseries
        window - rolling window size 
        plot_intervals - show confidence intervals
        plot_anomalies - show anomalies 

    """
    # Calculate and plot rolling mean
    rolling_mean = series.rolling(window=window).mean()
    
    plt.figure(figsize=(15,5))
    plt.title("Moving average\n window size = {}".format(window))
    plt.plot(rolling_mean, "g", label="Rolling mean trend")

    # Plot confidence intervals for smoothed values
    if plot_intervals:
        mae = mean_absolute_error(series[window:], rolling_mean[window:])
        deviation = np.std(series[window:] - rolling_mean[window:])
        lower_bond = rolling_mean - (mae + scale * deviation)
        upper_bond = rolling_mean + (mae + scale * deviation)
        plt.plot(upper_bond, "r--", label="Upper Bond / Lower Bond")
        plt.plot(lower_bond, "r--")
        
        # Having the intervals, find abnormal values
        if plot_anomalies:
            anomalies = pd.DataFrame(index=series.index, columns=series.columns)
            anomalies[series<lower_bond] = series[series<lower_bond]
            anomalies[series>upper_bond] = series[series>upper_bond]
            plt.plot(anomalies, "ro", markersize=10)
    
    # Plot original series values
    plt.plot(series[window:], label="Actual values")
    plt.legend(loc="upper left")
    plt.grid(True)


# In[ ]:


plotMovingAverage(metallica, 14)


# > ## Modeling and forecast

# In[ ]:


from fbprophet import Prophet


# In[ ]:


metallica.columns


# In[ ]:


metallica.rename(columns={'Metallica_es.wikipedia.org_all-access_all-agents': 'y'}, inplace=True)
metallica.head()


# In[ ]:


ds = pd.Series(metallica.index)
y = pd.Series(metallica.iloc[:,0].values)
frame = { 'ds': ds, 'y': y }
df = pd.DataFrame(frame)
df.head()


# In[ ]:


df.plot();


# In[ ]:


# Instantiate and fit the Prophet model
m = Prophet()
m.fit(df);


# In[ ]:


# Make future predictions to the next 60 days
forecast = m.make_future_dataframe(periods=60)


# In[ ]:


forecast.shape


# In[ ]:


forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(3)


# In[ ]:


forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head(3)


# ### Basic plotting

# In[ ]:


fig1 = m.plot(forecast)


# In[ ]:


fig2 = m.plot_components(forecast)


# In[ ]:


import matplotlib.pyplot as plt

plt.figure(figsize=(15, 7))
plt.plot(df.y)
plt.plot(forecast.yhat, "g");


# ### Saturating forecasts

# In[ ]:


df['cap'] = 500
df['floor'] = 0.0
future['cap'] = 500
future['floor'] = 0.0
m = Prophet(growth='logistic')
forecast = m.fit(df).predict(future)
fig1 = m.plot(forecast)
fig2 = m.plot_components(forecast)


# ### Seasonality

# **Seasonality & Holiday Parameters**
# 
# Parameter and Description
# 
# - yearly_seasonality -> Fit yearly seasonality
# - weekly_seasonality -> Fit weekly seasonality
# - daily_seasonality -> Fit daily seasonality
# - holidays -> Feed dataframe containing holiday name and date
# - seasonality_prior_scale -> Parameter for changing strength of seasonality model
# - holidays_prior_scale -> Parameter for changing strength of holiday model
# 
# Source: https://www.analyticsvidhya.com/blog/2018/05/generate-accurate-forecasts-facebook-prophet-python-r/

# In[ ]:


m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
forecast = m.fit(df).predict(future)
fig1 = m.plot(forecast)
fig2 = m.plot_components(forecast)


# ### Changepoints

# In[ ]:


from fbprophet.plot import add_changepoints_to_plot
fig = m.plot(forecast)
a = add_changepoints_to_plot(fig.gca(), m, forecast)


# In[ ]:


m = Prophet(changepoint_prior_scale=0.9)
forecast = m.fit(df).predict(future)
fig = m.plot(forecast)


# ### Holidays

# In[ ]:


from datetime import date
import holidays

# Select country
es_holidays = holidays.Spain(years = [2015,2016,2017])
es_holidays = pd.DataFrame.from_dict(es_holidays, orient='index')
es_holidays = pd.DataFrame({'holiday': 'Spain', 'ds': es_holidays.index})


# In[ ]:


m = Prophet(holidays=es_holidays)
m.add_country_holidays(country_name='ES')
forecast = m.fit(df).predict(future)
fig1 = m.plot(forecast)
fig2 = m.plot_components(forecast)


# ### Uncertainty interval

# #### This parameter determines if the model uses Maximum a posteriori (MAP) estimation or a full 
# ##### Bayesian inference with the specified number of Markov Chain Monte Carlo (MCMC) samples to train and predict.
# ##### So if you make MCMC zero then it will do MAP estimation, otherwise you need to specify the number of samples to use with MCMC.
# ##### Source: https://towardsdatascience.com/implementing-facebook-prophet-efficiently-c241305405a3

# #### Uncertainty in the trend

# In[ ]:


m = Prophet(interval_width=0.95)
forecast = m.fit(df).predict(future)
fig1 = m.plot(forecast)
fig2 = m.plot_components(forecast)


# ### Uncertainty in seasonality

# In[ ]:


# m = Prophet(mcmc_samples=0)
m = Prophet(mcmc_samples=300)
forecast = m.fit(df).predict(future)
fig1 = m.plot(forecast)
fig2 = m.plot_components(forecast)


# ## Prediction with parametrization

# In[ ]:


m = Prophet(growth='linear',
            daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True,
            seasonality_mode='multiplicative',
            seasonality_prior_scale=25,
            changepoint_prior_scale=0.05,
            holidays=es_holidays,
            holidays_prior_scale=20,
            interval_width=0.95,
            mcmc_samples=0)

m.add_country_holidays(country_name='ES')

forecast = m.fit(df).predict(future)

fig1 = m.plot(forecast)
a = add_changepoints_to_plot(fig1.gca(), m, forecast)

fig2 = m.plot_components(forecast)


# ### Dynamic Plotting

# In[ ]:


from fbprophet.plot import plot_plotly
import plotly.offline as py
py.init_notebook_mode()

fig = plot_plotly(m, forecast)  # This returns a plotly Figure
py.iplot(fig)


# ## Prediction

# In[ ]:


plt.figure(figsize=(15, 7))
plt.plot(df.y)
plt.plot(forecast.yhat, "g");


# ## Evaluating the Model

# ### Symmetric Mean Absolute Percentage Error
# $$ SMAPE = \frac{100\%}{n} \sum_{t=1}^{n} \frac{\left|F_t - A_t\right|}{(\left|A_t\right|+\left|F_t\right|)/2} $$

# In[ ]:


def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred))
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return 200 * np.mean(diff)

# Source: http://shortnotes.herokuapp.com/how-to-implement-smape-function-in-python-149


# In[ ]:


smape_metallica = smape(df.y, forecast.yhat)
smape_metallica


# ### Cross Validation

# ##### horizon: forecast horizon
# ##### initial: size of the initial training period
# ##### period: spacing between cutoff dates
# 
# ##### Here we do cross-validation to assess prediction performance on a horizon of 60 days, 
# ##### starting with 130 days of training data in the first cutoff and then making predictions every 60 days
# ##### On this 610 days time series, this corresponds to 8 total forecasts

# In[ ]:


from fbprophet.diagnostics import cross_validation


# In[ ]:


cv_results = cross_validation(m, initial='360 days', period='30 days', horizon='60 days')


# In[ ]:


smape_cv = smape(cv_results.y, cv_results.yhat)
smape_cv


# ## Conclusions

# ### page visit are clearly on the rise around November
# ### the popular weekdays are Tueday and Friday
# ### there is a clear growing trend, that should continue according to the forecast
# ### the forecast error is 12%
# 
