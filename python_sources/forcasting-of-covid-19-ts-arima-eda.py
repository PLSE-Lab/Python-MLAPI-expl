#!/usr/bin/env python
# coding: utf-8

# Importing all the required packages

# In[ ]:


import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import matplotlib
from pandas import datetime
import seaborn as sns
import plotly.express as px
from statsmodels.tsa.stattools import adfuller, acf, pacf, arma_order_select_ic
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from pandas.plotting import autocorrelation_plot
import pmdarima as pm


# Loading the data

# In[ ]:


covid_train=pd.read_csv(r"../input/covid19-global-forecasting-week-3/train.csv")
covid_test=pd.read_csv(r"../input/covid19-global-forecasting-week-3/test.csv")


# In[ ]:


covid_train.head()


# In[ ]:


covid_test.head()


# In[ ]:


covid_train.isnull().sum()/covid_train.shape[0]
covid_train.columns
covid_train.fillna('NA', inplace=True)


# Grouping up the confirmed cases as per countries

# In[ ]:


by_country = covid_train.groupby(['Country_Region','Province_State','Date'])['ConfirmedCases'].sum()                           .groupby(['Country_Region','Province_State']).max().sort_values()                           .groupby(['Country_Region']).sum().sort_values(ascending = False)
covid_train.fillna("NA").groupby('Country_Region')['ConfirmedCases'].sum().sort_values()
by_country_df = pd.DataFrame(by_country)


# Visualisation of Confirmed cases as per countries using Seaborn package

# In[ ]:


#using seaborn
sns.set(style="darkgrid")
sns.barplot(x=by_country_df.index[0:8], y="ConfirmedCases", data= by_country_df.head(8))


# Visualisation of Confirmed cases as per countries using Plotly package

# In[ ]:


#using plotly.express
sns.barplot(data = by_country_df.head(15), x= by_country_df.index[0:15], y= 'ConfirmedCases')

Grouping the data as per date and country region
# In[ ]:


by_date = covid_train.groupby(['Country_Region','Date'])['ConfirmedCases'].sum().sort_values().reset_index()
by_date_df = pd.DataFrame(by_date)


# Plotting the confirmed cases by per day in India

# In[ ]:


sns.barplot(data = by_date_df.loc[(by_date_df['Country_Region']=='India') & (by_date_df.Date >= '2020-01-22')].sort_values('ConfirmedCases', ascending= True),
            x= 'Date', y= 'ConfirmedCases')


# Plotting the confirmed cases by per day in United States

# In[ ]:


sns.barplot(data = by_date_df.loc[(by_date_df['Country_Region']=='US') & (by_date_df.Date >= '2020-01-22')].sort_values('ConfirmedCases', ascending= True),
            x= 'Date', y= 'ConfirmedCases')


# Plotting the confirmed cases by per day in China

# In[ ]:


sns.barplot(data = by_date_df.loc[(by_date_df['Country_Region']=='China') & (by_date_df.Date >= '2020-01-22')].sort_values('ConfirmedCases', ascending= True),
            x= 'Date', y= 'ConfirmedCases')


# Plotting the confirmed cases by per day in Germany

# In[ ]:


sns.barplot(data = by_date_df.loc[(by_date_df['Country_Region']=='Germany') & (by_date_df.Date >= '2020-01-22')].sort_values('ConfirmedCases', ascending= False),
            x= 'Date', y= 'ConfirmedCases')


# Plotting the confirmed cases by per day in Italy

# In[ ]:


sns.barplot(data = by_date_df.loc[(by_date_df['Country_Region']=='Italy') & (by_date_df.Date >= '2020-01-22')].sort_values('ConfirmedCases', ascending= True),
            x= 'Date', y= 'ConfirmedCases')


# In[ ]:


cols1 = ['Id', 'Province_State', 'Country_Region', 'Fatalities']
COVID_ts_ConfirmedCases_train = covid_train.drop(cols1, axis= 1)
#COVID_ts_ConfirmedCases_train.shape

cols2 = ['ForecastId', 'Province_State', 'Country_Region']
COVID_ts_ConfirmedCases_test = covid_test.drop(cols2, axis= 1)
#COVID_ts_ConfirmedCases_test.shape


# In[ ]:


COVID_ts_ConfirmedCases_train.index = pd.to_datetime(COVID_ts_ConfirmedCases_train['Date'])
COVID_ts_ConfirmedCases_train.drop('Date', inplace= True, axis= 1)
#COVID_ts_ConfirmedCases_train.shape
#COVID_ts_ConfirmedCases_train.info()

COVID_ts_ConfirmedCases_test.index = pd.to_datetime(COVID_ts_ConfirmedCases_test['Date'])
COVID_ts_ConfirmedCases_test.drop('Date', inplace= True, axis= 1)
#COVID_ts_ConfirmedCases_test.shape
#COVID_ts_ConfirmedCases_test.info()


# In[ ]:


COVID_ts_ConfirmedCases_train = COVID_ts_ConfirmedCases_train.resample('d').mean()
COVID_ts_ConfirmedCases_test = COVID_ts_ConfirmedCases_test.resample('d').sum()/COVID_ts_ConfirmedCases_test.shape[0]


# Confirmed Cases Globally

# In[ ]:


COVID_ts_ConfirmedCases_train.plot()


# Lag vs Autocorrelation

# In[ ]:


autocorrelation_plot(COVID_ts_ConfirmedCases_train)


# In[ ]:


result = adfuller(COVID_ts_ConfirmedCases_train.dropna())
print('ADF Statistic: {}'.format(result[0]))
print('p-value: {}'.format(result[1]))
#print('Critical Values:')
for key, value in result[4].items():
    print('Critical Values:','\t{}: {}'.format(key, value))

Original Series ACF
# In[ ]:


fig = plt.figure()
plt.plot(COVID_ts_ConfirmedCases_train.ConfirmedCases)
plt.show()

plot_acf(COVID_ts_ConfirmedCases_train.ConfirmedCases) #1
plt.show()


# 1st differencing of Origianl series and ACF
# To find MA(q) parameter

# In[ ]:


plt.plot(COVID_ts_ConfirmedCases_train.ConfirmedCases.diff())
plt.show()

plot_acf(COVID_ts_ConfirmedCases_train.ConfirmedCases.diff()) #0
plt.show()


# PACF plots

# In[ ]:


plot_pacf(COVID_ts_ConfirmedCases_train.ConfirmedCases)
plt.show()


# 1st differencing
# To find AR(p)

# In[ ]:


plot_pacf(COVID_ts_ConfirmedCases_train.ConfirmedCases.diff()) #1 = AR = p
plt.show()


# Building ARIMA

# In[ ]:


model = ARIMA(COVID_ts_ConfirmedCases_train.ConfirmedCases, order=(1,2,0))
model_fit = model.fit(disp= -1)
print(model_fit.summary())


# # Actual vs Fitted

# In[ ]:


model_fit.plot_predict(dynamic=False)
plt.show()


# Plotting Forecaste

# In[ ]:


fc, se, conf = model_fit.forecast(31, alpha= 0.05) #95% confidence
test = pd.DataFrame(COVID_ts_ConfirmedCases_test.loc['2020-04-07':])
# Make as pandas series
fc_series = pd.Series(fc, index=test.index)
lower_series = pd.Series(conf[:, 0], index=test.index)
upper_series = pd.Series(conf[:, 1], index=test.index)

# Plot
plt.figure(figsize=(12,5), dpi=100)
plt.plot(COVID_ts_ConfirmedCases_train, label='training')
#plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series,
                 color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()


# **Auto ARIMA**

# In[ ]:


model = pm.auto_arima(COVID_ts_ConfirmedCases_train.ConfirmedCases, start_p=1, start_q=1,
                   test='adf',       # use adftest to find optimal 'd'
                   max_p=3, max_q=3, # maximum p and q
                   m=1,              # frequency of series
                   d=None,           # let model determine 'd'
                   seasonal=False,   # No Seasonality
                   start_P=0,
                   D=0,
                   trace=True,
                   error_action='ignore',
                   suppress_warnings=True,
                   stepwise=True)


# Modifying p, d, q as per AIC factor provided by Aouto-ARIMA

# In[ ]:


model = ARIMA(COVID_ts_ConfirmedCases_train.ConfirmedCases, order=(1,2,1))
model_fit = model.fit(disp= -1)
print(model_fit.summary())


# In[ ]:


fc, se, conf = model_fit.forecast(31, alpha= 0.05) #95% confidence
test = pd.DataFrame(COVID_ts_ConfirmedCases_test.loc['2020-04-07':])
# Make as pandas series
fc_series = pd.Series(fc, index=test.index)
lower_series = pd.Series(conf[:, 0], index=test.index)
upper_series = pd.Series(conf[:, 1], index=test.index)

# Plot
plt.figure(figsize=(12,5), dpi=100)
plt.plot(COVID_ts_ConfirmedCases_train, label='training')
#plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series,
                 color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()

Prediction
# In[ ]:



from pandas import datetime
start_index = datetime(2020, 4, 7)
end_index = datetime(2020, 5, 7)
forecast = model_fit.predict(start=start_index, end=end_index)


# In[ ]:


forecast

