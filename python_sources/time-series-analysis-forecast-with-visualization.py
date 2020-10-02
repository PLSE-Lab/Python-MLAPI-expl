#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# ### In this notebook we are going to try explore the data we have and going try answer questions like:
# 
# * How's the trend of border crossings looks like?
# * Is there coherent seasonality?
# * Is there unexpected sharp spikes or drops if so why?
# * Can we forecast the entries in near future?
# * Which border and state is busier than other/s?
# * What are the most common entry type?
# * Do presidents have observable impact on total entries in their terms?
# * And much more!
# 
# **If you liked this kernel feel free to upvote and leave feedback, appreciated!**

# In[ ]:


# Importing neccesary packages.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import warnings

#

import statsmodels.api as sm
from pylab import rcParams
import scipy.stats as ss

warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')

#

from fbprophet import Prophet


# # Meeting the Data

# In[ ]:


# Loading data:

data = pd.read_csv('/kaggle/input/us-border-crossing-data/Border_Crossing_Entry_Data.csv')


# In[ ]:


# Looking at the shape of the data:

data.shape


# In[ ]:


# Checking how data looks:

data.sample(5)


# In[ ]:


# Parsing dates:

data['Date'] = pd.to_datetime(data['Date'])


# In[ ]:


# Replacing Values with proper name:

data.rename(columns={'Value': 'Total Entries'}, inplace=True)


# In[ ]:


# Extracting year and month data for future uses:

data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month


# # Visualizing Entry Time Series Data
# 
# **It looks like total entries have been in declining trend between 2002 and 2012 and then there is slight increase but it's pretty much stable.**
# 
# **When we inspecting the data by eye we can see the there is some sort of seasonality with 3 seperate sharp declines:**
# 
# * First one is somewhere between 1997-1998, I'm not sure what's the reason behind it yet...
# * Second one is in between 2001-2002 it's highly likely due to September 11 attacks,
# * Last one is between 2008-2009, probably it's related to 2008 finance crisis.
# 
# There is some sort of seasonality effect for sure, let's inspect it in next chapter.

# In[ ]:


# Inspecting time series and rolling mean:

crossing = data[['Date', 'Total Entries']].groupby('Date').sum()
tseries = data.groupby(['Date'])['Total Entries'].agg(
    ['sum']).reset_index().rename(columns={'sum': 'Total Entries'})

tseries = tseries.set_index('Date')

fig = plt.subplots(figsize=(30, 15))

g = sns.lineplot(x=tseries.index, y='Total Entries',
                 data=tseries, label="Actual Time Series")

rmean = crossing.rolling(12, win_type='triang').mean()

g = sns.lineplot(x=rmean.index, y='Total Entries',
                 data=rmean, label="Rolling Mean 12 Months")

plt.legend(fontsize='xx-large')
plt.show()


# # Decomposing the Time Series
# 
# * **The trend chart confirms our first observations with 10 years of decline and slightly climbing next 10 years.**
# 
# 
# * **Seasonality is there for sure! It's clearly observable, It looks highly related to tourism and weather conditions and holidays. When we look closely we can see entries peak around Jul and then there is small jump before christmas and then sharp decline right after new years...**

# In[ ]:


# Decomposing the data:

decompose = sm.tsa.seasonal_decompose(
    tseries, model='additive', extrapolate_trend='freq')

# Plot.
fig, axes = plt.subplots(nrows=4, figsize=(16, 8))

decompose.observed.plot(ax=axes[0], legend=False)
axes[0].set_ylabel('Observed')

decompose.trend.plot(ax=axes[1], legend=False)
axes[1].set_ylabel('Trend')

decompose.seasonal.plot(ax=axes[2], legend=False)
axes[2].set_ylabel('Seasonal')

decompose.resid.plot(ax=axes[3], legend=False)
axes[3].set_ylabel('Residual')


plt.tight_layout()
plt.show()


# # Forecasting With SARIMAX
# 
# **We going to apply one of the most common forecasting methods: ARIMA(Autoregressive Integrated Moving Average)**

# In[ ]:


# Choosing train data:

train = tseries.loc['2010':'2020']


# In[ ]:


# Examples for seasonality, trend, and noise combinations.

p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))

seasonal_pdq = [(x[0], x[1], x[2], 12)
                for x in list(itertools.product(p, d, q))]

print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))


# In[ ]:


## Searching for paramaters / Meta for now:

#for param in pdq:
#    for param_seasonal in seasonal_pdq:
#        try:
#            mod = sm.tsa.statespace.SARIMAX(train,
#                                            order=param,
#                                            seasonal_order=param_seasonal,
#                                            enforce_stationarity=False,
#                                            enforce_invertibility=False)
#            results = mod.fit()
#            print('ARIMA{}x{}12 - AIC:{}'.format(param,
#                                                 param_seasonal, results.aic))
#        except:
#            continue


# In[ ]:


# Fitting the model.

mod = sm.tsa.statespace.SARIMAX(train,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results = mod.fit()

print(results.summary().tables[1])


# # Validating our Model
# 
# **Our model looks decent, let's continue with validating our forecast.**

# In[ ]:


# Investigating the results:

results.plot_diagnostics(figsize=(16, 8))
plt.show()


# **Our forecasts for past 5 years align with the observed values, which is good...**

# In[ ]:


# Validating forecasts:

pred = results.get_prediction(
    start=pd.to_datetime('2015-01-01'), dynamic=False)

pred_ci = pred.conf_int()

ax = tseries['1996':].plot(label='observed')

pred.predicted_mean.plot(
    ax=ax, label='One-step ahead Forecast', alpha=.8, figsize=(14, 7))

ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.1)

ax.set_xlabel('Date')
ax.set_ylabel('Total Entries')

plt.legend()
plt.show()


# # SARIMAX Results
# 
# 
# **Alright! Let's forecast next 60 months... Hmm... It looks our model captured the seasonality well with the slight increasing trend. I'd say our models confidence interval grows alot after 2 years so gets less confident but still looks fine!**

# In[ ]:


# Forecasting.

pred_uc = results.get_forecast(steps=60)
pred_ci = pred_uc.conf_int()

ax = tseries.plot(label='Total Entries', figsize=(14, 7))

pred_uc.predicted_mean.plot(ax=ax, label='Forecast')

ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.15)

ax.set_xlabel('Date')
ax.set_ylabel('Total Entries')

plt.legend()
plt.show()


# # Simple Forecasting With Prophet
# 
# **Developed by Core Data Science team at Facebook, Prophet is a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality effects.**
# 
# **We're going to forecast with another model to see how it performs against the other one we used.**
# 
# **Prophet imposes the strict condition that the input columns be named ds (date column in our case) and y (entries in our case), so we should rename the columns in our DataFrame:**

# In[ ]:


# Getting our train data ready for Prophet modelling.

train=train.reset_index(level=0)

train=train.rename(columns={'Date': 'ds', 'Total Entries': 'y'})


# In[ ]:


# Fitting the model with 0.90 confidence interval:

model = Prophet(interval_width = 0.90, seasonality_mode = 'additive')

model.fit(train)


# In[ ]:


# Creating a future dataframe with model for next 5 years.

future = model.make_future_dataframe(periods=60, freq='MS')

future.tail()


# In[ ]:


# Predicting.

forecast = model.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head()


# # Prophet Results
# 
# **It seems there is slight difference between two models, while SARIMAX is somewhat stable, Prophet shows decrease in entries for next 5 years. With the latest developments around the world this makes more sense to me but who knows...**

# In[ ]:


# Plotting the trend with changepoints in the series using fbprophet:

from fbprophet.plot import add_changepoints_to_plot

fig = model.plot(forecast)

add_changepoints_to_plot(fig.gca(), model, forecast)

plt.show()


# In[ ]:


# Plotting the components of the series:

model.plot_components(forecast)
plt.show()


# # Visualizing the Data
# 
# **It's time to get deeper into our data and get some insights!**

# # Entries by Month
# 
# **We can see the seasonality effect on Canadian border much more than Mexican border, Canadian entries gets higher in summer months meanwhile Mexican entries somewhat stable...**

# In[ ]:


# Entries by month:

month_data = data.groupby(['Month', 'Border'])[
    'Total Entries'].agg(['sum']).reset_index().rename(columns={'sum': 'Total Entries'})

fig = plt.subplots(figsize=(30, 15))
sns.barplot(x='Month', y='Total Entries', hue='Border',
            data=month_data, palette='plasma')

plt.legend(fontsize='xx-large')

plt.show()


# # Entries by Year
# 
# **Well it's clear that between 2000 and 2011 there is decreasing trend, especially for Mexican entries, meanwhile Canadian entries somewhat more stable. Mexican entries climbing up slightly after 2012 but seems slowed down in 2019, we can't talk for 2020 yet since we don't have enough data...**

# In[ ]:


# Entries by year:

year_data = data.groupby(['Year', 'Border'])[
    'Total Entries'].agg(['sum']).reset_index().rename(columns={'sum': 'Total Entries'})

fig = plt.subplots(figsize=(30, 15))

sns.barplot(x='Year', y='Total Entries', hue='Border',
            data=year_data, palette='plasma')

plt.legend(fontsize='xx-large')

plt.show()


# # Entries by State
# 
# **When we look at the entries by U.S state we can see Texas and California are most common entry points for Mexican border, meanwhile New York and Michigan are main ports for Canadian entries.**

# In[ ]:


# Entries by state:

state_data = data.groupby(['State'])['Total Entries'].agg(
    ['sum']).reset_index().sort_values('sum', ascending=False).rename(columns={'sum': 'Total Entries'})

fig = plt.subplots(figsize=(30, 15))

sns.barplot(x='Total Entries', y='State', data=state_data, palette='plasma')

plt.show()


# # Entries by Type
# 
# **Most common entry type for both borders are personal vehicles for both borders followed by pedestrian entries. Pedestrian entries are much more common in Mexico border. Third most common type is Trucks and that type of crossing is more common at Canadian border. We can also say that most of the trucks entering U.S from Mexico for importing purposes because they seem to enter with empty truck containers meanwhile on Canadian border it's quite opposite...**

# In[ ]:


# Entries by measure type:

measure_data = data.groupby(['Border', 'Measure'])['Total Entries'].agg(
    ['sum']).reset_index().sort_values('sum', ascending=False).rename(columns={'sum': 'Total Entries'})

fig = plt.subplots(figsize=(30, 15))

sns.barplot(x='Total Entries', y='Measure', hue='Border',
            data=measure_data, palette='plasma')

plt.legend(loc='lower right', fontsize='xx-large')

plt.show()


# In[ ]:


data_m = data.groupby('Measure')['Total Entries'].agg(
    ['sum']).sort_values('sum', ascending=False).reset_index().rename(columns={'sum': 'Total Entries'})

data_m.head(50)


# In[ ]:


per_d = data_m.pivot_table(index=None, columns='Measure',
                           values='Total Entries').reset_index().drop('index', axis=1)

per_d['Per Bus'] = per_d['Bus Passengers'] / per_d['Buses']

per_d['Per Train'] = per_d['Train Passengers'] / per_d['Trains']

per_d['Per Car'] = per_d['Personal Vehicle Passengers'] /     per_d['Personal Vehicles']


# # Average Passenger Entries Per Vehicle Type

# In[ ]:


# Passenger entries per vehicle type:

per_dc = per_d.unstack().reset_index(name='Total Entries').drop('level_1', axis=1)

sns.barplot(x='Measure', y='Total Entries', data=per_dc.loc[(per_dc['Measure'] == 'Per Bus') | (
    per_dc['Measure'] == 'Per Train') | (per_dc['Measure'] == 'Per Car')], palette='plasma')

plt.show()


# # Entries by President
# 
# **When we look at the president data we can see most of the entries happened 8 years of Bush administration even with the decline in 2001-2002 followed by Obama's 8 year term. We don't have enough data for Trump's term to make assumptions yet...**

# In[ ]:


# Mapping the presidents:

conditions = [(data['Date'] <= '1997-01-20'),
              (data['Date'] >= '1997-01-20') & (data['Date'] < '2001-01-20'),
              (data['Date'] >= '2001-01-20') & (data['Date'] < '2005-01-20'),
              (data['Date'] >= '2005-01-20') & (data['Date'] < '2009-01-20'),
              (data['Date'] >= '2009-01-20') & (data['Date'] < '2013-01-20'),
              (data['Date'] >= '2003-01-20') & (data['Date'] < '2017-01-20'),
              (data['Date'] >= '2017-01-20')]
choices = [
    'Clinton 1st Term (Last Year)', 'Clinton 2nd Term', 'G. Bush 1st Term',
    'G. Bush 2nd Term', 'Obama 1st Term', 'Obama 2nd Term',
    'Trump 1st Term (First 3 Years)'
]

data['President'] = np.select(conditions, choices, default='None')


# In[ ]:


# Entries by president.

pres_data = data.groupby(['President', 'Border'])['Total Entries'].agg(
    ['sum']).reset_index().sort_values('President').rename(
        columns={'sum': 'Total Entries'})

fig = plt.subplots(figsize=(30, 15))

sns.barplot(x='Total Entries',
            y='President',
            hue='Border',
            data=pres_data,
            palette='plasma')

plt.legend(fontsize='xx-large')


# # One Last Experimental Correlation Map:
# **I'm not sure about this graph but I just wanted to go with it and we can observe some logical correlation with some types of transporting...**
# 

# In[ ]:


# Categorical correlation:

msk = data[['Date', 'Measure', 'Total Entries']].set_index('Date')

month_measure = msk.pivot_table(
    'Total Entries', index=msk.index.month, columns='Measure', aggfunc='mean')


fig, axes = plt.subplots(figsize=(12, 12))

sns.heatmap(month_measure.corr(),
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            square=True,
            linewidths=1)

plt.show()


# # Some Final Words
# 
# **I'm still a beginner and want to improve myself in every way I can. So if you have any ideas to feedback please let me know in comments, and if you liked my work please leave a like...**
# 
# **Thank you!**
