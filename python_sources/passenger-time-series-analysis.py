#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# Change default matplotlib figsize
plt.rcParams['figure.figsize'] = (20, 10)


# - [1. Preprocessing the dataset](#preprocessing)
# - [2. Exploratory Analysis](#exploratory)
# 
#     - [2.1. Autocorrelation](#autocorr)
#     - [2.2. Time Series Decomposition](#decompose)
#         - [2.2.1. Removing Trend using Linear Regression](#trend-linear)
#         - [2.2.2. Removing Trend using Moving Averages](#trend-ma)
#         - [2.2.3. Classical Decomposition](#classical)

# # [1. Preprocessing the Dataset](#preprocessing)

# In[ ]:


dataset_path = '/kaggle/input/russian-passenger-air-service-20072020/russian_passenger_air_service_2.csv'


# In[ ]:


df = pd.read_csv(dataset_path)


# In[ ]:


df.head()


# In[ ]:


print('Number of rows:', df.shape[0])
print('Number of Airports:', df['Airport name'].nunique())
print('First Year:', df['Year'].min())
print('Last Year:', df['Year'].max())


# #### Create time index

# The original dataset is a little anoying to work with, because the `Year` is a column but each month is a new one columns (e.g. `January`, `February`). Instead of using it like it is, let's create a datetime field and unpivot the dataframe to a long format using the [`pandas.melt()`](https://pandas.pydata.org/docs/reference/api/pandas.melt.html).

# In[ ]:


months = df.columns[~df.columns.isin([
    'Airport name',
    'Airport coordinates',
    'Whole year', 'Year'
])]


# In[ ]:


# Mapping (e.g. January = 1, February = 2, ...)
mapping = {v: k for k,v in enumerate(months, start=1)} 


# In[ ]:


time_series = df.melt(
    id_vars=['Airport name', 'Year'],
    value_vars=months,
    var_name='Month'
)


# In[ ]:


time_series['date'] = time_series.apply(lambda x: f"{x['Year']}-{mapping[x['Month']]:02d}", axis=1)


# In[ ]:


time_series['date'] = pd.to_datetime(time_series['date']) # Covert type


# In[ ]:


time_series = (
    time_series
    .rename(columns={'Airport name': 'airport', 'value': 'passengers'})
    .drop(columns=['Year', 'Month'])
)


# In[ ]:


time_series.info()


# For now, let's ignore the data for each airport and explore the aggregated time series from 2007 and 2019.

# In[ ]:


by_month = time_series.groupby('date').sum().loc[:'2019-12-01'] 


# In[ ]:


by_month.head()


# # [2. Exploratory Analysis](#exploratory)

# In[ ]:


fig = by_month.plot(title='Passengers by month')


# ## [2.1. Autocorrelation](#autocorr)

# In[ ]:


from statsmodels.graphics.tsaplots import plot_acf


# Autocorrelation measures the **linear relationship between lagged values** of the time series.  
# We can clearly see a seasonal pattern in the data because of the 12 period autocorrelation.

# In[ ]:


fig = plot_acf(by_month['passengers'], lags=32)


# ## [2.2. Time Series Decomposition](#decompose)

# In[ ]:


dec_df = by_month.copy() # Copy the DataFrame


# ### [2.2.1. Removing Trend using Linear Regression](#trend-linear)

# In[ ]:


from sklearn.linear_model import LinearRegression


# Let's model the trend using the _sklearn_ [`sklearn.LinearRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) class to perform a linear regression using Ordinary Least Squares (OLS). This could alse be done using the _statsmodel's_ OLS implementation: [`statsmodels.api.OLS`](https://www.statsmodels.org/devel/generated/statsmodels.regression.linear_model.OLS.html).

# In[ ]:


x, y = dec_df.index.values.reshape(-1, 1), dec_df.values.reshape(-1, 1)


# In[ ]:


model = LinearRegression()
model = model.fit(x, y)


# In[ ]:


print('Coefficient:', model.coef_[0][0])
print('Independent term:', model.intercept_[0])


# In[ ]:


dec_df['linear_trend'] = model.predict(x.astype('float').reshape(-1, 1))


# #### Additive ($y_t - Trend_t$)

# In[ ]:


dec_df['linear_detrended'] = dec_df['passengers'] - dec_df['linear_trend']


# In[ ]:


fig = dec_df[[
    'passengers',
    'linear_trend',
    'linear_detrended'
]].plot(subplots=True, title='Additive with Linear Regression')


# #### Multiplicative ($y_t * Trend_t^{-1}$)

# In[ ]:


dec_df['linear_detrended'] = dec_df['passengers'] / dec_df['linear_trend']


# In[ ]:


fig = dec_df[[
    'passengers',
    'linear_trend',
    'linear_detrended'
]].plot(subplots=True, title='Multiplicative with Linear Regression')


# ### [2.2.2. Removing Trend using Moving Averages](#trend-ma)

# In[ ]:


# Calculate a 12-period Simple Moving Average (SMA)
dec_df['ma_trend'] = dec_df['passengers'].rolling(12).mean()


# #### Additive ($y_t - Trend_t$)

# In[ ]:


dec_df['ma_detrended'] = dec_df['passengers'] - dec_df['ma_trend']


# In[ ]:


fig = dec_df[[
    'passengers',
    'ma_trend',
    'ma_detrended'
]].plot(subplots=True, title='Additive with Moving Average')


# #### Multiplicative ($y_t * Trend_t^{-1}$)

# In[ ]:


dec_df['ma_detrended'] = dec_df['passengers'] / dec_df['ma_trend']


# In[ ]:


fig = dec_df[[
    'passengers',
    'ma_trend',
    'ma_detrended'
]].plot(subplots=True, title='Multiplicative with Moving Average')


# ### [2.2.3. Classical Decomposition](#classical)

# In[ ]:


from statsmodels.tsa.seasonal import seasonal_decompose


# In[ ]:


decomposition = seasonal_decompose(
    x=by_month.passengers,
    model='multiplicative',
    period=12
)


# In[ ]:


fig = decomposition.plot() # Plot components

