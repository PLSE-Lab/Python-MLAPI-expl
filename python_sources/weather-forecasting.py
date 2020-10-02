#!/usr/bin/env python
# coding: utf-8

# # Weather Forecast Modelling
#    
# Using Facebook Prophet's open source forecasting library, Brisbane weather has been forecasted.
# 
# Prophet is an additive regression model that has intelligent forecasting methods out of the box. It is designed to operate on daily basis and factors in holiday effects, seasonality etc.
# 
# The library allows powerful forecasting without a significant amount of statistical tweaking, such as required in more heavy methods, such as ARIMA models.

# In[ ]:


# Check files
get_ipython().system('ls ../input/*')


# In[ ]:


# Set up modules/libraries
import pandas as pd
import numpy as np
from pandasql import sqldf
sql = lambda q: sqldf(q, globals())

# Data Viz libraries
import matplotlib.pyplot as plt # basic plotting
import seaborn as sns # for prettier plots
import plotly.graph_objects as go

# Config plot style sheets
# https://matplotlib.org/3.1.0/gallery/style_sheets/style_sheets_reference.html
plt.style.use('fivethirtyeight')
pd.plotting.register_matplotlib_converters()

# suppress warnings
import warnings
warnings.filterwarnings("ignore")


# Data from Kaggle
# https://www.kaggle.com/selfishgene/historical-hourly-weather-data
# 
# 

# In[ ]:


#import BOM dataset from Kaggle
weather_AU = pd.read_csv('../input/weather-dataset-rattle-package/weatherAUS.csv')
weather_AU.head(5)


# In[ ]:


weather_AU.describe().transpose()


# In[ ]:


print('min', weather_AU['Date'].min())
print('max', weather_AU['Date'].max())


# We can see that the dataset while having more data points, only goes back to 2007 compared to getting data directly from BOM, so therefore we will stick with BOM observation data.
# 
# Data also from BOM 
# Courtesy of http://www.bom.gov.au/climate/data/

# In[ ]:


# Import Australian BOM weather observation data
# Ingest daily maximum temperature for Sydney, Brisbane and Melbourne
weather_brisbane = pd.read_csv('../input/bom-weather-observation-data-select-stations/IDCJAC0010_040913_1800_Data.csv') # 040913 is Brisbane

weather_brisbane.head(10)


# # Basic Exploratory Data Analysis (EDA)
# First step in forecasting is getting to know the data - this involves knowing the spread, how many null values etc.

# In[ ]:


# Let's see spread of data
weather_brisbane.describe().transpose()


# First let's clean up the columns and check for missing values

# In[ ]:


# See how many missing values
weather_brisbane[weather_brisbane.isna().any(axis=1)]


# So we see we have data from 1999 to 2019, with a relatively good spread of temperature data.

# In[ ]:


# Let's get rid of the missing values - get rid of the row if any NaNs
weather_brisbane.drop(columns=['Bureau of Meteorology station number', 'Product code', 'Days of accumulation of maximum temperature', 'Quality'], inplace=True)

weather_brisbane.dropna(axis=0, how='any', inplace=True)

weather_brisbane.head(10)


# We need to create a date field by combining all the year, month, and day columns together

# In[ ]:


weather_brisbane['Date'] = pd.to_datetime(weather_brisbane[['Year', 'Month', 'Day']])

weather_brisbane.drop(columns=['Year', 'Month', 'Day'], inplace=True)

weather_brisbane.head(10)


# In[ ]:


# Let's set index to be datetime so we can filter easily
weather_brisbane.set_index('Date', inplace=True)


# As part of EDA, given this is weather data, seasonality is expected.
# 
# Therefore, we need to look at autocorrelation - which is how the same time in different days correlate with each other - 
# e.g. 9am March compared to 9am April, as well as 9am March 2019 compared to 9am March 2018.
# 
# If a correlation exists between months, this is monthly seasonality. If between years, yearly seasonality etc.

# In[ ]:


weather_brisbane.dtypes


# In[ ]:


# Given forecast weather, let's have a look at autocorrelation at the averaged to the monthly level
from pandas.plotting import autocorrelation_plot

weather_brisbane_monthly = weather_brisbane.resample('M').mean()

autocorrelation_plot(weather_brisbane_monthly['Maximum temperature (Degree C)'])


# We can see that there is a correlation around every 12 months (which makes sense due to seasons)

# Let's have a look at the data in an interactive way - you can drag the bottom bar to change the range

# In[ ]:


# Show interactive plot limited to date range
fig = go.Figure()
fig.add_trace(go.Scatter(x=weather_brisbane.index
                         ,y=weather_brisbane['Maximum temperature (Degree C)']
                         ,name='Weather - Brisbane observation for Max Temp (c)'
                         ,line_color='deepskyblue'
                         )
             )
fig.update_layout(title_text='Interactive - Brisbane weather max temperature'
                  ,xaxis_range=['1999-01-01','2019-12-31']
                  ,xaxis_rangeslider_visible=True)
fig.show()


# There's obvious seasonality and a nice spread of data

# # More comprehensive EDA using Pandas Profiling
# Now let's do more profiling using Pandas profiling - a library that does a lot of summaries for you out of the box

# In[ ]:


# More comprehensive profiling

# 3 aspects of EDA that it captures:
# 1. Data Quality - ie df.dtypes and df.describe
# 2. Variable relationship - Pearson correlation - sns.heatmap(df.corr().annot=True)
# 3. Data spread - mean, std dev, median, min, max, histograms - sns.boxplot('variable', data=df)

import pandas_profiling
weather_brisbane.profile_report(style={'full_width':True})


# # Forecasting using Prophet
# 
# Now that the data is ready, let's first separate the observation data into two categories (train and test)
# 
# As time-series data, we won't use a random splitter (e.g. scikit-learn train test split), but just set a cutoff date.
# 
# In this case, we will aim for 80% train and 20% test. This will be set using the Prophet's changepoint range to 0.8.

# In[ ]:


print("min", weather_brisbane.index.min())
print("max", weather_brisbane.index.max())


# Now the Facebook Prophet library requires the format to be time-series with 2 columns - ds for datetime and y for values

# In[ ]:


train = weather_brisbane

train.reset_index(inplace=True) # Reset index
train.rename(columns={'Date': 'ds', 'Maximum temperature (Degree C)': 'y'}, inplace=True)

train.head(10)


# # Facebook Prophet
# We are now ready to do the training and fitting the model using Prophet

# In[ ]:


# Create Prophet model - and fit training data to the model
# We set changepoint range to 80% and MCMC sampling to 100 - MCMC sampling adds uncertainty interval to seasonality
from fbprophet import Prophet
model = Prophet(changepoint_range=0.8, mcmc_samples=100)

model.fit(train)


# In[ ]:


# Using helper, create next forecast range to be next 2 years (720 days) - 
# aggregated to daily basis (since we only have daily readings)
from fbprophet import Prophet
future = model.make_future_dataframe(periods=720, freq='D')

future.tail()


# In[ ]:


# Make prediction using - show confidence interval 
# By default, Prophet uses Monte Carlo Markov Chain sampling to create confidence interval - 
# which covers 80% (not 95%) of the samples
forecast = model.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]


# In[ ]:


# Plot forecast
# Plot interactive plotly viz
from fbprophet.plot import plot_plotly
import plotly.offline as py
py.init_notebook_mode()

fig = plot_plotly(model, forecast)  # This returns a plotly Figure

fig.update_layout(title_text='Interactive - Brisbane weather max temperature with forecast')

py.iplot(fig)


# The above shows:
# 
# black dots are the actual observation data points (i.e. actual temp readings)
# 
# blue line represents the model's regression line that it identified
# 
# blue shaded around the blue line indicates the confidence interval - it represents the range that covers 95% of the actual data

# In[ ]:


# Plot components of forecast in detail
model.plot_components(forecast)


# The above shows daily, monthly and yearly trends of seasonality.
# 
# # Observations on forecasting
# Seasonality has been correctly identified by the model - it is clear that Brisbane's winter (July) has lowest temperature of the year.
# 
# There is a general trend of yearly average temperature increasing, consistent with a global warming pattern.
# 
# Interestingly, Thursday seems to be the coldest day in the week.

# # Measuring performance of Forecast Model
# We will grab the last 365 days of actual historical data to evaluate the model.
# 
# THis means the horizon will be 365 days.
# 
# We will have evaluations every month so the spacing between cutoff periods will be 180 days.
# 
# In particular, a forecast is made for every observed point between cutoff and cutoff + horizon. That is, model is trained up untll every cutoff point, then re-evaluated every 180 days.

# In[ ]:


# Cross validation between actuals and forecasted data between cutoff and horizon
from fbprophet.diagnostics import cross_validation
df_cv = cross_validation(model, period='180 days', horizon ='365 days')
df_cv.head()


# In[ ]:


# See diagnostics of Prophet
from fbprophet.diagnostics import performance_metrics
df_perf = performance_metrics(df_cv)
df_perf.describe()


# # Observations on Performance
# The Mean Absolute Percentage Error (MAPE) ranges between 7 to 9%, which is relatively low.
# 
# Furthermore, as visualised below, the overall trend of the MAPE stays relatively flat (which indicates that the forecasting does not 'decay' over time).

# In[ ]:


# mean absolute percentage error (MAPE) visualised
from fbprophet.plot import plot_cross_validation_metric
fig = plot_cross_validation_metric(df_cv, metric='mape')


# So in conclusion, using Prophet, we managed to get a pretty good forecasting model of Brisbane weather with a MAPE of less than 10%.
