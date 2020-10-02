#!/usr/bin/env python
# coding: utf-8

# # Analyzing Time Series Data
# 
# ### Practice in correcting time series data for seasonality.
# 
# #### Import Packages and Data.

# In[ ]:


import pandas as pd
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = [12.0, 8.0]


# In[ ]:


data_set = pd.read_excel('../input/MinDailyTemps.xlsx')


# In[ ]:


data_set.head(10)


# # Check for null values in data.

# In[ ]:


data_set.isnull().sum()


# # Check what types the data are.

# In[ ]:


data_set.dtypes


# # Set the dates to be the index.

# In[ ]:


data_set = data_set.set_index('Date')


# # Peek at the head and tail of the data.

# In[ ]:


data_set.head().append(data_set.tail())


# # Plot the data to get an overview.

# In[ ]:


data_set.plot(grid=True)


# # Let's take a closer look by observing the first two year's data.

# In[ ]:


matplotlib.rcParams['figure.figsize'] = [12.0, 8.0]
from datetime import datetime
start_date = datetime(1981, 1, 1)
end_date = datetime(1982, 12, 31)
data_set[(start_date <= data_set.index) & (data_set.index <= end_date)].plot(grid=True)


# There are a couple of models to consider during the Decomposition of Time Series data.
# 1. Additive Model: This model is used when the variations around the trend does not vary with the level of the time series. Here the components of a time series are simply added together using the formula:
# 
#     y(t) = Level(t) + Trend(t) + Seasonality(t) + Noise(t)
# 
# 
# 2. Multiplicative Model: Is used if the trend is proportional to the level of the time series. Here the components of a time series are simply multiplied together using the formula:
# 
#     y(t) = Level(t) * Trend(t) * Seasonality(t) * Noise(t)
#     
#     
# # We will use the additive model.

# In[ ]:


data_set.head()


# - Trend: The increasing or decreasing value in the series. 
# - Seasonality: The repeating short-term cycle in the series. 
# - Noise: The random variation in the series.

# In[ ]:


decompfreq = 365 # for yearly seasonality


# # Import statsmodels which has a tsa (time series analysis) package as well as the sesonal_decompose() function.

# In[ ]:


import statsmodels.api as sm
decomposition = sm.tsa.seasonal_decompose(data_set, 
                                          freq=decompfreq, 
                                          model = 'additive')
fig = decomposition.plot()
matplotlib.rcParams['figure.figsize'] = [9.0, 5.0]


# # Plot the trend alongside the observed time series. To do this, we will use Matplotlib's .YearLocator() function to set each year to begin from the month of January month=1 and month as the minor locator showing ticks for every 3 months (intervals=3). Then we plot our dataset (and gave it blue color) using the index of the dataframe as x-axis and the temperatures as the y-axis. 
# # We did the same for the trend observations which we plotted in red color.

# In[ ]:


import matplotlib.pyplot as plt
import matplotlib.dates as mdates
fig, ax = plt.subplots()
ax.grid(True)
year = mdates.YearLocator(month=1)
month = mdates.MonthLocator(interval=3)
year_format = mdates.DateFormatter('%Y')
month_format = mdates.DateFormatter('%m')
ax.xaxis.set_minor_locator(month)
ax.xaxis.grid(True, which = 'minor')
ax.xaxis.set_major_locator(year)
ax.xaxis.set_major_formatter(year_format)
plt.plot(data_set.index, data_set['Temp'], c='blue')
plt.plot(decomposition.trend.index, decomposition.trend, c='red')


# # When we plotted just the trend line, the graph seemed to indicate that the temperature was increasing slightly through the decade. But when comparing the trend line to the plot of temperatures, the rise seems insignificant. Of course, more than a decade and more than one location would be needed to truely make any conclusions.
