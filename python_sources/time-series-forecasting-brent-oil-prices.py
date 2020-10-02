#!/usr/bin/env python
# coding: utf-8

# # Time Series Forecasting, Euro Brent Oil Prices
# 
# ### Introduction
# This is a Time-Series forecast of [European Brent Oil Prices (US Dollars per Barrel)](https://www.eia.gov/dnav/pet/hist_xls/RBRTEd.xls) using the Prophet forecasting method.
# 
# #### Breakdown:
# * First data clean and produce a time series visualisation
# * Next, get the data ready for prophet model forecasting
# * Forecast
# * Compare forecast data to actual real-time data
# * Conclusion and Discussion
# --------
# 

# In[ ]:


import pandas as pd
import seaborn as sns
from fbprophet import Prophet
df = pd.read_csv('../input/brent-oil-prices/BrentOilPrices.csv')
df.info()
df.head()


# 1) Some cleaning and visuals

# In[ ]:


df['Date'] = pd.to_datetime(df['Date'], format="%b %d, %Y") #format date data to appropriate format
df.head()


# #### A quick time series line plot:

# In[ ]:


sns.lineplot(x='Date', y='Price', data=df)


# 2) Get the prophet forecast ready

# In[ ]:


#Standard procedure is to rename date column to DS:
df.rename(columns={"Date": "ds", "Price": "y"}, inplace=True)


# Now, let's slice the data into a period where more of a trend is evident. It is worth noting that we have uncertainty in the trend, as we do not know  the potential future trend changes, when they will take place and why they will take place (it is virtually impossible). We therefore can do the best we can in the present time - assume the future price of the oil will see similar trend changes as it's history. 
# 
# Thus, we are going to try and slice the time data as much as we can to where it is evident that more of a trend is occuring - as we can see from the above plot, lets decide to forecast the trend occuring after the year 2004.
# 
# The Prohpet forecast model will detect the changes in the trend, and an uncertainty level is set into this by default (80%) - this increases the forecast uncertainty and gives a "window" for change in future trends. We will alter the uncertainty level to 95% just to further compensate for uncertain trend changes in the future.

# In[ ]:


df = df[df['ds'] > '2004-01-01']


# 3) Forecast

# In[ ]:


m = Prophet()
m.fit(df)
future = m.make_future_dataframe(periods=365) #forecasting 365 days in future
forecast = Prophet(interval_width=0.95).fit(df).predict(future)
m.plot(forecast)


# As we can see the uncertainty level of 95% caters for the price to swing drastically bullish or bearish. However, we can see that the prediction line does follow the trend of the historical data from 2004 and onwards. Let's visualise the graph without the uncertainty bands and from 2016 onwards:

# In[ ]:


dfn = forecast.set_index('ds')[['yhat']].join(df.set_index('ds'))
dfn = dfn[dfn.index > '2016-01-01']


# In[ ]:


ax = sns.lineplot(data = dfn)
sns.set_style('darkgrid')
sns.set_palette('rainbow')


# The prediction from 2016 onwards gets a little shaky as we can see, this can be expected in part with the drastic decrease in trend between 2014 and 2016

# 4) Comparing the forecast to actual (price to-date)

# Now, lets bring in the latest brent oil price data to compare the prediction to actual

# In[ ]:


dftd = pd.read_csv('../input/brent-oil-td/RBRTEd.csv')
dftd['Date'] = pd.to_datetime(dftd['Date'], format="%b %d, %Y") #format date data to appropriate format
dftd.rename(columns={"Date": "ds", "Price": "Price TD"}, inplace=True)
dfx = dfn.join(dftd.set_index('ds'))


# In[ ]:


df['ds'].max() #last record of original df to segregate predicted from actual


# In[ ]:


from matplotlib import pyplot as plt
a = dfx['y']
b = dfx['yhat']
c = dfx['Price TD']
c = dfx[dfx.index > '2019-09-30']

plt.plot(a)
plt.plot(b)
plt.plot(c)
plt.show()


# 5) Conclusion and Discussion
# 
# We can see in our final plot that our forecasting model performed somewhat well. We can draw the following conclusions:
# 
# * The model correctly predicted (the turqoise line) that the price will increase from the date our original data ends (2019-09-30)
# * However, the prediction did not detect the suddon decrease the actual price data experienced (orange line). Possible reasons are that the model did not develope sufficiently after the drastic trend decrease we identified between 2014 - 2016. Moreover, our forecasting 365 day forecast period may be rolled forward too much i.e. the prediction model is not sensitive enough to the data to predict short term fluctuations
