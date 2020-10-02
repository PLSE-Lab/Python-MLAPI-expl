#!/usr/bin/env python
# coding: utf-8

# # Import & Clean Data
# Let us read the State Time Series data and do some basic analysis. The data is available from 2010 onwards, so let us remove the previous data from the data set.

# In[ ]:


import pandas as pd
df = pd.read_csv('../input/State_time_series.csv')
df.Date = pd.to_datetime(df.Date)
df = df[df['Date'] >= '01-01-2010']
df.head()


# # Exploratory Data Analysis (EDA)
# Let us explore statewise data. Whether there is any trend for a specific state or what are the top 5 states?

# ## Top 7 States of Median Price Per Sqft
# This is the top 7 states of Median Listring Price per Sqft All Homes

# In[ ]:


dfallhomes = df.groupby('RegionName', as_index=False)['MedianListingPricePerSqft_AllHomes'].    mean().dropna().sort_values('MedianListingPricePerSqft_AllHomes', ascending=False)
dfallhomes.head(7)


# ## Bottom 5 states of Median Listing Price per Sqft All Homes
# List the bottom 5 states from the results

# In[ ]:


dfallhomes.tail(5)


# ## Trend in State Time Series
# The most important EDA on time series data is to identify trend, seasonality & correlation. Let us check whether there is any trend in the data with the Top state data.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
statelist = ['Hawaii', 'DistrictofColumbia', 'California', 'Massachusetts', 'NewYork', 'Colorado']
stateseries = pd.DataFrame(df[(df['RegionName'].    isin(statelist))][['Date','RegionName','MedianListingPricePerSqft_AllHomes']].    dropna().    groupby(['Date', 'RegionName'])['RegionName','MedianListingPricePerSqft_AllHomes'].mean().unstack())
stateseries.plot(figsize=(15,8), linewidth=3)
plt.show()


# from the above graph, we can see that
# 1. There are trends for Hawaii & District of Columbia, even though it is up & down but there is a strong upward trend on these two states. This also shows that there is a seasonality in the trends. But the investment is costly in these two states.
# 2. There is no point in investing the homes in New York, it's almost stationary.
# 3. The similar trend of top 2 states is reflecting in Colorado also. This seems to be the best investment for homes as it is steadily increasing over the years among the top 5.

# ## Seasonality in trends
# Let us check whether there is any seasonality in the trends. This is important for the predictions.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set()
season = df
season['Date'] = df.Date
season['Year'] = df['Date'].dt.year
season['Month'] = df['Date'].dt.month
spivot = pd.pivot_table(season, index='Month', columns = 'Year', values = 'MedianListingPricePerSqft_AllHomes', aggfunc=np.mean)
spivot.plot(figsize=(20,10), linewidth=3)
plt.show()


# from above graph, we can see that
# 1. The price start decreasing in year 2010 and continued till 2012. The price in year 2013 is almost equivalent to 2010.
# 2. The best time to sell the house in a year is from June to October. The price is peak in these months consistently in all the years even during the down trend. Alternatively the best time to buy the house is in December & January.
# 3. The price drops may be due to holiday season or some other reason is a problem for us to solve another day.

# ## Correlation
# I always have a doubt whether various bedroom types have any relation during the trends. Let's find out.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

brtypes = df.groupby('Date')['Date','MedianListingPricePerSqft_1Bedroom', 'MedianListingPricePerSqft_2Bedroom','MedianListingPricePerSqft_3Bedroom','MedianListingPricePerSqft_4Bedroom','MedianListingPricePerSqft_5BedroomOrMore'].    mean().dropna()
pd.plotting.autocorrelation_plot(brtypes);
plt.show()


# The above graph shows there is a positive correlation for all the bedroom types, but still it is not clear how each bedroom types are correlated. Let us find out.

# In[ ]:


brtypes.corr()


# The above table confirms the correlation and to be more specific let us remove the seasonality from the data and see. This is called order of correlation.

# In[ ]:


brtypes.diff().corr()


# First order difference in correlation still has better correlation between bedroom types. You can see 1 Bedroom & 2 Bedroom are highly correlated than 1 bedroom & 5 bedroom. 

# # Forecast
# We will do the forecast with Median Listing price per sq ft All Homes. Let us see the trend first.

# In[ ]:


allhomes = df.groupby('Date')['Date','MedianListingPricePerSqft_AllHomes'].mean().dropna()
allhomes.plot(figsize=(10,8))
plt.show()


# In the above graph, you can see the clear trend but also there are seasonality in the trend. The forecast for the time series should be stationary otherwise the predictions may not correct.
# 
# ## ARIMA forecast model
# One of the common model used to forecast time series data is ARIMA. It stands for Autoregressive integrated moving average. One of the parameters are p, d & q.  As you know the data has seasonality and let us use Seasonal ARIMA, SARIMAX to forecast the mode. 
# There is a separate process to to identify the optmimum parameters, I did a grid search on GPU machine and it stopped after 700+ iterations. 

# In[ ]:


import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")
mod = sm.tsa.statespace.SARIMAX(allhomes,
                                    order = (2, 0, 4),
                                    seasonal_order = (3, 1, 2, 12),
                                    enforce_stationarity = False,
                                    enforce_invertibility = False)
results = mod.fit()
results.plot_diagnostics(figsize=(15,12))
plt.show()


# From the above graph, we can see that
# 1. The histogram has minor difference with KDE
# 2. Linear regression can be improved
# 3. There are still positive correlation, this can be optimized further.

# ## Validate the Model
# Let us validate the model by Train Test & Split.

# In[ ]:


train_size = int(len(allhomes) * 0.60)
train, test = allhomes[0:train_size], allhomes[train_size:]

pred = results.get_prediction(start = test.iloc(train_size)[0].name, dynamic = False)
pred_ci = pred.conf_int()

ax = allhomes.plot(label='actual', figsize=(10,8))
pred.predicted_mean.plot(ax=ax, label='Forecast', alpha=0.7, color='red')
plt.legend()
plt.show()


# The predictions above is not perfect but it is still better compare to other parameters. Let us see the Mean Squared Error of the model.

# In[ ]:


from sklearn.metrics import mean_squared_error
error = mean_squared_error(test, pred.predicted_mean)
print('MSE {}'.format(error))


# The mean squared error is 0.170, so it means the forecast can be till improved. Now let us predict for the future.

# In[ ]:


pred_uc = results.get_forecast(steps=24)

pred_ci = pred_uc.conf_int()

ax = allhomes.plot(label = 'Actual', figsize=(15,8))
pred_uc.predicted_mean.plot(ax=ax, label='Forecasted')
ax.fill_between(pred_ci.index,
                   pred_ci.iloc[:,0],
                   pred_ci.iloc[:,1],
                   color='k', alpha=0.25)
ax.set_xlabel('Date')
ax.set_ylabel('MedianListingPricePerSqft_AllHomes')
plt.legend()
plt.show()


# In the above graph you can see the forecast for next 24 months and the confidence interval is also better and not too much variance.

# In[ ]:




