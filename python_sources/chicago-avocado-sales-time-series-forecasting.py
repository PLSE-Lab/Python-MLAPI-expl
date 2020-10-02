#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import itertools
sns.set()

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('/kaggle/input/avocado.csv')
df.head()


# In[ ]:


df.info()


# The first colummn of our dataset is Unnamed: 0 and looks like Index column. We don't really need two index columns in our dataset. Therefore, we will delete Unnamed column.

# In[ ]:


df = df.drop(['Unnamed: 0'], axis = 1)


# As we can see from the info() function that our date column datatype is object not date/time. We need to convert it to date/time type - so we can easily do an analysis.

# In[ ]:


df["Date"] = pd.to_datetime(df["Date"])


# There are 54 unique regions in this dataset. It is strange but all of the regions in this dataset have the same value 338 except of WestTexNewMexico that has only 335.

# In[ ]:


print(df['region'].nunique())
df['region'].value_counts().tail()


# Let's take a look at avocado sales in Chicago - the city I live in

# In[ ]:


Chicago_df = df[df['region']=='Chicago']
Chicago_df.head()


# Let's us set the "Date' column as an index.

# In[ ]:


Chicago_df.set_index('Date', inplace = True)


# If we look carefully at the dates of the dataset, we will notice that dates are not in order. So let's sort the dates and plot average price over the years 2015-2018.

# In[ ]:


Chicago_df = Chicago_df.sort_values(by = 'Date')
Chicago_df.head()


# Let's take a look on a plot of an average avocado price during 2015 - 2018 in Chicago. From the plot below, we can clearly see a huge spike in avocado prices in November 2016. This graph looks quite crazy with its everyday sharp spikes and valleys. 

# In[ ]:


Chicago_df['AveragePrice'].plot(figsize = (18,6))


# We will use rolling average function in order to smooth our crazy fluctuations (by the way I am partially using analysis done by Sentdex at the following Youtube video https://www.youtube.com/watch?v=DamIIzp41Jg&t=263s). Our plot looks much better now.

# In[ ]:


Chicago_df['AveragePrice'].rolling(25).mean().plot(figsize = (18,6))


# The maximum average price for organic avocado in Chicago was equal to $ 2.3, which happened twice in November 2016. Notice there were only 3,000 total bags.

# In[ ]:


print(Chicago_df['AveragePrice'].max())
Chicago_df[Chicago_df['AveragePrice']==2.3]


# The least average price for conventional avocado in Chicago was in February 2017. Notice there were more than 105,000 total bags.

# In[ ]:


print(Chicago_df['AveragePrice'].min())
Chicago_df[Chicago_df['AveragePrice']==0.7]


# In[ ]:


#Chicago_df[Chicago_df.index=='2017-02-05']


# **Let's do time series decomposition to see trend, seasonality, and noise.**

# We have 3 years and 1 quarter of avocado sales data in Chicago.

# In[ ]:


print(Chicago_df.index.min())
print(Chicago_df.index.max())


# **Data Preprocessing**
#     We will remove columns that we don't need, check for missing values, and do an avocado prices forecast.

# In[ ]:


columns = ['Total Volume','4046','4225','4770','Total Bags','Small Bags','Large Bags','XLarge Bags','type',
           'year','region']
Chicago_df.drop(columns, axis =1, inplace=True)


# In[ ]:


Chicago_df.isnull().sum()


# Now our dataset looks nice and clean with only two columns - index and Average price and zero null values.

# In[ ]:


Chicago_df = Chicago_df.groupby('Date')['AveragePrice'].sum().reset_index()


# In[ ]:


Chicago_df = Chicago_df.set_index('Date')
Chicago_df.index


# In[ ]:


y = Chicago_df['AveragePrice']


# As we have seen from the graphs above that there is a pattern when we plot the data. We want to do a time-series decomposition that will decompose our data into three distinct components: trend, seasonality, and noise.

# In[ ]:


from pylab import rcParams
rcParams['figure.figsize'] = 18,8
decomposition = sm.tsa.seasonal_decompose(y, model = 'additive')
fig = decomposition.plot()
plt.show()


# The plot above clearly shows that Chicago avocado sales exzibit a strong seasonality with a peak in November and an increasing trend, which means that people love avocado.

# We will apply ARIMA model or Autoregressive Integrated Moving Average. They have 3 parameters ARIMA(p, d, q). They account for seasonality, trend, and noise in the data.

# In[ ]:


p = d = q = range(0,2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p,d,q))]


# In[ ]:


print('Examples of parameter control combinations for Seasonal ARIMA...')
print('SARIMAX:{} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX:{} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX:{} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX:{} x {}'.format(pdq[2], seasonal_pdq[4]))


# In[ ]:


# fix the error
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y, order=param, seasonal_order=param_seasonal,
                                            enforce_stationarity=False, enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue


# The following output suggests that SARIMAX(1, 1, 0)x(0, 1, 1, 12) with the following AIC:0.409 is our optinal model.

# In[ ]:


mod = sm.tsa.statespace.SARIMAX(y,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 0, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])


# We should always run model diagnostics to investigate any unusual behavior.

# In[ ]:


results.plot_diagnostics(figsize =(16,8))
plt.show()


# Our model looks pretty good with the residuals being normally distributed.

# **Forecast Validating**

# We will compare predicted sales to real sales of the time series to understand the accuracy of our forecast.
# We start forecasting at 2017-01-01 and to the end of the data.

# In[ ]:


pred = results.get_prediction(start=pd.to_datetime('2017-01-01'), dynamic=False)
pred_ci = pred.conf_int()
ax = y['2015':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color = 'k', alpha = 0.2)
ax.set_xlabel('Date')
ax.set_ylabel('Avocado Sales')
plt.legend()
plt.show()


# Our forecast looks good following overall trend pretty closely. It closely alligns with true values.

# In[ ]:


y_forecasted = pred.predicted_mean
y_truth = y['2017-01-01':]
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))


# In[ ]:


print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))


# Both errors are very small and non-negative that suggest that we found a good model for the avocado price forecasting in Chicago.

# **Visualizing of the Avocado Price Forecasting in Chicago**

# In[ ]:


pred_uc = results.get_forecast(steps=100)
pred_ci = pred_uc.conf_int()
ax = y.plot(label='observed', figsize=(14, 7))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Avocado Sales')
plt.legend()
plt.show()


# **Our forecast has a decreasing trend and doesn't look to be seasonal. So we will do time series forecasting using Prophet and compare two models later.**

# In[ ]:





# In[ ]:




