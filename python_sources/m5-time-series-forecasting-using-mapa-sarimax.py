#!/usr/bin/env python
# coding: utf-8

# # Introduction

# In this notebook, we will use Walmart's M5 competition database and forecast sales for each product category using the MAPA - Multiple Aggregation Forecast Algorithm technique. Temporal aggregation can help to identify the characteristics of the series as they are improved
# through different frequencies (daily, weekly, monthly, bimonthly, for example). The objective is to show that the use of various levels of temporal aggregation can lead to substantial improvements in terms of performance forecasting.

# ## MAPA - Multiple Aggregation Prediction Algorithm
# 
# As the resources of the time series change with the frequency of the data (or the level of aggregation), different methods will be identified as ideal. This will produce different predictions, which will lead to different decisions. In essence, we have to deal with the "true" uncertainty of the model, which is the adequacy or incorrect specification of the model identified as ideal at a specific level of data aggregation. [1]
# The idea of MAPA technique is to reduce the risk of overffiting and select an incorrect model. According to [1], the MAPA technique provides better forecasting performance when compared to traditional approaches.
# 
# <img src="https://www.researchgate.net/profile/Nikolaos_Kourentzes/publication/265069129/figure/fig1/AS:295915979067408@1447563275667/The-standard-versus-the-MAPA-forecasting-approach.png" alt="some text">
# 
# There are three steps:
# <ul>
#   <li><b>Aggregation</b>.  the MAPA uses multiple
# instances of the same data, which correspond to the different frequencies or aggregation levels. Here we use the levels daily, weekly, monthly and bimonthly of aggregation.</li>
#   <li><b>Forecasting</b>. Each one of the series calculated in step 1 should be forecasted separately (using SARIMAX). Seasonality and various high frequency components are expected to be modelled better at lower aggregation levels (such as monthly data), while the long-term trend will be better captured as the frequency is decreased (bimonthly data)</li>
#   <li><b>Combination</b>. The final step in the proposed MAPA approach concerns the appropriate combination of different forecasts derived from alternative frequencies. Before we can combine the predictions produced at the various frequencies, we need to turn them back to the original frequency. Then, we make the combination using functions such as mean, median, maximum or minimum of time series aggregations.</li>
# </ul>
#  [1] PETROPOULOS, Fotios; KOURENTZES, Nikolaos. Improving forecasting via multiple temporal aggregation. Foresight: The International Journal of Applied Forecasting, v. 34, p. 12-17, 2014.

# # Contents
# 
# * [<font size=4>The dataset</font>](#1)
# 
# 
# * [<font size=4>Pre-processing and Exploratory Data Analysis</font>](#2)
#     * [Load the data](#2.1)
#     * [Resampling of time series](#2.2)
#     * [Decomposition](#2.3)
#     * [Autocorrelation Test](#2.4)
#     * [Stationarity test](#2.5)
# 
#     
# * [<font size=4>SARIMAX Forecasting</font>](#3)
#     * [Foods Category forecasting](#3.1)
#     * [Hobbies Category forecasting](#3.2)
#     * [Household Category forecasting](#3.3)
# 
# * [<font size=4>MAPA farecasting (Weekly, monthly and Bimonthly frequencies)</font>](#4)
#     * [Foods Category forecasting](#4.1)
#     * [Hobbies Category forecasting](#4.2)
#     * [Household Category forecasting](#4.3)
# 
# * [<font size=4>Daily forecasting with exogenous variables</font>](#5)
# * [<font size=4>MAPA farecasting (Daily, Weekly, and monthly frequencies)</font>](#6)

# # The dataset <a id="1"></a>
# 
# We will use two dataset .csv files:
# 
# * <code>calendar.csv</code> - Contains the dates on which products are sold. The dates are in a <code>yyyy/dd/mm</code> format.
# 
# * <code>sales_train_validation.csv</code> - Contains the historical daily unit sales data per product and store <code>[d_1 - d_1913]</code>.
# 

# ### Import libraries

# In[ ]:


import pandas as pd
import numpy as np
import plotly.graph_objs as go #visualization library
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf #autocorrelation test
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller #stationarity test
from statsmodels.tsa.statespace.sarimax import SARIMAX 
from datetime import datetime, timedelta
import seaborn as sns
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Pre-processing and Exploratory Data Analysis <a id="2"></a>

# ### Load the data <a id="2.1"></a>
# > 
# First, we will group the database into the three product categories (Foods, Hobbies and Household). We will use the base 'calendar.csv' to export the dates corresponding to the days of the column of the base 'sales_train_validation.csv'.

# In[ ]:


data = pd.read_csv('../input/m5-forecasting-accuracy/sales_train_validation.csv')

data_dept = data.groupby(['dept_id']).sum() #group sales by department
data_item = data.groupby(['item_id']).sum() #group sales by item_id
data_cat = data.groupby(['cat_id']).sum().T #group sales by category
data_cat['day'] = data_cat.index

data_store = data.groupby(['store_id']).sum()
data_state_id = data.groupby(['state_id']).sum()

calendar = pd.read_csv('../input/m5-forecasting-accuracy/calendar.csv')
data_calendar = calendar.iloc[:, [0, 2,3,4,5,6,7]]

#Merge data_calendar columns related to commemorative data, days of the week, month and year.
data_cat = pd.merge(data_calendar, data_cat, how = 'inner', left_on='d', right_on='day')
data_cat_final = data_cat.iloc[:,[7,8,9]]
data_cat_final.index = data_cat['date']
data_cat_final.index = pd.to_datetime(data_cat_final.index , format = '%Y-%m-%d')
data_cat_final.parse_dates=data_cat_final.index
data_cat_final.head(10)


# Below is an interactive graph of sales by category from January 2011 to April 2016.

# In[ ]:


fig = go.Figure(
    data=[go.Scatter(y=data_cat_final['2011-01':'2016-04'].FOODS, x=data_cat_final.index, name= 'Foods'), 
          go.Scatter(y=data_cat_final['2011-01':'2016-04'].HOBBIES, x=data_cat_final.index, name = 'Hobbies'),
          go.Scatter(y=data_cat_final['2011-01':'2016-04'].HOUSEHOLD, x=data_cat_final.index, name = 'HouseHold')],
    layout=go.Layout(
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
    )
)
fig.update_layout(title_text="Sales by Category")
fig.show()


# We can see that there is a moderate correlation between the three time series.

# In[ ]:


sns.heatmap(data_cat_final[['FOODS','HOBBIES','HOUSEHOLD']].corr(), annot = True,  cbar=False)


# ### Resampling of time series <a id="2.2"></a>
# As we can see, the original database contains daily sales data. We will do a resampling here for weekly, monthly and bimonthly frequencies. I going to ignore the first and last lines, otherwise we would be adding lines in an incomplete period

# In[ ]:


data_cat_final_monthly = data_cat_final.iloc[:,[0,1,2]].resample('M').sum()[2:-1] #mensal resampling
data_cat_final_weekly = data_cat_final.iloc[:,[0,1,2]].resample('W').sum()[8:-1] #weekly resampling
data_cat_final_bimonthly = data_cat_final.iloc[:,[0,1,2]].resample('2M').sum()[1:-1] #bimonthy resamply


# Plot with monthly frequencies.

# In[ ]:


fig = go.Figure(
    data=[go.Scatter(y=data_cat_final_monthly.FOODS, x=data_cat_final_monthly.FOODS.index, name= 'Foods'), 
          go.Scatter(y=data_cat_final_monthly.HOBBIES, x=data_cat_final_monthly.HOBBIES.index, name = 'Hobbies'),
          go.Scatter(y=data_cat_final_monthly.HOUSEHOLD, x=data_cat_final_monthly.HOUSEHOLD.index, name = 'HouseHold')],
    layout=go.Layout(
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
    )
)
fig.update_layout(title_text="Sales by Category - Monthly")
fig.show()


# ### Decomposition <a id="2.3"></a>
# 
# The decomposition of time series is a statistical task that deconstructs a time series into several components, each representing one of the underlying categories of patterns. Analyzing the decomposition of the time series of the category in monthly frequency, we can verify a semiannual seasonality with an approximately linear trend

# #### Foods Category - Decomposition

# In[ ]:


decomposed = sm.tsa.seasonal_decompose(np.array(data_cat_final_monthly.FOODS),period=6) # The frequency is semestral
figure = decomposed.plot()


# #### Hobbies Category - Decomposition

# In[ ]:


decomposed = sm.tsa.seasonal_decompose(np.array(data_cat_final_monthly.HOBBIES),period=6) # The frequency is semestral
figure = decomposed.plot()


# In[ ]:


#### Household Category - Decomposition


# In[ ]:


decomposed = sm.tsa.seasonal_decompose(np.array(data_cat_final_monthly.HOUSEHOLD),period=6) # The frequency is semestral
figure = decomposed.plot()


# ### Autocorrelation Test <a id="2.4"></a>
# 
# A plot of the autocorrelation of a time series by lag is called the AutoCorrelation Function, ACF. PACF - partial autocorrelation is the resulting correlation after removing the effect of any correlation due to the terms in shorter intervals. Confidence intervals are drawn as a cone. This is defined as a 95% confidence interval, suggesting that the correlation values outside this code are likely to be a correlation and not a statistical fluke.
# 
# 
# All results below have autocorrelation of 4 to 6 lags and autocorrelation of 2 lags that are statistically significant

# #### ACF and PACF - Foods Category

# In[ ]:


plt.show()
plot_acf(data_cat_final_monthly.FOODS,lags=12,title="ACF Foods")
plt.show()
plot_pacf(data_cat_final_monthly.FOODS,lags=6,title="PACF Foods")
plt.show()


# #### ACF and PACF - Hobbies Category

# In[ ]:


plot_acf(data_cat_final_monthly.HOBBIES,lags=12,title="ACF HObbies")
plt.show()
plot_pacf(data_cat_final_monthly.HOBBIES,lags=12,title="PACF Hobbies")
plt.show()


# #### ACF and PACF - Household Category

# In[ ]:


plot_acf(data_cat_final_monthly.HOUSEHOLD,lags=12,title="ACF Household")
plt.show()
plot_pacf(data_cat_final_monthly.HOUSEHOLD,lags=12,title="PACF Household")
plt.show()


# ### Stationarity test <a id="2.5"></a>
# To check stationarity we use the Augmented Dickey-Fuller test, a type of statistical test called a unit root test.
# 
# If it is not possible to reject the null hypothesis ($p-value > 0.05$), this suggests that the time series has a unit root, which means that it is not stationary. It has some time-dependent structure.
# 
# If the null hypothesis is rejected ($p-value < 0.05$); it suggests the time series does not have a unit root, meaning it is stationary. It does not have time-dependent structure.
# 
# According to the results below, all three series are non-stationary.

# In[ ]:


# Augmented Dickey-Fuller test
adf1 = adfuller(data_cat_final.FOODS, autolag='AIC')
print("p-value of Foods serie is: {}".format(float(adf1[1])))
#The test statistic is negative, we don't reject the null hypothesis (it looks non-stationary).
adf2 = adfuller(data_cat_final.HOBBIES, autolag='AIC')
print("p-value of Hobbies serie is: {}".format(float(adf2[1]))) #It isn't a random walk if p-value is less than 5%
#The test statistic is negative, we don't reject the null hypothesis (it looks non-stationary).
adf3 = adfuller(data_cat_final.HOUSEHOLD, autolag='AIC')
print("p-value of Household serie is: {}".format(float(adf3[1]))) #It isn't a random walk if p-value is less than 5%
#The test statistic is negative, we don't reject the null hypothesis (it looks non-stationary).


# ## SARIMAX Forecasting <a id="3"></a>

# <b>ARIMA</b> (Autoregressive Integrated Moving Average) is a forecasting method for univariate time series data supports both an autoregressive (p) and moving average elements  (q). The integrated element  (d) refers to differencing allowing the method to support time series data with a trend.
# <img src="https://miro.medium.com/max/576/0*Ql_BphTqarSBmgrZ" alt="some text">
# 
# <b>SARIMA</b> (Seasonal Autoregressive Integrated Moving Average)  is an extension of ARIMA that explicitly supports univariate time series data with a seasonal component.
# Trend Elements:
# <ul>
#   <li>p: Trend autoregression order.</li>
#   <li>d: Trend difference order</li>
#   <li>q: Trend moving average order.</li>
# </ul>
# Seasonal Elements:
# <ul>
#   <li>P: Seasonal autoregressive order </li>
#   <li>D: Seasonal difference order</li>
#   <li>Q: Seasonal moving average order</li>
#   <li>m: The number of time steps for a single seasonal period.</li>
# </ul>
# 
# <b>SARIMAX</b> (Seasonal Auto-regressive Integrated Moving Average with exogenous variables) is an extension of SARIMA that supports the use of exogenous variables that can contribute to forecasting performance.

# In[ ]:


#I created a function that should return a forecast and a summary with the main statistics.
#actual - Time series that we will predict
#order - [p, d, q] terms
#seasonal order - [P,S,Q,m] sazonal terms
#t- lag use for the test base and future prediction
def sarimax_predictor(actual, order, seasonal_order, t , start, title):
    mdl = sm.tsa.statespace.SARIMAX(actual[start:-t],
                                            order=order, seasonal_order=seasonal_order,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
    results = mdl.fit()
    results.plot_diagnostics()
    print(results.summary())
    predict = results.predict(start=start, end=len(actual)+t)

    fig = go.Figure(
        data=[go.Scatter(y=actual[0:-t],x=actual[0:-t].index, name= 'Actual'),
          go.Scatter(y=actual[-t-1::],x=actual[-t-1::].index, name= 'Test'),
          go.Scatter(y=predict, x=predict.index, name= 'Predict')],
        layout=go.Layout(
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        )
    )
    fig.update_layout(title_text= title)
    fig.show()
    return predict
#defined function when we need to use exogenous variables
def sarimax_predictor_exog(actual, order, seasonal_order, t, title, exog):
    mdl = sm.tsa.statespace.SARIMAX(actual[0:-t],
                                            order=order, seasonal_order=seasonal_order, exog = exog[0:-t],
                                            enforce_stationarity=False,
                                            enforce_invertibility=False, time_varying_regression = False,
                                            mle_regression = True)
    results = mdl.fit()
    results.plot_diagnostics()
    print(results.summary())
    #use only exogenous to forecasting (test set)
    predict = results.predict(start=0, end=len(actual), exog=exog[-t-1::]) 

    fig = go.Figure(
        data=[go.Scatter(y=actual[0:-t],x=actual[0:-t].index, name= 'Actual'),
          go.Scatter(y=actual[-t-1::],x=actual[-t-1::].index, name= 'Test'),
          go.Scatter(y=predict, x=predict.index, name= 'Predict')],
        layout=go.Layout(
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        )
    )
    fig.update_layout(title_text= title)
    fig.show()
    return predict

#evaluation test
def rmse(actual, predict, title):
    from sklearn import metrics
    rmse = np.sqrt(metrics.mean_squared_error(actual, predict))
    print('The RMSE of ' + title + ' is:', rmse)


# 
# Here we will use a 6 month test base. The model should perform a 1 year forecasting (6 months beyond the test basis). Seasonality is 6 months (28 weeks and 3 months)

# ### Foods Category forecasting <a id="3.1"></a>

# In[ ]:


start = 0
predicted_result_foods_weekly = sarimax_predictor(data_cat_final_weekly.FOODS, [1,1,0], [1,1,0,24], 7*4, start,
                                                  'Weekly forecast - Foods')


# In[ ]:


predicted_result_foods_monthly = sarimax_predictor(data_cat_final_monthly.FOODS, [5,1,1], [1,1,0,6], 6, start,
                                                  'Monthly forecast- Foods')


# In[ ]:


predicted_result_foods_bimonthly = sarimax_predictor(data_cat_final_bimonthly.FOODS, [2,1,0], [1,0,0,6], 3, start,
                                                     'Bimonthly forecast')


# In[ ]:


#RMSE in test fold
rmse_foods_weekly= rmse(data_cat_final_weekly.FOODS[-28::], predicted_result_foods_weekly[-56-1:-28-1], 'weekly Foods - Test')
rmse_foods_monthly= rmse(data_cat_final_monthly.FOODS[-6::], predicted_result_foods_monthly[-12-1:-6-1], 'monthy Foods - Test')
rmse_foods_bimonthly= rmse(data_cat_final_bimonthly.FOODS[-3::], predicted_result_foods_bimonthly[-6-1:-3-1], 'bimonthy Foods - Test')

#RMSE in train fold
rmse_foods_weekly= rmse(data_cat_final_weekly.FOODS[start:-28], predicted_result_foods_weekly[0:-28*2-1], 'weekly Foods - Train')
rmse_foods_monthly= rmse(data_cat_final_monthly.FOODS[start:-6], predicted_result_foods_monthly[0:-6*2-1], 'monthy Foods - Train')
rmse_foods_bimonthly= rmse(data_cat_final_bimonthly.FOODS[start:-3], predicted_result_foods_bimonthly[0:-3*2-1], 'bimonthy Foods - Train')


# ### Hobbies Category forecasting <a id="3.2"></a>

# In[ ]:


predicted_result_hobbies_weekly = sarimax_predictor(data_cat_final_weekly.HOBBIES, [2,1,0], [1,1,0,24], 28, start,
                                                  'Weekly forecast - Hobbies')


# In[ ]:


predicted_result_hobbies_monthly = sarimax_predictor(data_cat_final_monthly.HOBBIES, [2,1,0], [2,0,0,12], 6, start,
                                                  'Monthly forecast- Hobbies')


# In[ ]:


predicted_result_hobbies_bimonthly = sarimax_predictor(data_cat_final_bimonthly.HOBBIES, [2,1,0], [1,0,0,3], 3, start,
                                                     'Bimonthly forecast - Hobbies')


# In[ ]:


#RMSE in test fold
rmse_hobbies_weekly= rmse(data_cat_final_weekly.HOBBIES[-28::], predicted_result_hobbies_weekly[-56-1:-28-1], 'weekly Hobbies - Test')
rmse_hobbies_monthly= rmse(data_cat_final_monthly.HOBBIES[-6::], predicted_result_hobbies_monthly[-12-1:-6-1], 'monthy Hobbies - Test')
rmse_hobbies_bimonthly= rmse(data_cat_final_bimonthly.HOBBIES[-3::], predicted_result_hobbies_bimonthly[-6-1:-3-1], 'bimonthy Hobbies - Test')

#RMSE in train fold
rmse_hobbies_weekly= rmse(data_cat_final_weekly.HOBBIES[start:-28], predicted_result_hobbies_weekly[0:-28*2-1], 'weekly Hobbies - Train')
rmse_hobbies_monthly= rmse(data_cat_final_monthly.HOBBIES[start:-6], predicted_result_hobbies_monthly[0:-6*2-1], 'monthy Hobbies - Train')
rmse_hobbies_bimonthly= rmse(data_cat_final_bimonthly.HOBBIES[start:-3], predicted_result_hobbies_bimonthly[0:-3*2-1], 'bimonthy Hobbies - Train')


# ### Household Category forecasting <a id="3.3"></a>

# In[ ]:


predicted_result_household_weekly = sarimax_predictor(data_cat_final_weekly.HOUSEHOLD, [7,1,0], [1,1,0,24], 28, start,
                                                  'Weekly forecast - Household')


# In[ ]:


predicted_result_household_monthly = sarimax_predictor(data_cat_final_monthly.HOUSEHOLD, [2,1,0], [2,0,0,6], 6, start,
                                                  'Monthly forecast- Household')


# In[ ]:


predicted_result_household_bimonthly = sarimax_predictor(data_cat_final_bimonthly.HOUSEHOLD, [2,0,0], [1,1,0,3], 3, start,
                                                     'Bimonthly forecast - Household')


# In[ ]:


#RMSE in test fold
rmse_household_weekly= rmse(data_cat_final_weekly.HOUSEHOLD[-28::], predicted_result_household_weekly[-56-1:-28-1], 'weekly Household - Test')
rmse_household_monthly= rmse(data_cat_final_monthly.HOUSEHOLD[-6::], predicted_result_household_monthly[-12-1:-6-1], 'monthy Household - Test')
rmse_household_bimonthly= rmse(data_cat_final_bimonthly.HOUSEHOLD[-3::], predicted_result_household_bimonthly[-6-1:-3-1], 'bimonthy Household - Test')

#RMSE in train fold
rmse_household_weekly= rmse(data_cat_final_weekly.HOUSEHOLD[start:-28], predicted_result_household_weekly[0:-28*2-1], 'weekly Household - Train')
rmse_household_monthly= rmse(data_cat_final_monthly.HOUSEHOLD[start:-6], predicted_result_household_monthly[0:-6*2-1], 'monthy Household - Train')
rmse_household_bimonthly= rmse(data_cat_final_bimonthly.HOUSEHOLD[start:-3], predicted_result_household_bimonthly[0:-3*2-1], 'bimonthy Household - Train')


# ## MAPA farecasting (Weekly, monthly and Bimonthly frequencies) <a id="4"></a>

# The time series of MAPA smetimes is out of X axis of original time series. The objective of the alpha function is to find a constanst which when multiplied by the predicted time series, adjusts the x-axis minimizing the RMSE

# In[ ]:


def alpha(actual, predict):
    RMSE =[]
    for i in np.arange(0.5,15, 0.01):
        RMSE.append([i,np.sqrt(metrics.mean_squared_error(actual, predict*i))])
    return np.array(RMSE)[np.argmin(np.array(RMSE)[:,1]),0]


# ### Foods Cateory forecasting <a id="4.1"></a>
# 
# In the combining step, we need to disaggregate the data (from low to high frequency), which is more complicated than aggregation. Here, we will calculate the average we will apply to all periods equally. For example. If in a two-month period we have forecasted 100. The monthly breakdown will be 50 in the two months in the two-month period

# In[ ]:


#step of combination
predicted_result_foods_bimonthly = predicted_result_foods_bimonthly.resample('W').mean()
predicted_result_foods_monthly = predicted_result_foods_monthly.resample('W').mean()
#equally assigns the mean value of the low frequency time series
predictions_foods = pd.DataFrame({'bimonthly': predicted_result_foods_bimonthly.groupby(predicted_result_foods_bimonthly.notnull().cumsum()).transform(lambda x : x.sum()/len(x))[0:-4],
                         'monthly': predicted_result_foods_monthly.groupby(predicted_result_foods_monthly.notnull().cumsum()).transform(lambda x : x.sum()/len(x)),
                         'weekly': predicted_result_foods_weekly[1::]})
prediction_foods_mean = pd.DataFrame.mean(predictions_foods, axis = 1)
prediction_foods_median = pd.DataFrame.median(predictions_foods, axis = 1)
prediction_foods_min = pd.DataFrame.min(predictions_foods, axis = 1)
prediction_foods_max = pd.DataFrame.max(predictions_foods, axis = 1)
alpha_foods_max = alpha(data_cat_final_weekly.FOODS[start+1:-28], prediction_foods_max[start:-28*2-1])
alpha_foods_min = alpha(data_cat_final_weekly.FOODS[start+1:-28], prediction_foods_min[start:-28*2-1])
alpha_foods_mean = alpha(data_cat_final_weekly.FOODS[start+1:-28], prediction_foods_mean[start:-28*2-1])
alpha_median = alpha(data_cat_final_weekly.FOODS[start+1:-28], prediction_foods_median[start:-28*2-1])

fig = go.Figure(
    data=[go.Scatter(y=data_cat_final_weekly.FOODS, x = data_cat_final_weekly.FOODS.index, name= 'Actual'), 
          go.Scatter(y=prediction_foods_min[0:-1]*alpha_foods_min, x= data_cat_final_weekly.FOODS.index, name= 'Predict Min'),
          go.Scatter(y=prediction_foods_max[0:-1]*alpha_foods_max, x= data_cat_final_weekly.FOODS.index, name= 'Predict Max'),
          go.Scatter(y=prediction_foods_mean[0:-1]*alpha_foods_mean, x= data_cat_final_weekly.FOODS.index, name= 'Predict Mean'),
          go.Scatter(y=prediction_foods_median[0:-1]*alpha_median, x= data_cat_final_weekly.FOODS.index, name= 'Predict median')],
    layout=go.Layout(
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
    )
)
fig.update_layout(title_text="Foods Category - MAPA SARIMA forecast")
fig.show()


# 
# The best result was to use a maximum combination function. The result on the test base was closer, but it did not exceed the individual SARIMA model.

# In[ ]:


rmse(data_cat_final_weekly.FOODS[-28::], predicted_result_foods_weekly[-56-1:-28-1], 'weekly Foods - Test')
rmse(data_cat_final_weekly.FOODS[start+1:-28], predicted_result_foods_weekly[start+1:-28*2-1], 'weekly Foods - Train')
#test Fold
rmse_foods_max = rmse(data_cat_final_weekly.FOODS[-28::], prediction_foods_max[-57:-29]*alpha_foods_max, 'Foods Test - max')
rmse_foods_min = rmse(data_cat_final_weekly.FOODS[-28::], prediction_foods_min[-57:-29]*alpha_foods_min, 'Foods Test - min')
rmse_foods_mean = rmse(data_cat_final_weekly.FOODS[-28::], prediction_foods_mean[-57:-29]*alpha_foods_mean, 'Foods Test - mean')
rmse_foods_median = rmse(data_cat_final_weekly.FOODS[-28::], prediction_foods_median[-57:-29]*alpha_median, 'Foods Test - median')
#train Fold
rmse_foods_max = rmse(data_cat_final_weekly.FOODS[start+1:-28], prediction_foods_max[start:-28*2-1]*alpha_foods_max, 'Foods Train - max')
rmse_foods_min = rmse(data_cat_final_weekly.FOODS[start+1:-28], prediction_foods_min[start:-28*2-1]*alpha_foods_min, 'Foods Train - min')
rmse_foods_mean = rmse(data_cat_final_weekly.FOODS[start+1:-28], prediction_foods_mean[start:-28*2-1]*alpha_foods_mean, 'Foods Train - mean')
rmse_foods_median = rmse(data_cat_final_weekly.FOODS[start+1:-28], prediction_foods_median[start:-28*2-1]*alpha_median, 'Foods Train - median' )


# ### Hobbies Category forecasting <a id="4.2"></a>

# In[ ]:


predicted_result_hobbies_bimonthly = predicted_result_hobbies_bimonthly.resample('W').mean()
predicted_result_hobbies_monthly = predicted_result_hobbies_monthly.resample('W').mean()
predictions_hobbies = pd.DataFrame({'bimestral': predicted_result_hobbies_bimonthly.groupby(predicted_result_hobbies_bimonthly.notnull().cumsum()).transform(lambda x : x.sum()/len(x))[0:-4],
                         'mensal': predicted_result_hobbies_monthly.groupby(predicted_result_hobbies_monthly.notnull().cumsum()).transform(lambda x : x.sum()/len(x)),
                         'weekly': predicted_result_hobbies_weekly[1::]})
prediction_hobbies_mean = pd.DataFrame.mean(predictions_hobbies, axis = 1)
prediction_hobbies_median = pd.DataFrame.median(predictions_hobbies, axis = 1)
prediction_hobbies_min = pd.DataFrame.min(predictions_hobbies, axis = 1)
prediction_hobbies_max = pd.DataFrame.max(predictions_hobbies, axis = 1)

alpha_hobbies_max = alpha(data_cat_final_weekly.HOBBIES[start+1:-28], prediction_hobbies_max[start:-28*2-1])
alpha_hobbies_min = alpha(data_cat_final_weekly.HOBBIES[start+1:-28], prediction_hobbies_min[start:-28*2-1])
alpha_hobbies_mean = alpha(data_cat_final_weekly.HOBBIES[start+1:-28], prediction_hobbies_mean[start:-28*2-1])
alpha_hobbies_median = alpha(data_cat_final_weekly.HOBBIES[start+1:-28], prediction_hobbies_median[start:-28*2-1])

fig = go.Figure(
    data=[go.Scatter(y=data_cat_final_weekly.HOBBIES, x = data_cat_final_weekly.HOBBIES.index, name= 'Actual'), 
          go.Scatter(y=prediction_hobbies_min[0:-1]*alpha_hobbies_min, x = data_cat_final_weekly.HOBBIES.index,name= 'Predict Min'),
          go.Scatter(y=prediction_hobbies_max[0:-1]*alpha_hobbies_max, x = data_cat_final_weekly.HOBBIES.index, name= 'Predict Max'),
          go.Scatter(y=prediction_hobbies_mean[0:-1]*alpha_hobbies_mean, x = data_cat_final_weekly.HOBBIES.index, name= 'Predict Mean'),
          go.Scatter(y=prediction_hobbies_median[0:-1]*alpha_hobbies_median, x = data_cat_final_weekly.HOBBIES.index, name= 'Predict median')],
    layout=go.Layout(
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
    )
)
fig.update_layout(title_text="Hobbies Category - MAPA SARIMA forecast")
fig.show()


# 
# As for the Hobbies category, we have an improvement using the MAPA technique with the max function, beating the result of individual SARIMA.

# In[ ]:


rmse_hobbies_weekly= rmse(data_cat_final_weekly.HOBBIES[-28::], predicted_result_hobbies_weekly[-56-1:-28-1], 'weekly Hobbies - Test')
rmse_hobbies_weekly= rmse(data_cat_final_weekly.HOBBIES[start+1:-28], predicted_result_hobbies_weekly[start+1:-28*2-1], 'weekly Hobbies - Train')

#test Fold
rmse_hobbies_max = rmse(data_cat_final_weekly.HOBBIES[-28::], prediction_hobbies_max[-57:-29]*alpha_hobbies_max, 'Hobbies Test - max')
rmse_hobbies_min = rmse(data_cat_final_weekly.HOBBIES[-28::], prediction_hobbies_min[-57:-29]*alpha_hobbies_min, 'Hobbies Test - min')
rmse_hobbies_mean = rmse(data_cat_final_weekly.HOBBIES[-28::], prediction_hobbies_mean[-57:-29]*alpha_hobbies_mean, 'Hobbies Test - mean')
rmse_hobbies_median = rmse(data_cat_final_weekly.HOBBIES[-28::], prediction_hobbies_median[-57:-29]*alpha_hobbies_median, 'Hobbies Test - median')
#train Fold
rmse_hobbies_max = rmse(data_cat_final_weekly.HOBBIES[start+1:-28], prediction_hobbies_max[start:-28*2-1]*alpha_hobbies_max, 'Hobbies Train - max')
rmse_hobbies_min = rmse(data_cat_final_weekly.HOBBIES[start+1:-28], prediction_hobbies_min[start:-28*2-1]*alpha_hobbies_min, 'Hobbies Train - min')
rmse_hobbies_mean = rmse(data_cat_final_weekly.HOBBIES[start+1:-28], prediction_hobbies_mean[start:-28*2-1]*alpha_hobbies_mean, 'Hobbies Train - mean')
rmse_hobbies_median = rmse(data_cat_final_weekly.HOBBIES[start+1:-28], prediction_hobbies_median[start:-28*2-1]*alpha_hobbies_median, 'Hobbies Train - median' )


# ### Household Category Forecasting <a id="4.3"></a>

# In[ ]:


predicted_result_household_bimonthly = predicted_result_household_bimonthly.resample('W').mean()
predicted_result_household_monthly = predicted_result_household_monthly.resample('W').mean()
predictions_household = pd.DataFrame({'bimestral': predicted_result_household_bimonthly.groupby(predicted_result_household_bimonthly.notnull().cumsum()).transform(lambda x : x.sum()/len(x))[0:-4],
                         'mensal': predicted_result_household_monthly.groupby(predicted_result_household_monthly.notnull().cumsum()).transform(lambda x : x.sum()/len(x)),
                         'weekly': predicted_result_household_weekly[1::]})
prediction_household_mean = pd.DataFrame.mean(predictions_household, axis = 1)
prediction_household_median = pd.DataFrame.median(predictions_household, axis = 1)
prediction_household_min = pd.DataFrame.min(predictions_household, axis = 1)
prediction_household_max = pd.DataFrame.max(predictions_household, axis = 1)

alpha_household_max = alpha(data_cat_final_weekly.HOUSEHOLD[start+1:-28], prediction_household_max[start:-28*2-1])
alpha_household_min = alpha(data_cat_final_weekly.HOUSEHOLD[start+1:-28], prediction_household_min[start:-28*2-1])
alpha_household_mean = alpha(data_cat_final_weekly.HOUSEHOLD[start+1:-28], prediction_household_mean[start:-28*2-1])
alpha_household_median = alpha(data_cat_final_weekly.HOUSEHOLD[start+1:-28], prediction_household_median[start:-28*2-1])
fig = go.Figure(
    data=[go.Scatter(y=data_cat_final_weekly.HOUSEHOLD, x=data_cat_final_weekly.HOUSEHOLD.index, name= 'Actual'), 
          go.Scatter(y=prediction_household_min[0:-1]*alpha_household_min, x=data_cat_final_weekly.HOUSEHOLD.index, name= 'Predict Min'),
          go.Scatter(y=prediction_household_max[0:-1]*alpha_household_max, x=data_cat_final_weekly.HOUSEHOLD.index, name= 'Predict Max'),
          go.Scatter(y=prediction_household_mean[0:-1]*alpha_household_mean, x=data_cat_final_weekly.HOUSEHOLD.index, name= 'Predict Mean'),
          go.Scatter(y=prediction_household_median[0:-1]*alpha_household_median, x=data_cat_final_weekly.HOUSEHOLD.index, name= 'Predict median')],
    layout=go.Layout(
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
    )
)
fig.update_layout(title_text="Household Category - MAPA SARIMA forecast")
fig.show()


# 
# The result here is not the biggest on the training base, but the MAPA model was better on the test base.

# In[ ]:


rmse_household_weekly= rmse(data_cat_final_weekly.HOUSEHOLD[-28::], predicted_result_household_weekly[-56-1:-28-1], 'weekly Household - Test')
rmse_household_weekly= rmse(data_cat_final_weekly.HOUSEHOLD[start:-28], predicted_result_household_weekly[start:-28*2-1], 'weekly Household - Train')

#test Fold
rmse_household_max = rmse(data_cat_final_weekly.HOUSEHOLD[-28::], prediction_household_max[-57:-29]*alpha_household_max, 'Household Test - max')
rmse_household_min = rmse(data_cat_final_weekly.HOUSEHOLD[-28::], prediction_household_min[-57:-29]*alpha_household_min, 'Household Test - min')
rmse_household_mean = rmse(data_cat_final_weekly.HOUSEHOLD[-28::], prediction_household_mean[-57:-29]*alpha_household_mean, 'Household Test - mean')
rmse_household_median = rmse(data_cat_final_weekly.HOUSEHOLD[-28::], prediction_household_median[-57:-29]*alpha_household_median, 'Household Test - median')
#train Fold
rmse_household_max = rmse(data_cat_final_weekly.HOUSEHOLD[start+1:-28], prediction_household_max[start:-28*2-1]*alpha_household_max, 'Household Train - max')
rmse_household_min = rmse(data_cat_final_weekly.HOUSEHOLD[start+1:-28], prediction_household_min[start:-28*2-1]*alpha_household_min, 'Household Train - min')
rmse_household_mean = rmse(data_cat_final_weekly.HOUSEHOLD[start+1:-28], prediction_household_mean[start:-28*2-1]*alpha_household_mean, 'Household Train - mean')
rmse_household_median = rmse(data_cat_final_weekly.HOUSEHOLD[start+1:-28], prediction_household_median[start:-28*2-1]*alpha_household_median, 'Household Train - median' )


# ### Daily forecasting with exogenous variables <a id="5"></a>

# 
# For daily forecast, we will make use of exogenous variables (commemorative dates and days of the week)

# In[ ]:


holidays = pd.get_dummies(data_cat['event_name_1'], dummy_na=True)
weekdays = pd.get_dummies(data_cat['wday'])
exog = pd.concat([holidays, weekdays], axis = 1)
exog.index = pd.to_datetime(data_cat_final.index , format = '%Y-%m-%d')
exog.head(10)


# For the daily time series, the seasonality is weekly.

# In[ ]:


predicted_result_foods_daily = sarimax_predictor_exog(data_cat_final.FOODS, [2,1,0], [2,1,0,7], 28*7,
                        'Daily forecast - Foods', exog)


# In[ ]:


predicted_result_hobbies_daily = sarimax_predictor_exog(data_cat_final.HOBBIES, [6,1,1], [1,1,0,7], 28*7,
                        'Daily forecast - Hobbies', exog)


# In[ ]:


predicted_result_household_daily = sarimax_predictor_exog(data_cat_final.HOUSEHOLD, [3,0,0], [1,1,0,7], 28*7,
                        'Daily forecast - Household', exog)


# ### MAPA farecasting (Daily, Weekly, and monthly frequencies) <a id="6"></a>

# In[ ]:


predicted_result_foods_monthly = predicted_result_foods_monthly.resample('W').mean()
predictions_foods = pd.DataFrame({'mensal': predicted_result_foods_monthly.groupby(predicted_result_foods_monthly.notnull().cumsum()).transform(lambda x : x.sum()/len(x))['2011-04-03':'2016-05-01'],
                         'weekly': predicted_result_foods_weekly[1::]['2011-04-03':'2016-05-01'],
                          'daily': predicted_result_hobbies_daily.resample('W').sum()['2011-04-03':'2016-05-01']})
prediction_foods_mean = pd.DataFrame.mean(predictions_foods, axis = 1)
prediction_foods_median = pd.DataFrame.median(predictions_foods, axis = 1)
prediction_foods_min = pd.DataFrame.min(predictions_foods, axis = 1)
prediction_foods_max = pd.DataFrame.max(predictions_foods, axis = 1)
alpha_foods_max = alpha(data_cat_final_weekly.FOODS['2011-04-03':'2015-03-29 '], prediction_foods_max['2011-04-03':'2015-03-29 '])
alpha_foods_min = alpha(data_cat_final_weekly.FOODS['2011-04-03':'2015-03-29 '], prediction_foods_min['2011-04-03':'2015-03-29 '])
alpha_foods_mean = alpha(data_cat_final_weekly.FOODS['2011-04-03':'2015-03-29 '], prediction_foods_mean['2011-04-03':'2015-03-29 '])
alpha_median = alpha(data_cat_final_weekly.FOODS['2011-04-03':'2015-03-29 '], prediction_foods_median['2011-04-03':'2015-03-29 '])

fig = go.Figure(
    data=[go.Scatter(y=data_cat_final_weekly.FOODS, x= data_cat_final_weekly.FOODS.index, name= 'Actual'), 
          go.Scatter(y=prediction_foods_min[0:-1]*alpha_foods_min, x= data_cat_final_weekly.FOODS.index, name= 'Predict Min'),
          go.Scatter(y=prediction_foods_max[0:-1]*alpha_foods_max, x= data_cat_final_weekly.FOODS.index, name= 'Predict Max'),
          go.Scatter(y=prediction_foods_mean[0:-1]*alpha_foods_mean, x= data_cat_final_weekly.FOODS.index, name= 'Predict Mean'),
          go.Scatter(y=prediction_foods_median[0:-1]*alpha_median, x= data_cat_final_weekly.FOODS.index, name= 'Predict median')],
    layout=go.Layout(
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
    )
)
fig.update_layout(title_text="Foods Category - MAPA SARIMA forecast")
fig.show()


# The result here is also not better using the MAPA technique.

# In[ ]:


rmse(data_cat_final_weekly.FOODS[-28::], predicted_result_foods_weekly[-56-1:-28-1], 'weekly Foods - Test')
rmse(data_cat_final_weekly.FOODS[start+1:-28], predicted_result_foods_weekly[start+1:-28*2-1], 'weekly Foods - Train')
#test Fold
rmse_foods_max = rmse(data_cat_final_weekly.FOODS[-28::], prediction_foods_max[-57:-29]*alpha_foods_max, 'Foods Test - max')
rmse_foods_min = rmse(data_cat_final_weekly.FOODS[-28::], prediction_foods_min[-57:-29]*alpha_foods_min, 'Foods Test - min')
rmse_foods_mean = rmse(data_cat_final_weekly.FOODS[-28::], prediction_foods_mean[-57:-29]*alpha_foods_mean, 'Foods Test - mean')
rmse_foods_median = rmse(data_cat_final_weekly.FOODS[-28::], prediction_foods_median[-57:-29]*alpha_median, 'Foods Test - median')
#train Fold
rmse_foods_max = rmse(data_cat_final_weekly.FOODS['2011-04-03':'2015-03-29 '], prediction_foods_max['2011-04-03':'2015-03-29 ']*alpha_foods_max, 'Foods Train - max')
rmse_foods_min = rmse(data_cat_final_weekly.FOODS['2011-04-03':'2015-03-29 '], prediction_foods_min['2011-04-03':'2015-03-29 ']*alpha_foods_min, 'Foods Train - min')
rmse_foods_mean = rmse(data_cat_final_weekly.FOODS['2011-04-03':'2015-03-29 '], prediction_foods_mean['2011-04-03':'2015-03-29 ']*alpha_foods_mean, 'Foods Train - mean')
rmse_foods_median = rmse(data_cat_final_weekly.FOODS['2011-04-03':'2015-03-29 '], prediction_foods_median['2011-04-03':'2015-03-29 ']*alpha_median, 'Foods Train - median' )


# In[ ]:


predicted_result_hobbies_monthly = predicted_result_hobbies_monthly.resample('W').mean()
predictions_hobbies = pd.DataFrame({'mensal': predicted_result_hobbies_monthly.groupby(predicted_result_hobbies_monthly.notnull().cumsum()).transform(lambda x : x.sum()/len(x))['2011-04-03':'2016-05-01'],
                         'weekly': predicted_result_hobbies_weekly['2011-04-03':'2016-05-01'],
                         'daily': predicted_result_hobbies_daily.resample('W').sum()['2011-04-03':'2016-05-01']})
prediction_hobbies_mean = pd.DataFrame.mean(predictions_hobbies, axis = 1)
prediction_hobbies_median = pd.DataFrame.median(predictions_hobbies, axis = 1)
prediction_hobbies_min = pd.DataFrame.min(predictions_hobbies, axis = 1)
prediction_hobbies_max = pd.DataFrame.max(predictions_hobbies, axis = 1)

alpha_hobbies_max = alpha(data_cat_final_weekly.HOBBIES['2011-04-03':'2015-03-29 '], prediction_hobbies_max['2011-04-03':'2015-03-29 '])
alpha_hobbies_min = alpha(data_cat_final_weekly.HOBBIES['2011-04-03':'2015-03-29 '], prediction_hobbies_min['2011-04-03':'2015-03-29 '])
alpha_hobbies_mean = alpha(data_cat_final_weekly.HOBBIES['2011-04-03':'2015-03-29 '], prediction_hobbies_mean['2011-04-03':'2015-03-29 '])
alpha_hobbies_median = alpha(data_cat_final_weekly.HOBBIES['2011-04-03':'2015-03-29 '], prediction_hobbies_median['2011-04-03':'2015-03-29 '])

fig = go.Figure(
    data=[go.Scatter(y=data_cat_final_weekly.HOBBIES, x= data_cat_final_weekly.HOBBIES.index, name= 'Actual'), 
          go.Scatter(y=prediction_hobbies_min[0:-1]*alpha_hobbies_min, x= data_cat_final_weekly.HOBBIES.index, name= 'Predict Min'),
          go.Scatter(y=prediction_hobbies_max[0:-1]*alpha_hobbies_max, x= data_cat_final_weekly.HOBBIES.index, name= 'Predict Max'),
          go.Scatter(y=prediction_hobbies_mean[0:-1]*alpha_hobbies_mean, x= data_cat_final_weekly.HOBBIES.index, name= 'Predict Mean'),
          go.Scatter(y=prediction_hobbies_median[0:-1]*alpha_hobbies_median, x= data_cat_final_weekly.HOBBIES.index, name= 'Predict median')],
    layout=go.Layout(
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
    )
)
fig.update_layout(title_text="Hobbies Category - MAPA SARIMA forecast")
fig.show()


# 
# The result here was better with the MAPA technique using the median and mean for combination.

# In[ ]:


rmse_hobbies_weekly= rmse(data_cat_final_weekly.HOBBIES[-28::], predicted_result_hobbies_weekly[-56-1:-28-1], 'weekly Hobbies - Test')
rmse_hobbies_weekly= rmse(data_cat_final_weekly.HOBBIES[start+1:-28], predicted_result_hobbies_weekly[start+1:-28*2-1], 'weekly Hobbies - Train')

#test Fold
rmse_hobbies_max = rmse(data_cat_final_weekly.HOBBIES[-28::], prediction_hobbies_max[-57:-29]*alpha_hobbies_max, 'Hobbies Test - max')
rmse_hobbies_min = rmse(data_cat_final_weekly.HOBBIES[-28::], prediction_hobbies_min[-57:-29]*alpha_hobbies_min, 'Hobbies Test - min')
rmse_hobbies_mean = rmse(data_cat_final_weekly.HOBBIES[-28::], prediction_hobbies_mean[-57:-29]*alpha_hobbies_mean, 'Hobbies Test - mean')
rmse_hobbies_median = rmse(data_cat_final_weekly.HOBBIES[-28::], prediction_hobbies_median[-57:-29]*alpha_hobbies_median, 'Hobbies Test - median')
#train Fold
rmse_hobbies_max = rmse(data_cat_final_weekly.HOBBIES['2011-04-03':'2015-03-29 '], prediction_hobbies_max['2011-04-03':'2015-03-29 ']*alpha_hobbies_max, 'Hobbies Train - max')
rmse_hobbies_min = rmse(data_cat_final_weekly.HOBBIES['2011-04-03':'2015-03-29 '], prediction_hobbies_min['2011-04-03':'2015-03-29 ']*alpha_hobbies_min, 'Hobbies Train - min')
rmse_hobbies_mean = rmse(data_cat_final_weekly.HOBBIES['2011-04-03':'2015-03-29 '], prediction_hobbies_mean['2011-04-03':'2015-03-29 ']*alpha_hobbies_mean, 'Hobbies Train - mean')
rmse_hobbies_median = rmse(data_cat_final_weekly.HOBBIES['2011-04-03':'2015-03-29 '], prediction_hobbies_median['2011-04-03':'2015-03-29 ']*alpha_hobbies_median, 'Hobbies Train - median' )


# In[ ]:


predicted_result_household_bimonthly = predicted_result_household_bimonthly.resample('W').mean()
predicted_result_household_monthly = predicted_result_household_monthly.resample('W').mean()
predictions_household = pd.DataFrame({'mensal': predicted_result_household_monthly.groupby(predicted_result_household_monthly.notnull().cumsum()).transform(lambda x : x.sum()/len(x))['2011-04-03':'2016-05-01'],
                         'weekly': predicted_result_household_weekly['2011-04-03':'2016-05-01'],
                          'daily': predicted_result_household_daily.resample('W').sum()['2011-04-03':'2016-05-01']})
prediction_household_mean = pd.DataFrame.mean(predictions_household, axis = 1)
prediction_household_median = pd.DataFrame.median(predictions_household, axis = 1)
prediction_household_min = pd.DataFrame.min(predictions_household, axis = 1)
prediction_household_max = pd.DataFrame.max(predictions_household, axis = 1)

alpha_household_max = alpha(data_cat_final_weekly.HOUSEHOLD['2011-04-03':'2015-03-29 '], prediction_household_max['2011-04-03':'2015-03-29 '])
alpha_household_min = alpha(data_cat_final_weekly.HOUSEHOLD['2011-04-03':'2015-03-29 '], prediction_household_min['2011-04-03':'2015-03-29 '])
alpha_household_mean = alpha(data_cat_final_weekly.HOUSEHOLD['2011-04-03':'2015-03-29 '], prediction_household_mean['2011-04-03':'2015-03-29 '])
alpha_household_median = alpha(data_cat_final_weekly.HOUSEHOLD['2011-04-03':'2015-03-29 '], prediction_household_median['2011-04-03':'2015-03-29 '])
fig = go.Figure(
    data=[go.Scatter(y=data_cat_final_weekly.HOUSEHOLD, x= data_cat_final_weekly.HOUSEHOLD.index, name= 'Actual'), 
          go.Scatter(y=prediction_household_min[0:-1]*alpha_household_min, x= data_cat_final_weekly.HOUSEHOLD.index, name= 'Predict Min'),
          go.Scatter(y=prediction_household_max[0:-1]*alpha_household_max, x= data_cat_final_weekly.HOUSEHOLD.index, name= 'Predict Max'),
          go.Scatter(y=prediction_household_mean[0:-1]*alpha_household_mean, x= data_cat_final_weekly.HOUSEHOLD.index, name= 'Predict Mean'),
          go.Scatter(y=prediction_household_median[0:-1]*alpha_household_median, x= data_cat_final_weekly.HOUSEHOLD.index, name= 'Predict median')],
    layout=go.Layout(
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
    )
)
fig.update_layout(title_text="Household Category - MAPA SARIMA forecast")
fig.show()


# 
# The result here was excellent, better than the MAPA technique using aggregation with weekly, monthly and bimonthly frequencies

# In[ ]:


rmse_household_weekly= rmse(data_cat_final_weekly.HOUSEHOLD[-28::], predicted_result_household_weekly[-56-1:-28-1], 'weekly Household - Test')
rmse_household_weekly= rmse(data_cat_final_weekly.HOUSEHOLD[start:-28], predicted_result_household_weekly[start:-28*2-1], 'weekly Household - Train')

#test Fold
rmse_household_max = rmse(data_cat_final_weekly.HOUSEHOLD[-28::], prediction_household_max[-57:-29]*alpha_household_max, 'Household Test - max')
rmse_household_min = rmse(data_cat_final_weekly.HOUSEHOLD[-28::], prediction_household_min[-57:-29]*alpha_household_min, 'Household Test - min')
rmse_household_mean = rmse(data_cat_final_weekly.HOUSEHOLD[-28::], prediction_household_mean[-57:-29]*alpha_household_mean, 'Household Test - mean')
rmse_household_median = rmse(data_cat_final_weekly.HOUSEHOLD[-28::], prediction_household_median[-57:-29]*alpha_household_median, 'Household Test - median')
#train Fold
rmse_household_max = rmse(data_cat_final_weekly.HOUSEHOLD['2011-04-03':'2015-03-29 '], prediction_household_max['2011-04-03':'2015-03-29 ']*alpha_household_max, 'Household Train - max')
rmse_household_min = rmse(data_cat_final_weekly.HOUSEHOLD['2011-04-03':'2015-03-29 '], prediction_household_min['2011-04-03':'2015-03-29 ']*alpha_household_min, 'Household Train - min')
rmse_household_mean = rmse(data_cat_final_weekly.HOUSEHOLD['2011-04-03':'2015-03-29 '], prediction_household_mean['2011-04-03':'2015-03-29 ']*alpha_household_mean, 'Household Train - mean')
rmse_household_median = rmse(data_cat_final_weekly.HOUSEHOLD['2011-04-03':'2015-03-29 '], prediction_household_median['2011-04-03':'2015-03-29 ']*alpha_household_median, 'Household Train - median' )

