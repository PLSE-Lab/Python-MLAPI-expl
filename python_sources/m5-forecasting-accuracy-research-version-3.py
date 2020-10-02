#!/usr/bin/env python
# coding: utf-8

# # M5 Forecasting Accuracy Research
# 
# This is a continuation of my work on analyzing the sales data of Walmart's TX_1 store (Version 1 found here:https://www.kaggle.com/jimmyliuu/m5-forecast-accuracy-research-version-1). This week, I add a couple exogenous variables into my SARIMA model in hopes of improving forecast accuracy. I chose to force weekly seasonality and event_name_1 into the model. 
# 
# I followed Jason Brownlee's "How to Decompose Time Series Data into Trend and Seasonality" to build the SARIMAX model with forced weekly seasonality (found here:https://machinelearningmastery.com/decompose-time-series-data-trend-seasonality/)
# 
# I followed Oscar Arzamendia's "Time Series Forecasting - A Getting Started Guide" as a guide to feature engineering. I used one-hot-encoding on event_name_1 to feed it into the SARIMAX model (found here:https://towardsdatascience.com/time-series-forecasting-a-getting-started-guide-c435f9fa2216#:~:text=An%20exogenous%20variable%20is%20one,without%20being%20affected%20by%20it.)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Load in relevant datasets

# In[ ]:


CalendarDF=pd.read_csv("/kaggle/input/m5-forecasting-accuracy/calendar.csv", header=0)
SalesDF=pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sales_train_evaluation.csv", header=0) #June 1st Dataset


# In[ ]:


import os, psutil

pid = os.getpid()
py = psutil.Process(pid)
memory_use = py.memory_info()[0] / 2. ** 30
print ('memory GB:' + str(np.round(memory_use, 2)))


# # Preparing the dataset

# In[ ]:


CalendarDF['date'] = pd.to_datetime(CalendarDF.date)

TX_1_Sales = SalesDF[['TX_1' in x for x in SalesDF['store_id'].values]]
TX_1_Sales = TX_1_Sales.reset_index(drop = True)
TX_1_Sales.info()


# In[ ]:


# Generate MultiIndex for easier aggregration.
TX_1_Indexed = pd.DataFrame(TX_1_Sales.groupby(by = ['cat_id','dept_id','item_id']).sum())
TX_1_Indexed.info()


# In[ ]:


# Aggregate total sales per day for each sales category
Food = pd.DataFrame(TX_1_Indexed.xs('FOODS').sum(axis = 0))
Hobbies = pd.DataFrame(TX_1_Indexed.xs('HOBBIES').sum(axis = 0))
Household = pd.DataFrame(TX_1_Indexed.xs('HOUSEHOLD').sum(axis = 0))
Food.info()


# In[ ]:


# Merge the aggregated sales data to the calendar dataframe based on date
CalendarDF = CalendarDF.merge(Food, how = 'left', left_on = 'd', right_on = Food.index)
CalendarDF = CalendarDF.rename(columns = {0:'Food'})
CalendarDF = CalendarDF.merge(Hobbies, how = 'left', left_on = 'd', right_on = Hobbies.index)
CalendarDF = CalendarDF.rename(columns = {0:'Hobbies'})
CalendarDF = CalendarDF.merge(Household, how = 'left', left_on = 'd', right_on = Household.index)
CalendarDF = CalendarDF.rename(columns = {0:'Household'})
CalendarDF.head(10)


# In[ ]:


# Drop dates with null sales data
CalendarDF = CalendarDF.drop(CalendarDF.index[1941:])
CalendarDF.reset_index(drop = True)


# Here, I perform a couple of correlation tests between each of the sales categories (food, hobbies, and household).

# In[ ]:


# Collect sales data from each category into one dataframe
categoriesDF = CalendarDF[['Food','Hobbies','Household']]
categoriesDF.corr(method = 'pearson')
categoriesDF.corr(method = 'spearman')
categoriesDF.corr(method = 'kendall')


# # Building the SARIMAX model with forced weekly seasonality
# 
# Reference: https://machinelearningmastery.com/decompose-time-series-data-trend-seasonality/ Section 15

# In[ ]:


from statsmodels.tsa.seasonal import seasonal_decompose

Food.index = CalendarDF['date']

# Split food sales data into train and test 
foodTrain = Food['20110129':'20160410']
foodTest = Food['20160411':'20160522']

# Drop 0 sales values to prepare data for multiplicative seasonal decomposition
foodTrain = foodTrain[foodTrain[foodTrain.columns[0]] !=0]

# Seasonal decomposition
result = seasonal_decompose(foodTrain, model = 'multiplicative', extrapolate_trend = 'freq', freq = 7) # frequency set to weekly

# Store seasonality component of decomposition
seasonal = result.seasonal.to_frame()
seasonal_index = result.seasonal[-7:].to_frame()

# Merge the train data and the seasonality 
foodTrain = foodTrain.merge(seasonal, how = 'left', on = foodTrain.index , left_index = True, right_index = True)


# In[ ]:


# Building the SARIMAX model
# I use the Pyramid Arima package to perform an auto-SARIMAX forecast

get_ipython().system('pip install pmdarima')
import pmdarima as pm

#SARIMAX Model setting the exogenous variable to weekly seasonality 
sxmodel = pm.auto_arima(foodTrain[foodTrain.columns[0]], exogenous= foodTrain[['seasonal']],
                           start_p=1, start_q=1,
                           test='adf',
                           max_p=3, max_q=3, m=7,
                           start_P=0, seasonal=True,
                           d=None, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)

sxmodel.summary()


# In[ ]:


# Forecasting using the SARIMAX model
import matplotlib.pyplot as plt

n_periods = 42
fitted, confint = sxmodel.predict(n_periods = n_periods,  exogenous= np.tile(seasonal_index['seasonal'], 6).reshape(-1,1),  return_conf_int = True)

index_of_fc = pd.date_range(foodTest.index[0], periods = n_periods, freq = 'D')

# make series for plotting purpose
fitted_series = pd.Series(fitted, index=index_of_fc)
lower_series = pd.Series(confint[:, 0], index=index_of_fc)
upper_series = pd.Series(confint[:, 1], index=index_of_fc)

# Plot
plt.plot(foodTest)
plt.plot(fitted_series, color='darkgreen')
plt.fill_between(lower_series.index, 
                 lower_series, 
                 upper_series, 
                 color='k', alpha=.15)

plt.title("SARIMA - Total Sales of TX_1")
plt.show()


# # Building the SARIMAX model using event_name_1 as an exogenous variable

# In[ ]:


# data engineering for event_name_1
CalendarDF['isweekday'] = [1 if wday >= 3 else 0 for wday in CalendarDF.wday.values]
CalendarDF['isweekend'] = [0 if wday > 2 else 1 for wday in CalendarDF.wday.values]
CalendarDF['holiday_weekend'] = [1 if (we == 1 and h not in [np.nan]) else 0 for we,h in CalendarDF[['isweekend','event_name_1']].values]
CalendarDF['holiday_weekday'] = [1 if (wd == 1 and h not in [np.nan]) else 0 for wd,h in CalendarDF[['isweekday','event_name_1']].values]

# one-hot-encoding event_name_1
CalendarDF = pd.get_dummies(CalendarDF, columns=['event_name_1'], prefix=['holiday'], dummy_na=True)

Food = CalendarDF['Food']
Food.index = CalendarDF['date']

# Section out the columns created by encoding and concat with Food dataframe
temp = CalendarDF.iloc[:,16:50]
temp.index = CalendarDF['date']
Food = pd.concat([Food, temp], axis = 1)

foodTrain = Food['20110129':'20160410']
foodTest = Food['20160411':'20160522']


# In[ ]:


# Build the SARIMAX model
sxmodel_event = pm.auto_arima(foodTrain[foodTrain.columns[0]], exogenous= foodTrain.iloc[:,1:],
                           start_p=1, start_q=1,
                           test='adf',
                           max_p=3, max_q=3, m=7,
                           start_P=0, seasonal=True,
                           d=None, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)

sxmodel_event.summary()


# In[ ]:


# Forecast
n_periods = 42
event_predict, confint = sxmodel_event.predict(n_periods = n_periods,  exogenous= foodTest.iloc[:,1:],  return_conf_int = True)

index_of_fc = pd.date_range(foodTest.index[0], periods = n_periods, freq = 'D')

# make series for plotting purpose
fitted_series = pd.Series(event_predict, index=index_of_fc)
lower_series = pd.Series(confint[:, 0], index=index_of_fc)
upper_series = pd.Series(confint[:, 1], index=index_of_fc)

# Plot
#plt.plot(foodTrain)
plt.plot(foodTest)
plt.plot(fitted_series, color='darkgreen')
plt.fill_between(lower_series.index, 
                 lower_series, 
                 upper_series, 
                 color='k', alpha=.15)

plt.title("SARIMA - Total Sales of TX_1")
plt.show()


# # Comparing Results
# 
# I will now compare the forecasting results from the ARIMA model, the SARIMA model, and the two SARIMAX models using the sMAPE and MASE functions. (reference: https://gist.github.com/bshishov/5dc237f59f019b26145648e2124ca1c9)

# In[ ]:


#Accuracy metrics
def symmetric_mean_absolute_percentage_error(actual,forecast):
    return 1/len(actual) * np.sum(2 * np.abs(forecast-actual)/(np.abs(actual)+np.abs(forecast)))

def mean_absolute_error(actual, forecast):
    return np.mean(np.abs(actual - forecast))

def naive_forecasting(actual, seasonality):
    return actual[:-seasonality]

def mean_absolute_scaled_error(actual, forecast, seasonality):
    return mean_absolute_error(actual, forecast) / mean_absolute_error(actual[seasonality:], naive_forecasting(actual, seasonality))


# In[ ]:


symmetric_mean_absolute_percentage_error(foodTest[foodTest.columns[0]], fitted) #sMAPE of SARIMAX with forced seasonality


# In[ ]:


symmetric_mean_absolute_percentage_error(foodTest[foodTest.columns[0]], event_predict) #sMAPE of SARIMAX with event_name_1


# Both SARIMAX models are slightly more accurate than the ARIMA model based on sMAPE.
