#!/usr/bin/env python
# coding: utf-8

# # Forecasting 3 Months of Sales
# Given 5 years of daily sales data across 10 stores for 50 items, we have been tasked to forecast the next 3 months of sales. We will be exploring the data using Pandas and building models using ARIMA, tensorflow's DNN regressor, and xgboost.
# 
# Let's get started!
# ##### NOTE
# This is my first competition and I'm still learning the models myself. At the end I share what I learned while building this.

# # Import Libraries
# Below are all the libraries that we'll use (with some extra for notebook aesthetics).

# In[ ]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
#from jupyterthemes import jtplot
#jtplot.style(theme='chesterish')

from scipy.spatial.distance import euclidean #used for fdt
import fastdtw as fdt #fast dynamic time warping
from statsmodels.tsa.seasonal import seasonal_decompose #decompose seasonality
from statsmodels.tsa.stattools import adfuller #test if series is stationary (then can perform ARIMA)

"""from pyramid.arima import auto_arima #auto ARIMA model (pip install pyramid-arima)"""
import xgboost as xgb #xgboost model
import tensorflow as tf #DNN estimator model

path = '../input/'


# In[ ]:


plt.rcParams["figure.figsize"] = [16,9]


# # Metrics and 2 Models
# ## Error Metric
# We'll be using the Symmetric Mean Absolute Percentage Error as our forecasting error metric. Defining a function saves us from writing the code multiple times.

# In[ ]:


def SMAPE (forecast, actual):
    """Returns the Symmetric Mean Absolute Percentage Error between two Series"""
    masked_arr = ~((forecast==0)&(actual==0))
    diff = abs(forecast[masked_arr] - actual[masked_arr])
    avg = (abs(forecast[masked_arr]) + abs(actual[masked_arr]))/2
    
    print('SMAPE Error Score: ' + str(round(sum(diff/avg)/len(forecast) * 100, 2)) + ' %')


# ## Stationarity Test (Dickey Fuller)
# Time Series data should be stationary before applying an ARIMA model. Stationary means that the mean, standard deviation, and variance don't change over time. The function below tests whether or not a Time Series is stationary.

# In[ ]:


def Fuller(TimeSeries):
    """Provides Fuller test results for TimeSeries"""
    stationary_test = adfuller(TimeSeries)
    print('ADF Statistic: %f' % stationary_test[0])
    print('p-value: %f' % stationary_test[1])
    print('Critical Values:')
    for key, value in stationary_test[4].items():
        print('\t%s: %.3f' % (key, value))


# ## ARIMA Model
# General ARIMA model that will be used.

# In[ ]:


"""def ARIMA(TimeSeries, maxP, maxQ, maxD):"""
    """Returns ARIMA model (not fitted)"""
    """stepwise_model = auto_arima(TimeSeries, start_p=1, start_q=1,
                           max_p=maxP, max_q=maxQ,
                           start_P=0, seasonal=True,
                           d=1, max_d=maxD, D=1, trace=False,
                           error_action='ignore',
                           suppress_warnings=True,
                           stepwise=True,
                           maxiter=500)
    print(stepwise_model.aic())
    return stepwise_model"""


# ## XGBoost Model
# General xgboost model that will be used.

# In[ ]:


def xboost(x_train, y_train, x_test):
    """Trains xgboost model and returns Series of predictions for x_test"""
    dtrain = xgb.DMatrix(x_train, label=y_train, feature_names=list(x_train.columns))
    dtest = xgb.DMatrix(x_test, feature_names=list(x_test.columns))

    params = {'max_depth':3,
              'eta':0.2,
              'silent':1,
              'subsample':1}
    num_rounds = 1500

    bst = xgb.train(params, dtrain, num_rounds)
    
    return pd.Series(bst.predict(dtest))


# # Data Exploration
# ## Retrieve Data
# Open the competition training data. We'll be exploring this before splitting for our models.

# In[ ]:


df = pd.read_csv(path +'train.csv', index_col=0)
df.index = pd.to_datetime(df.index)
df.tail()


# In[ ]:


df.info()


# ## Store Trends
# Here we're looking to see if there are any seasonality trends in the total store sales. We'll group by week so we can more clearly see trends in the plots.

# In[ ]:


stores = pd.DataFrame(df.groupby(['date','store']).sum()['sales']).unstack()
stores = stores.resample('7D',label='left').sum()
stores.sort_index(inplace = True)


# In[ ]:


stores.plot(figsize=(16,9), title='Weekly Store Sales', legend=None)
plt.show()


# The above plot charts every store's sales by week. But how does the average trend? The 25% quartile?
# 
# Let's look:

# In[ ]:


store_qtr = pd.DataFrame(stores.quantile([0.0,0.25,0.5,0.75,1.0],axis=1)).transpose()
store_qtr.sort_index(inplace = True)
store_qtr.columns = ['Min','25%','50%','75%','Max']
store_qtr.plot(figsize=(16,9), title='Weekly Quartile Sales')
plt.show()


# We can see there's quite a gap between the 25% quartile and average. However, as the other chart shows as well, each store shares a general seasonality. They have highs and lows during the same periods of time.
# 
# Let's take a look at the seasonality aspect of the average. But before that, we're going to track the week-to-week difference.

# In[ ]:


seasonal = seasonal_decompose(pd.DataFrame(store_qtr['50%']).diff(1).iloc[1:,0],model='additive')
seasonal.plot()
plt.suptitle = 'Additive Seasonal Decomposition of Average Store Week-to-Week Sales'
plt.show()


# In[ ]:


Fuller(pd.DataFrame(store_qtr['50%']).diff(1).iloc[1:,0])


# ### Store Trends Conclusion
# There is definitely seasonality in the store sales. Taking the week-to-week difference provides a dataset that is very likely to be stationary (< 1% chance that it's not). If we were to use this as a starting point for our model, we could cluster the stores to the nearest 25% quartile.

# ## Item Sales Trends
# Now we'll do the same analysis for the total item sales. And again, we're looking at weekly sales.

# In[ ]:


items = pd.DataFrame(df.groupby(['date','item']).sum()['sales']).unstack()
items = items.resample('7D',label='left').sum()
items.sort_index(inplace = True)

items.tail(13)


# In[ ]:


items.plot(figsize=(16,9), title='Weekly Item Sales', legend=None)
plt.show()


# Since there are more items than there were stores, we can look at more quartiles. Let's see how every 10% quartile trends.

# In[ ]:


item_WK_qtr = pd.DataFrame(items.quantile([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],axis=1)).transpose()
item_WK_qtr.sort_index(inplace = True)
item_WK_qtr.columns = ['Min','10%','20%','30%','40%','50%','60%','70%','80%','90%','Max']
item_WK_qtr.plot(figsize=(16,9), title='Weekly Quartile Sales')
plt.show()


# Like we saw in the store sales plots, there is seasonality in item sales. Let's break out the seasonal component for the average like we had before:

# In[ ]:


seasonal = seasonal_decompose(pd.DataFrame(item_WK_qtr['50%']).diff(1).iloc[1:,0],model='additive')
seasonal.plot()
plt.title = 'Additive Seasonal Decomposition of Average Item Week-to-Week Sales'
plt.show()


# In[ ]:


Fuller(pd.DataFrame(item_WK_qtr['50%']).diff(1).iloc[1:,0])


# ### Item Trend Conclusion
# Item sales are also seasonal. No surprise there. Week-to-week differencing provides a dataset that is very likely to be stationary (< 1% chance that it's not). If we were to use this as a basis for our model, we could cluster the items to the nearest 10% quartiles.

# ## Store & Item Variability
# We've seen how stores and items trend by themselves, but do some stores sell more of one item? In other words: do the stores have the same sales mix? Are the items sold evenly (percentage-wise) across all stores?
# 
# Below is a plot for the % distribution of each item's sales across the stores (each row adds to 100%). As we can see, it's very uniform. The takeaway here is that the items are sold evenly across the stores.

# In[ ]:


store_item = df.groupby(by=['item','store']).sum()['sales'].groupby(level=0).apply(
    lambda x: 100* x/ x.sum()).unstack()
sns.heatmap(store_item, cmap='Blues', linewidths=0.01, linecolor='gray').set_title(
    'Store % of Total Sales by Item')
plt.show()


# Now to confirm, let's look at the % distribution of each store's sales across the different items (each row adds to 100%).
# 
# We can see that each store overall sold roughly the same percentage of each item.

# In[ ]:


item_store = df.groupby(by=['store','item']).sum()['sales'].groupby(level=0).apply(
    lambda x: 100* x/ x.sum()).unstack()
sns.heatmap(item_store , cmap='Blues', linewidths=0.01, linecolor='gray').set_title(
    'Item % of Total Sales by Store')
plt.show()


# ### Store vs Item Conclusion
# Items have roughly same percentage sales across all stores. We could use this in our model.

# ## Day of Week Variability
# How do sales vary by day of week? Is there seasonality as well? Do stores share same trends? 

# In[ ]:


df['Day'] = df.index.weekday_name
df.head()


# In[ ]:


dow_store = df.groupby(['store','Day']).sum()['sales'].groupby(level=0).apply(
    lambda x: 100* x/ x.sum()).unstack().loc[:,['Monday',
                                                'Tuesday',
                                                'Wednesday',
                                                'Thursday',
                                                'Friday',
                                                'Saturday',
                                                'Sunday']]
sns.heatmap(dow_store, cmap='Blues', linewidths=0.01, linecolor='gray').set_title(
    'Day % of Total Sales by Store')
plt.show()


# The plot above shows the % mix of store sales by day. We can see that the stores are very similar in what days are popular.
# 
# Let's do the same for the items.

# In[ ]:


dow_item = df.groupby(['item','Day']).sum()['sales'].groupby(level=0).apply(
    lambda x: 100* x/ x.sum()).unstack().loc[:,['Monday',
                                                'Tuesday',
                                                'Wednesday',
                                                'Thursday',
                                                'Friday',
                                                'Saturday',
                                                'Sunday']]
sns.heatmap(dow_item, cmap='Blues', linewidths=0.01, linecolor='gray').set_title(
    'Day % of Total Sales by Item')
plt.show()


# This plot tells us that each item's sales are nearly identical in terms of which days are more popular.
# 
# Now let's see if each day generally trends the same as the total week.

# In[ ]:


dow = pd.DataFrame(df.groupby(['date','Day']).sum()['sales']).unstack()['sales'].loc[:,
                                                                                ['Monday',
                                                                               'Tuesday',
                                                                               'Wednesday',
                                                                               'Thursday',
                                                                               'Friday',
                                                                               'Saturday',
                                                                               'Sunday']]
dow = dow.resample('7D',label='left').sum()
dow.sort_index(inplace = True)


# In[ ]:


dow.plot(figsize=(16,9), title='Sales by Day of Week')
plt.show()


# ### Day of Week Conclusion
# Day of week does impact sales, however all stores & items have similar distributions. Day of week trends follow general weekly trend.

# ## Findings and Steps Forward
# Items and stores weekly sales have seasonality and can be munged into a stationary dataset. They also have similar day of week variability, and items have roughly same distributions in stores.
# 
# ### Modeling Process
# Split the data into train and test data (3 months of test). Will compare several models, all of which are outlined below. The goal is to find the model with the best accuracy.
# #### Model (1.1)
# + Dynamic Time Warping (DTW) on item *__weekly__* sales to cluster to nearest 10% quartile
# + Forecast with *__ARIMA__*
# + Percentages will be used to find item sales by store by day
# 
# #### Model (1.2)
# + Forecast weekly item sales with *__ARIMA__*
# + Percentages will be used to find item sales by store by day
# 
# #### Model (2)
# + Item *__daily__* sales with added features:
#  + Day of year (in mod 364)
#  + Day of Quarter (in mod 91)
#  + Quarter (in mod 4)
#  + Day of week (binary columns)
#  + Month
#  + Prior year sales
#  + Average sales by item by store by day of year
#  + Average sales by item by store by day of week by quarter
#  + Whether or not a weekend (Fri-Sun)
#  + Dynamic Time Warping (DTW) on item weekly sales to cluster to nearest 10% quartile
# + Forecast with *__feed forward neural network__*
# 
# #### Model (3)
# + Item *__daily__* sales with added features:
#  + Day of year (in mod 364)
#  + Day of Quarter (in mod 91)
#  + Quarter (in mod 4)
#  + Day of week (binary columns)
#  + Month
#  + Prior year sales
#  + Average sales by item by store by day of year
#  + Average sales by item by store by day of week by quarter
#  + Whether or not a weekend (Fri-Sun)
#  + Dynamic Time Warping (DTW) on item weekly sales to cluster to nearest 10% quartile
# + Forecast with *__xgboost__*

# # ARIMA Models
# ## Model (1.1) - Clustered Weekly Data
# 
# NOTE: Most of the ARIMA model code is commented due to Kaggle only allowing one custom library. Error results are reported at the end of each model.

# Will be using the 10% quartile weekly item sales that was created during the exploratory analysis. Since the competition is predicting the next 3 months of sales, we will use 3 months (13 weeks) of test data.
# 
# We will build an ARIMA model for each quartile then use clustering and percentages to arrive at daily items sales by store.

# In[ ]:


train = item_WK_qtr[:-13]
test = df.loc[df.index >= pd.to_datetime('October 3, 2017')] # last 13 weeks of data


# In[ ]:


store_pct = store_item.transpose()
store_pct


# #### Dynamic Time Warping to 10% Quartiles
# Matches each item to nearest 10% quartile. Outputs list of item id, % quartile /10, and dtw score.

# In[ ]:


fitted_items_WK = []
qtr_list = [0] *11

for column in items:
    for c in range(11):
        qtr_list[c] = [fdt.fastdtw(items[column],item_WK_qtr.iloc[:,c], dist= euclidean)[0], c]
    qtr_list.sort()
    fitted_items_WK.append([column[1], qtr_list[0][1], qtr_list[0][0]])


# #### Fitting Models and Forecasting

# In[ ]:


"""ARIMA_predictions = pd.DataFrame()

for column in item_WK_qtr:
    model = ARIMA(item_WK_qtr[column], 52, 52, 52)
    model.fit(train[column])
    ARIMA_predictions[column] = model.predict(n_periods=13)"""


# In[ ]:


"""item_WK_predictions = pd.DataFrame()

for i in range(50):
    item_WK_predictions[fitted_items_WK[i][0]] = ARIMA_predictions.iloc[:,fitted_items_WK[i][1]]

item_WK_predictions.head()"""


# #### Convert Item Weekly Predictions to Daily Predictions
# Use day of week percentages from before to calculate daily item sales.

# In[ ]:


"""item_Day_pred = []

for column in item_WK_predictions:
    for i, row in item_WK_predictions.iterrows():
        for col in range(7):
            item_Day_pred.append([i, dow_item.columns[col], column, dow_item.iloc[int(column)-1,col]
                                 * item_WK_predictions[column][i]/100])
            
item_Day_fcst = pd.DataFrame(item_Day_pred, columns=['Week #','Day','item','Prediction'])

item_Day_fcst.head()"""


# #### Split Predictions by Store
# Reshape the store_item DataFrame and use percentages to calculate daily item sales by store.

# In[ ]:


"""store_item = pd.DataFrame(store_item.stack()).reset_index()
store_item.columns = ['item','store','pct']

item_Day_fcst = item_Day_fcst.merge(store_item, on= 'item')

item_Day_fcst['sales'] = item_Day_fcst['Prediction'] * item_Day_fcst['pct']/100"""


# In[ ]:


"""item_Day_fcst = item_Day_fcst.loc[:,['Week #','Day','store','item','sales']]

item_Day_fcst.head()"""


# #### Convert Week Number and Day of Week into Datetime
# Based on where the data was split for testing, the weeks start on Tuesdays so there's no offset then. This adds an additional day of data that we'll need to cutoff.
# 
# This is needed so we can remove the additional day in a readable way.

# In[ ]:


"""def str_to_date(row):"""
    """Takes day of week string and week offset to calculate date"""
    """switcher = {
        'Tuesday': 0, #data starts on a Tuesday, so 0 offset
        'Wednesday': 1,
        'Thursday': 2,
        'Friday': 3,
        'Saturday': 4,
        'Sunday': 5,
        'Monday': 6
    }
    weeks = pd.to_timedelta(7* row['Week #'], unit='D')
    days = pd.to_timedelta(switcher.get(row['Day']), unit='D')
    
    return pd.to_datetime('October 3, 2017') + weeks + days


item_Day_fcst['Date'] = item_Day_fcst.apply(lambda row: str_to_date(row), axis=1)
item_Day_fcst.index = item_Day_fcst['Date']"""


# In[ ]:


"""item_Day_fcst.sort_values(['item','store','Date'], inplace=True)
item_Day_fcst['sales']= round(item_Day_fcst['sales'], 0)

item_Day_fcst = item_Day_fcst[['store','item','sales']].loc[
    item_Day_fcst.index < pd.to_datetime('January 1, 2018')]"""


# #### Model Accuracy
# The predictions have been organized the same as the testing data, so we can simply plug both into our error function.
# 
# From this model we get 19.49% error.

# In[ ]:


"""SMAPE(item_Day_fcst['sales'], test['sales'])"""


# ## Model (1.2) - Unclustered Weekly Data

# Now that we've forecasted item quartiles, let's forecast for each item separately. This is to see if there's a difference in accuracy.
# 
# We will build an ARIMA model for each item then use percentages to arrive at daily items sales by store.

# In[ ]:


train = items['sales'][:-13]


# #### Fitting Models and Forecasting

# In[ ]:


"""item_WK_predictions = pd.DataFrame()

for column in items['sales']:
    model = ARIMA(items['sales'][column], 52, 52, 52)
    model.fit(train[column])
    item_WK_predictions[column] = model.predict(n_periods=13)"""


# #### Convert Item Weekly Predictions to Daily Predictions
# Using day of week percentages from before.

# In[ ]:


"""item_Day_pred = []

for column in item_WK_predictions:
    for i, row in item_WK_predictions.iterrows():
        for col in range(7):
            item_Day_pred.append([i, dow_item.columns[col], column, dow_item.iloc[int(column)-1,col]
                                 * item_WK_predictions[column][i]/100])
            
item_Day_fcst = pd.DataFrame(item_Day_pred, columns=['Week #','Day','item','Prediction'])

item_Day_fcst.head()"""


# #### Split Predictions by Store
# Reshape the store_item DataFrame and use percentages.

# In[ ]:


"""item_Day_fcst = item_Day_fcst.merge(store_item, on= 'item')

item_Day_fcst['sales'] = item_Day_fcst['Prediction'] * item_Day_fcst['pct']/100"""


# In[ ]:


"""item_Day_fcst = item_Day_fcst.loc[:,['Week #','Day','store','item','sales']]"""


# #### Convert Week Number and Day of Week into Datetime

# In[ ]:


"""item_Day_fcst['Date'] = item_Day_fcst.apply(lambda row: str_to_date(row), axis=1)
item_Day_fcst.index = item_Day_fcst['Date']"""


# In[ ]:


"""item_Day_fcst.sort_values(['item','store','Date'], inplace=True)
item_Day_fcst['sales']= round(item_Day_fcst['sales'], 0)

item_Day_fcst = item_Day_fcst[['store','item','sales']].loc[
    item_Day_fcst.index < pd.to_datetime('January 1, 2018')]"""


# #### Model Accuracy
# 
# We get 19.60% error.

# In[ ]:


"""SMAPE(item_Day_fcst['sales'], test['sales'])"""


# # DNN Model
# ## Model (2) - Feed Forward Neural Network with Daily Data

# To really take advantage of the DNN, we need to add features. We won't be adding any rolling/ expanding windows since they'd be unreliable on the competition data. Most of the engineered features are categorical, with the exception being prior year sales.
# 
# Below are some constants we'll need to use for working with datetimes.

# In[ ]:


ns_per_day = 86400000000000
start_date = pd.to_datetime('January 1, 2013')


# ### Feature Engineering
# **Day of Week**
#  + Utilizing pandas builtin dayofweek call to create binary columns for each day of week.
# 
# **Month**
#  + Utilizing pandas buitlin month call.
#  
# **Day of Year**
#  + Take the number of days since the data started, then take (mod 364) for a like-for-like day of year.
#  + Want to do this instead of calendar day of year because the dates land on different days of the week. Example: January 1 might be a Tuesday one year so it'll be a Wednesday next year.
#  
# **Day of Quarter**
#  + Take the number of days since the data started, then take the quotient when divided by 91 and put into (mod 4).
#  + This is to give us similar quarters, same reasoning as above.
#  
# **Day Number**
#   + Number of days since start of train data.
#  
# **Quarter**
#  + Take the number of days since the data started, then take the quotient when divided by 91.
#  + This is to give us similar quarters.
#  
# **Is Weekend**
#  + Boolean value if the date falls on a weekend. This is because a majority of sales occur between Friday and Sunday.
#  
# **Item Quart**
#  + Which quartile trend the item most closely resembles. This comes from the dynamic time warping we had done for the ARIMA models.
#  
# **12 Month Lag**
#  + Prior year's sales (same store, same item, 364 days prior).
# 
# **Averages**
#  + Averages by item by store for:
#   + Day of Week by Quarter
#   + Day of Year

# In[ ]:


itm_quart = pd.DataFrame(fitted_items_WK, columns=['item','item_quart','item_metric'])

def add_feat(df, train_end_str):
    """Adds Features to DataFrame and Takes Averages for Dates Before train_end_str"""
    
    dataf = df
    
    dataf['Weekday'] = dataf.index.dayofweek
    dataf['Is_Mon'] = (dataf.index.dayofweek == 0) *1
    dataf['Is_Tue'] = (dataf.index.dayofweek == 1) *1
    dataf['Is_Wed'] = (dataf.index.dayofweek == 2) *1
    dataf['Is_Thu'] = (dataf.index.dayofweek == 3) *1
    dataf['Is_Fri'] = (dataf.index.dayofweek == 4) *1
    dataf['Is_Sat'] = (dataf.index.dayofweek == 5) *1
    dataf['Is_Sun'] = (dataf.index.dayofweek == 6) *1
    dataf['Is_wknd'] = dataf.index.dayofweek // 4 # Fri-Sun are 4-6, Monday is 0 so this is valid
    dataf['Day_Num'] = ((dataf.index - start_date)/ ns_per_day).astype(int)
    
    dataf['Month'] = dataf.index.month
    dataf['Day_of_Year'] = ((dataf.index - start_date)/ ns_per_day).astype(int) % 364
    dataf['Year'] = ((dataf.index - start_date)/ ns_per_day).astype(int) // 364 -1
    dataf['Day_of_Quarter'] = ((dataf.index - start_date)/ ns_per_day).astype(int) % 91
    dataf['Quarter'] = (((dataf.index - start_date)/ ns_per_day).astype(int) // 91) % 4
    dataf.reset_index(inplace=True)
    
    # Add item quartile as feature
    dataf = dataf.merge(itm_quart, on='item').drop('item_metric', axis=1)

    # Add prior year sales as additional feature
    prior_year_sales = dataf[['date','sales','store','item']]
    prior_year_sales['date'] += pd.Timedelta('364 days')
    prior_year_sales.columns =['date','lag_12mo','store','item']

    dataf = dataf.merge(prior_year_sales, on=['date','store','item'])
    
    # Add average by item by store by day of year as additional feature
    avg = dataf.loc[df['date'] < pd.to_datetime(train_end_str), ['Day_of_Year','sales','store','item']].groupby(by=['Day_of_Year','store','item']).mean().reset_index()
    avg.columns =['Day_of_Year','store','item','DoY_Mean']
    
    dataf = dataf.merge(avg, on=['Day_of_Year','store','item'])
    
    # Add average by day of week by quarter by item by store as additional feature
    avg = dataf.loc[df['date'] < pd.to_datetime(train_end_str), ['Quarter','Weekday','sales','store','item']].groupby(by=['Quarter','Weekday','store','item']).mean().reset_index()
    avg.columns =['Quarter','Weekday','store','item','DoW_Mean']
    
    dataf = dataf.merge(avg, on=['Quarter','Weekday','store','item'])
    
    # Id's start at 0 instead of 1
    dataf['store'] -=1
    dataf['item'] -=1
    
    # Remove first year of data as there is no prior year sales for them, then sort to match competition id's
    dataf = dataf[dataf['Year'] >=0].drop('Year', axis=1).sort_values(['item','store','date'])
    
    return dataf


# In[ ]:


df_test = add_feat(df, 'October 3, 2017') # Takes average of training data

df_test.tail(10)


# In[ ]:


df_test.head(10)


# ### Train & Test Data Split
# Split train and test data by setting the last 91 days (everything after October 3, 2017) as test data.

# In[ ]:


x_train = df_test.loc[df['date'] < pd.to_datetime('October 3, 2017')].drop(['sales','date','Day', 'Weekday'], axis=1)
y_train = df_test.loc[df['date'] < pd.to_datetime('October 3, 2017'), 'sales']

x_test = df_test.loc[df['date'] >= pd.to_datetime('October 3, 2017')].drop(['sales','date','Day', 'Weekday'], axis=1).reset_index(drop=True)
y_test = df_test.loc[df['date'] >= pd.to_datetime('October 3, 2017'), 'sales'].reset_index(drop=True)


# ### Feature Columns
# Setup the feature colunms in the tensorflow model. Most of the features are categorical, the only numeric one is 'lag_12mo'

# In[ ]:


feat_cols =[]

for col in x_train.drop(['lag_12mo','DoW_Mean','DoY_Mean'], axis=1).columns:
    feat_cols.append(tf.feature_column.embedding_column(
        tf.feature_column.categorical_column_with_identity(col, max(df_test[col])+1),1))
    
feat_cols.append(tf.feature_column.numeric_column(key='lag_12mo'))
feat_cols.append(tf.feature_column.numeric_column(key='DoY_Mean'))
feat_cols.append(tf.feature_column.numeric_column(key='DoW_Mean'))


# ### Training the Model & Forecasting
# Setup the training (input) function in tensorflow. Sending 6 months (180 days) of data to train on at once and will run through the entire dataset 80 times. We won't shuffle the observations for this exercise. Idea being that the order of observations matters since this is a time series.

# In[ ]:


input_func = tf.estimator.inputs.pandas_input_fn(x= x_train, y= y_train, batch_size= 180, num_epochs= 80,
                                                 shuffle= False)


# The model we'll use is tensorflow's builtin DNNRegressor with 3 hidden layers.

# In[ ]:


regressor = tf.estimator.DNNRegressor(hidden_units= [20, 10, 20], feature_columns= feat_cols)


# In[ ]:


regressor.train(input_fn= input_func)


# In[ ]:


pred_fn = tf.estimator.inputs.pandas_input_fn(x= x_test, batch_size =len(x_test), shuffle=False)


# In[ ]:


x_test.head()


# In[ ]:


predictions = list(regressor.predict(input_fn= pred_fn))


# ### Model Performance
# Gather predictions into a series then use SMAPE to compare with actual test values.

# In[ ]:


final_pred = []

for pred in predictions:
    final_pred.append(pred['predictions'][0])

final_pred = pd.DataFrame(final_pred)


# In[ ]:


SMAPE(final_pred.iloc[:,0], y_test)


# # XGBoost Model 
# ## Model (3) - Extreme Gradient Boost with Daily Data
# According to the competition description, this model should provide the best accuracy. Let's feed it the same data as the DNN and compare.

# In[ ]:


preds = xboost(x_train, y_train, x_test)


# ### Model Performance
# Compare forecasts to the actual test sales using SMAPE.

# In[ ]:


SMAPE(preds, y_test)


# # My Learnings
# 
# ## ARIMA
# Clustering the items to the nearest quartile keeps roughly the same accuracy as not clustering while taking less time to forecast. The models provided quick results although the least accurate of those tested.
# 
# ## DNN
# Likely due to the amount of data and how many times the model ran through all the data, the training sessions took a reltaively long time to run. However there was an accuracy boost compared to the ARIMA models. A different neural network structure (i.e. a deep and wide net) could possibly provide even better results.
# 
# ## XGBoost
# This model is a beast. It didn't take very long to train and tied for best accuracy. Can see why this model is preferred.

# # Competition Submission

# Using the same xgboost model with same feature engineering. This time we'll use the entire training data.

# In[ ]:


df1 = pd.read_csv(path +'train.csv', index_col=0)
df2 = pd.read_csv(path +'test.csv', index_col=1)

df2.head()


# In[ ]:


df = pd.concat([df1,df2])
df.index = pd.to_datetime(df.index)

df.tail()


# In[ ]:


df = add_feat(df, 'April 1, 2018') # Takes average of non-competition data

df.head(10)


# In[ ]:


df.tail(10)


# In[ ]:


x_train = df[pd.isnull(df['id'])].drop(['id','sales','date'], axis=1)
y_train = df[pd.isnull(df['id'])]['sales']

x_test = df[pd.notnull(df['id'])].drop(['id','sales','date'], axis=1)


# In[ ]:


preds = pd.DataFrame(xboost(x_train, y_train, x_test)).reset_index()
preds.columns =['id','sales']


# In[ ]:


preds.head()


# In[ ]:


preds.to_csv('sample.csv', index=False)


# # Thanks for Making It to the End!
# Thank you for sharing in my first competition! Hopefully you learned something as well. As this is my first competition and kernel, any feedback would be greatly appreciated.
