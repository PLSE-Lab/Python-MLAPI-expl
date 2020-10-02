#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

import xgboost as xgb
from sklearn.metrics import mean_squared_error

import os
print(os.listdir("../input"))


# ***For the initial data analysis, some lines of data are copied from PJME_hourly.csv ***
# * Datetime,PJME_MW
# * 2002-12-31 01:00:00,26498.0
# * 2002-12-31 02:00:00,25147.0
# * 2002-12-31 03:00:00,24574.0
# * 2002-12-31 04:00:00,24393.0
# * 2002-12-31 05:00:00,24860.0
# * 2002-12-31 06:00:00,26222.0
# 
# From this data sample, we can understand about the data and its format.

# In[ ]:


# As per the inital analysis, data read from PJME_hourly.csv. 
#   timestamp column set as index column
#   and speccifying date parsing in column 0.
#   default delimeter used
data = pd.read_csv('../input/PJME_hourly.csv', index_col=[0], parse_dates=[0])


# In[ ]:


# Verifying the loaded data
data.head()


# ### Data Cleaning
# * Check the data types
# * Check the missing data
#     * If the data is missing, fix it

# In[ ]:


data.info()


# In[ ]:


data.describe()


# In[ ]:


# to get the null value of PJME_MW column
data['PJME_MW'].isnull().sum()


# No missing data found. And statistically data is good.

# In[ ]:


_ = data.plot( style='.', figsize=( 15, 7 ), title='PJM East Energy consumption' )
plt.ylabel ( 'MW' )


# In[ ]:


# looks like MW field is integer, not float
data['PJME_MW'] = data['PJME_MW'].astype('int')
data.info()


# As per the graph from 2012 -2014, we can see a dip in energy consumption. Need to analyse the reason behind that.

# In[ ]:


data['date'] = data.index
data_2012 = data.loc[np.logical_and( np.logical_and( data['date'].dt.year == 2012 , data['date'].dt.month == 10), data['date'].dt.day > 20  )]
data_2012.info()
data_2012.head()


# In[ ]:


_ = data_2012['PJME_MW'].plot( style='.', figsize=( 15, 7 ), title='PJM East Energy consumption' )


# In[ ]:


data_2012.tail( 30)


# In[ ]:


# will check the same type of issue is there in other data or not.
data_deok = pd.read_csv('../input/DEOK_hourly.csv', index_col=[0], parse_dates=[0])
_ = data_deok.plot( style='.', figsize=( 15, 7 ), title='DEOK Energy consumption' )


# In[ ]:


# DOM_hourly
data_dom = pd.read_csv('../input/DOM_hourly.csv', index_col=[0], parse_dates=[0])
_ = data_dom.plot( style='.', figsize=( 15, 7 ), title='DOM Energy consumption' )


# In[ ]:


# NI_hourly
data_ni = pd.read_csv('../input/NI_hourly.csv', index_col=[0], parse_dates=[0])
_ = data_ni.plot( style='.', figsize=( 15, 7 ), title='NI Energy consumption' )


# > > **Analysis details of energey consumption *dip on 2012-10-30* in PJME data**
# * As per the graph and data analysis, the data is real. No missing data found
# * As per the date, I didn't find any specific reason for this dip 
# * In other regions data also, similar dip is there. But its not consistent. 
#     * Might be this data is an aggregation of some sub regions. If any sub region data is missing, these type of scenario can occur.

# **Feature Analysis**

# In[ ]:


data.head(20)


# From the Timestamp, we can create lot of new features( year, month, hour, day of month, day of week .. ) for the better prediction.

# In[ ]:


data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['dayofmonth'] = data['date'].dt.day
data['quarter'] = data['date'].dt.quarter
data['dayofyear'] = data['date'].dt.dayofyear
data['weekofyear'] = data['date'].dt.weekofyear
data['dayofweek'] = data['date'].dt.dayofweek
data['hour'] = data['date'].dt.hour


# In[ ]:


data.info()


# In[ ]:


data = data.sort_index()
data.head()


# In[ ]:


# the date column is parsed to multiple features. So that column is not required
data.drop( columns = 'date', inplace = True )


# I tried to add holidays also to the feature list. But its not giving any improvement in the model. So I commented it.

# In[ ]:


# from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
# from datetime import timedelta
# one_day = timedelta(days=1)
# cal = calendar()
# data['is_holiday'] = data.index.isin([d.date() for d in cal.holidays()])
# data['is_prev_holiday'] = data.index.isin([( d.date() + one_day ) for d in cal.holidays()])
# data['is_next_holiday'] = data.index.isin([( d.date() - one_day ) for d in cal.holidays()])
# data['is_holiday'] = data['is_holiday'].astype('category')
# data['is_prev_holiday'] = data['is_prev_holiday'].astype('category')
# data['is_next_holiday'] = data['is_next_holiday'].astype('category')


# In[ ]:


data.info()


# All the above independent variables are like category type. But no benfit to the model. So this also commented.

# In[ ]:


# New Category variables created
#data = pd.get_dummies( data,
#                       columns=[ 'year','month','dayofmonth','quarter','dayofyear', 'weekofyear', 'dayofweek','hour'])


# In[ ]:


# New Category variables created
# data = pd.get_dummies( data,
#                        columns=[ 'is_holiday','is_prev_holiday', 'is_next_holiday'])


# **Data Splitting**

# In[ ]:


def split_data( data, split_date ):
    return data[data.index <= split_date].copy(),            data[data.index >  split_date].copy()


# In[ ]:


train, test = split_data( data, '01-Jan-2015')

plt.figure( figsize=( 15, 7 ))
plt.xlabel('Time')
plt.ylabel('Energy consumed ( MW )')
plt.plot(train.index,train['PJME_MW'], label='train data' )
plt.plot(test.index,test['PJME_MW'], label='test data')
plt.title( 'Energy Comsumed - train and test data' )
plt.legend()
plt.show()


# In[ ]:


X_train = train
y_train = train['PJME_MW']
X_test = test
y_test = test['PJME_MW']
X_train.drop( columns = 'PJME_MW', inplace = True )
X_test.drop( columns = 'PJME_MW', inplace = True )


# **Applying XGBoost Model**

# In[ ]:


model = xgb.XGBRegressor(  n_estimators = 250,
                           max_depth= 5,
                           learning_rate= 0.069,
                           subsample=1,
                           colsample_bytree=1)


# **For fine-tune the model, the hyper parameters are tried with reasonable values.**
# * Based on the MSE, the best suitable parmeters are set.
# *     Hyper parmeter tried values are given below
#     * max_depth = 3,5,6,8
#     * learning_rate = 0.01, 0.02, 0.05, 0.08, 0.07, 0.069, 0.065, 0.071,0.69
#     * subsample = 0.8, 0.9, 0.95, 1
# 
# 

# In[ ]:


model.fit( X_train, y_train,
           eval_set=[ ( X_train, y_train ), ( X_test, y_test )],
           early_stopping_rounds = 100, # stop if 50 consequent rounds without decrease of error
           verbose = True ) # Change verbose to True if you want to see it train


# **Predicting using test data**

# In[ ]:


X_test_pred = model.predict(X_test)
pd.DataFrame(X_test_pred).head()


# **Measure the Mean Square Error**

# In[ ]:


mean_squared_error(y_true=y_test,
                   y_pred=X_test_pred)


# In[ ]:


def mean_absolute_percentage_error(y_true, y_pred): 
    """Calculates MAPE given y_true and y_pred"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# In[ ]:


mean_absolute_percentage_error(y_test,X_test_pred)


# **Feature importance**

# In[ ]:


def plot_performance(base_data, test_data, test_pred, date_from, date_to, title=None):
    plt.figure(figsize=(15,6))
    if title == None:
        plt.title('From {0} To {1}'.format(date_from, date_to))
    else:
        plt.title( title )
    plt.xlabel( 'Time' )
    plt.ylabel( 'Energy consumed ( MW )' )
    plt.plot( base_data.index,base_data['PJME_MW'], label='data' )
    plt.plot( test_data.index, test_pred, label='prediction' )
    plt.xlim( left=date_from, right=date_to )
    plt.legend()


# In[ ]:


xgb.plot_importance( model, height = 0.9, max_num_features= 10 )


# ****Plot the predicitions****

# In[ ]:


plot_performance(data, X_test, X_test_pred, data.index[0].date(), data.index[-1].date(),
                 'Original and Predicted Data')

plot_performance(data, X_test, X_test_pred, y_test.index[0].date(), y_test.index[-1].date(),
                 'Test and Predicted Data')

plot_performance(data, X_test, X_test_pred, '01-01-2015', '12-01-2015', '2015 Snapshot')
plot_performance(data, X_test, X_test_pred, '01-01-2016', '12-01-2016', '2016 Snapshot')
plot_performance(data, X_test, X_test_pred, '01-01-2017', '12-01-2017', '2017 Snapshot')
plot_performance(data, X_test, X_test_pred, '01-01-2018', '08-01-2018', '2018 Snapshot')

plot_performance(data, X_test, X_test_pred, '03-01-2018', '04-01-2018', '2018 March Snapshot')
plot_performance(data, X_test, X_test_pred, '03-01-2018', '03-11-2018', '2018 March Ten Days Snapshot')
plot_performance(data, X_test, X_test_pred, '03-01-2018', '03-02-2018', '2018 March one day Snapshot')

plt.show()

