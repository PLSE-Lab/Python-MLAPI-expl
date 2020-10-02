#!/usr/bin/env python
# coding: utf-8

# ### 1. Timeseries Practice with Forex Dataset

# Purpose is to practice generating features on timeseries data on an easy dataset that does not run into memory issues.
# I am only going to predict 1 currency, the AUD/USD rate, and use other countries as inputs. 
# Otherwise you could just use pd.melt to stack the countries in order to generate predictions for all, if you want to see an example of this let me know in the comments.

# About the dataset, the data is foreign exchange rates from 2000 to 2019. 
# This data is reasonably clean and requires minimal data preparation and cleansing. We want to be able to make a prediction on the daily AUD/USD rate at the end of this notebook.
# 

# This notebook is useful for absolute beginners who are new to Python, through to intermediate users who I will introduce some more advanced features. The code deliberately avoids functions where possible to keep things easy to read for all levels.

# So let's get straight into it!

# ### 2. Install Packages

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #charting
from scipy.stats import mode #statistics for slope
from sklearn.metrics import mean_squared_error #error metric to optimise when we build a model
from math import sqrt #Other math functions
import plotly.express as px #alternative charting function
import lightgbm as lgb #popular model choice
import seaborn as sns #alternative charting function

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


#change display when printing .head for example the default settings only displays limited number of columns
pd.set_option('display.max_columns', 500)


# ### 3. Import & Cleanse Data

# In[ ]:


#The data we will be using is a foreign exhange rates dataset kindly provided to Kaggle. 
forex_df = pd.read_csv('/kaggle/input/foreign-exchange-rates-per-dollar-20002019/Foreign_Exchange_Rates.csv',engine = 'python')


# In[ ]:


#let's go ahead and preview the top or bottom n rows
forex_df.tail(6)


# You can see that there is 'ND' in the above example for 25th December, markets closed for Christmas, so we could either drop this or apply some cleansing to enable us to convert to a numeric field.

# In[ ]:


#you can see that the currencies are 'objects' which effectively means they are strings. We will need to convert later on to numeric to enable calculations.
forex_df.info()


# In[ ]:


#What is the number of rows, and columns in this dataset?
forex_df.shape


# #### 3.1 Cleanse data and Convert to Numeric Format

# In[ ]:


#create a list of all the currency columns in the dataset
currency_list = ['AUSTRALIA - AUSTRALIAN DOLLAR/US$','EURO AREA - EURO/US$','NEW ZEALAND - NEW ZELAND DOLLAR/US$','UNITED KINGDOM - UNITED KINGDOM POUND/US$','BRAZIL - REAL/US$','CANADA - CANADIAN DOLLAR/US$','CHINA - YUAN/US$','HONG KONG - HONG KONG DOLLAR/US$','INDIA - INDIAN RUPEE/US$','KOREA - WON/US$','MEXICO - MEXICAN PESO/US$','SOUTH AFRICA - RAND/US$','SINGAPORE - SINGAPORE DOLLAR/US$','DENMARK - DANISH KRONE/US$','JAPAN - YEN/US$','MALAYSIA - RINGGIT/US$','NORWAY - NORWEGIAN KRONE/US$','SWEDEN - KRONA/US$','SRI LANKA - SRI LANKAN RUPEE/US$','SWITZERLAND - FRANC/US$','TAIWAN - NEW TAIWAN DOLLAR/US$','THAILAND - BAHT/US$']
#cleanse data
for c in currency_list:
    #ffill simply takes the previous row and applies it to the next row. We have conditioned this to only be applied to non numeric data.
    forex_df[c] = forex_df[c].where(~forex_df[c].str.isalpha()).ffill()
    #we then want to convert the currency columns into numeric so that we can apply functions to it.
    forex_df[c] = pd.to_numeric(forex_df[c], errors='coerce') 


# In[ ]:


#Let's check that this actually did what we intended.
forex_df.tail()


# note that the value for 24th Dec is carried forward to 25th Dec for those countries, but not for others that did not have this holiday.

# In[ ]:


#let's check that the columns are now numeric, yep that worked!
forex_df.info()


# ### 4. Generate Features

# Generally with timeseries there are two main approaches. One being getting the data into a stationary format, accounting for the trend and seasonality. You then apply more traditional models you may have come across such as 'ARIMA', 'GARCH' etc. In this case, we won't be doing that, but will be applying machine learning models to the data directly without the need for this step which is simpler.

# #### 4.1 Date Features

# In[ ]:


#generate features

# time features
forex_df['date'] = pd.to_datetime(forex_df['Time Serie'])
forex_df['year'] = forex_df['date'].dt.year
forex_df['month'] = forex_df['date'].dt.month
forex_df['week'] = forex_df['date'].dt.week
forex_df['day'] = forex_df['date'].dt.day
forex_df['dayofweek'] = forex_df['date'].dt.dayofweek


# #### 4.2 Lag features

# We need to shift the data by 1 or more days, so that we can use yesterday's data to predict today and so on. Later on we will remove today's data as we don't want to cause 'data leakage' whereby our model has information available to it that is not known at the time, which would not help it to work in practice.

# In[ ]:


# lag features
forex_df['lag_t1'] = forex_df['AUSTRALIA - AUSTRALIAN DOLLAR/US$'].transform(lambda x: x.shift(1))
forex_df['lag_t3'] = forex_df['AUSTRALIA - AUSTRALIAN DOLLAR/US$'].transform(lambda x: x.shift(3))
forex_df['lag_t7'] = forex_df['AUSTRALIA - AUSTRALIAN DOLLAR/US$'].transform(lambda x: x.shift(7))


# We will also use other countries values from yesterday to aid in predicting today's data

# In[ ]:


# lag other country features
for c in [x for x in currency_list if x != "AUSTRALIA - AUSTRALIAN DOLLAR/US$"]:
    forex_df['lag_t1_%s' % c] = forex_df[c].transform(lambda x: x.shift(1))


# We may want to add in a ratio as the raw value for GBP may not be as stable as a ratio value

# In[ ]:


# ratio lag other country features
for c in [x for x in currency_list if x != "AUSTRALIA - AUSTRALIAN DOLLAR/US$"]:
    forex_df['lag_t1_ratio_%s' % c] = forex_df['lag_t1']  / forex_df['lag_t1_' + c] 


# In[ ]:


forex_df.tail()


# #### 4.3 Rolling Features

# With rolling features you can set min_periods as 1, that way you don't lose that much data (if you were to drop nulls in a subsequent step) as for example a 7 day rolling average the previous 6 days would be 'n/a'. This may help especially for longer lags e.g.365 days.

# So let's go ahead and create a bunch of rolling features across different days and metrics

# In[ ]:


#rolling features
#mean
forex_df['rolling_mean_t1_t7'] = forex_df['lag_t1'].rolling(7,min_periods=1).mean()
forex_df['rolling_mean_t1_t14'] = forex_df['lag_t1'].rolling(14,min_periods=1).mean()
forex_df['rolling_mean_t1_t28'] = forex_df['lag_t1'].rolling(28,min_periods=1).mean()
forex_df['rolling_mean_t1_t90'] = forex_df['lag_t1'].rolling(90,min_periods=1).mean()
forex_df['rolling_mean_t1_t180'] = forex_df['lag_t1'].rolling(180,min_periods=1).mean()
forex_df['rolling_mean_t1_t360'] = forex_df['lag_t1'].rolling(360,min_periods=1).mean()

#max
forex_df['rolling_max_t1_t7'] = forex_df['lag_t1'].rolling(7,min_periods=1).max()
forex_df['rolling_max_t1_t14'] = forex_df['lag_t1'].rolling(14,min_periods=1).max()
forex_df['rolling_max_t1_t28'] = forex_df['lag_t1'].rolling(28,min_periods=1).max()
forex_df['rolling_max_t1_t90'] = forex_df['lag_t1'].rolling(90,min_periods=1).max()
forex_df['rolling_max_t1_t180'] = forex_df['lag_t1'].rolling(180,min_periods=1).max()
forex_df['rolling_max_t1_t360'] = forex_df['lag_t1'].rolling(360,min_periods=1).max()

#min
forex_df['rolling_min_t1_t7'] = forex_df['lag_t1'].rolling(7,min_periods=1).min()
forex_df['rolling_min_t1_t14'] = forex_df['lag_t1'].rolling(14,min_periods=1).min()
forex_df['rolling_min_t1_t28'] = forex_df['lag_t1'].rolling(28,min_periods=1).min()
forex_df['rolling_min_t1_t90'] = forex_df['lag_t1'].rolling(90,min_periods=1).min()
forex_df['rolling_min_t1_t180'] = forex_df['lag_t1'].rolling(180,min_periods=1).min()
forex_df['rolling_min_t1_t360'] = forex_df['lag_t1'].rolling(360,min_periods=1).min()

#standard deviation
forex_df['rolling_std_t1_t7'] = forex_df['lag_t1'].rolling(7,min_periods=1).std()
forex_df['rolling_std_t1_t14'] = forex_df['lag_t1'].rolling(14,min_periods=1).std()
forex_df['rolling_std_t1_t28'] = forex_df['lag_t1'].rolling(28,min_periods=1).std()
forex_df['rolling_std_t1_t90'] = forex_df['lag_t1'].rolling(90,min_periods=1).std()
forex_df['rolling_std_t1_t180'] = forex_df['lag_t1'].rolling(180,min_periods=1).std()
forex_df['rolling_std_t1_t360'] = forex_df['lag_t1'].rolling(360,min_periods=1).std()

#median
forex_df['rolling_med_t1_t7'] = forex_df['lag_t1'].rolling(7,min_periods=1).median()
forex_df['rolling_med_t1_t14'] = forex_df['lag_t1'].rolling(14,min_periods=1).median()
forex_df['rolling_med_t1_t28'] = forex_df['lag_t1'].rolling(28,min_periods=1).median()
forex_df['rolling_med_t1_t90'] = forex_df['lag_t1'].rolling(90,min_periods=1).median()
forex_df['rolling_med_t1_t180'] = forex_df['lag_t1'].rolling(180,min_periods=1).median()
forex_df['rolling_med_t1_t360'] = forex_df['lag_t1'].rolling(360,min_periods=1).median()


# Exponential moving averages provide more weight to recent values, which in finance are generally useful. 

# In[ ]:


# exponential moving averages
forex_df['rolling_ema_t1_t7'] = forex_df['lag_t1'].ewm(span=7,adjust=False).mean()
forex_df['rolling_ema_t1_t14'] = forex_df['lag_t1'].ewm(span=14,adjust=False).mean()
forex_df['rolling_ema_t1_t28'] = forex_df['lag_t1'].ewm(span=28,adjust=False).mean()
forex_df['rolling_ema_t1_t90'] = forex_df['lag_t1'].ewm(span=90,adjust=False).mean()
forex_df['rolling_ema_t1_t180'] = forex_df['lag_t1'].ewm(span=180,adjust=False).mean()
forex_df['rolling_ema_t1_t360'] = forex_df['lag_t1'].ewm(span=360,adjust=False).mean()


# In[ ]:


#Take a quick look at the data over time now that we have some features to compare against:
# This is a relatively easy method to plot multiple values on a line chart plus it allows you to dynamically interact with the chart
df_long=pd.melt(forex_df, id_vars=['date'], value_vars=['AUSTRALIA - AUSTRALIAN DOLLAR/US$', 'rolling_ema_t1_t7', 'rolling_mean_t1_t7', 'rolling_ema_t1_t360', 'rolling_med_t1_t360'])

# plotly 
fig = px.line(df_long, x='date', y='value', color='variable')

# Show plot 
fig.show()


# #### 4.4 Other Features - Decimals, Rounding, Mode, Coefficient of Variation

# It's interesting to try decimal or rounded features if you remember, it can often give a small boost to results

# In[ ]:


#round the value to 0 decimals
forex_df['lag_t1_round_0'] = forex_df['lag_t1'].round(0)
forex_df['lag_t3_round_0'] = forex_df['lag_t3'].round(0)
forex_df['lag_t7_round_0'] = forex_df['lag_t7'].round(0)

#get the decimal place
forex_df['lag_t1_dec'] = forex_df['lag_t1'] - forex_df['lag_t1_round_0']
forex_df['lag_t3_dec'] = forex_df['lag_t3'] - forex_df['lag_t3_round_0']
forex_df['lag_t7_dec'] = forex_df['lag_t7'] - forex_df['lag_t7_round_0']

#round the value to 1 decimals, as the rounded value to 0 decimals is nearly always 1 in the case of AUD/USD
forex_df['lag_t1_round_1'] = forex_df['lag_t1'].round(1)
forex_df['lag_t3_round_1'] = forex_df['lag_t3'].round(1)
forex_df['lag_t7_round_1'] = forex_df['lag_t7'].round(1)


# Mode is often overlooked, don't forget to give it a try when tackling your next problem

# In[ ]:


#rolling mode of rounded figure
forex_df['lag_t1_mode_7'] = forex_df['lag_t1_round_1'].rolling(window=7,min_periods=1).apply(lambda x: mode(x)[0])
forex_df['lag_t1_mode_14'] = forex_df['lag_t1_round_1'].rolling(window=14,min_periods=1).apply(lambda x: mode(x)[0])
forex_df['lag_t1_mode_28'] = forex_df['lag_t1_round_1'].rolling(window=28,min_periods=1).apply(lambda x: mode(x)[0])
forex_df['lag_t1_mode_90'] = forex_df['lag_t1_round_1'].rolling(window=90,min_periods=1).apply(lambda x: mode(x)[0])
forex_df['lag_t1_mode_180'] = forex_df['lag_t1_round_1'].rolling(window=180,min_periods=1).apply(lambda x: mode(x)[0])
forex_df['lag_t1_mode_360'] = forex_df['lag_t1_round_1'].rolling(window=360,min_periods=1).apply(lambda x: mode(x)[0])


# In[ ]:


#frequency of mode


# In[ ]:


#ranges
forex_df['rolling_range_t1_t7'] = forex_df['rolling_max_t1_t7'] - forex_df['rolling_min_t1_t7']
forex_df['rolling_range_t1_t14'] = forex_df['rolling_max_t1_t14'] - forex_df['rolling_min_t1_t14']
forex_df['rolling_range_t1_t28'] = forex_df['rolling_max_t1_t28'] - forex_df['rolling_min_t1_t28']
forex_df['rolling_range_t1_t90'] = forex_df['rolling_max_t1_t90'] - forex_df['rolling_min_t1_t90']
forex_df['rolling_range_t1_t180'] = forex_df['rolling_max_t1_t180'] - forex_df['rolling_min_t1_t180']
forex_df['rolling_range_t1_t360'] = forex_df['rolling_max_t1_t360'] - forex_df['rolling_min_t1_t360']


# In[ ]:


#coefficient of variation - the ratio of standard deviation to mean
forex_df['rolling_coefvar_t1_t7'] =  forex_df['rolling_std_t1_t7'] / forex_df['rolling_mean_t1_t7']
forex_df['rolling_coefvar_t1_t14'] = forex_df['rolling_std_t1_t14'] / forex_df['rolling_mean_t1_t14']
forex_df['rolling_coefvar_t1_t28'] = forex_df['rolling_std_t1_t28'] / forex_df['rolling_mean_t1_t28']
forex_df['rolling_coefvar_t1_t90'] = forex_df['rolling_std_t1_t90'] / forex_df['rolling_mean_t1_t90']
forex_df['rolling_coefvar_t1_t180'] = forex_df['rolling_std_t1_t180'] / forex_df['rolling_mean_t1_t180']
forex_df['rolling_coefvar_t1_t360'] = forex_df['rolling_std_t1_t360'] / forex_df['rolling_mean_t1_t360']


# In[ ]:


#ratio of change to standard deviation
#I like this because if the currency is normally volatile (high std dev), then a change in the rolling mean may be normal. 
#On the other hand if the currency is not normally volatile (low std dev), then it adds weight to any changes observed
forex_df['rolling_meanstd_t1_t14'] = (forex_df['rolling_mean_t1_t7'] - forex_df['rolling_mean_t1_t14']) / forex_df['rolling_std_t1_t14']
forex_df['rolling_meanstd_t1_t28'] = (forex_df['rolling_mean_t1_t7'] - forex_df['rolling_mean_t1_t28']) / forex_df['rolling_std_t1_t28']
forex_df['rolling_meanstd_t1_t90'] = (forex_df['rolling_mean_t1_t7'] - forex_df['rolling_mean_t1_t90']) / forex_df['rolling_std_t1_t90']
forex_df['rolling_meanstd_t1_t180'] = (forex_df['rolling_mean_t1_t7'] - forex_df['rolling_mean_t1_t180']) / forex_df['rolling_std_t1_t180']
forex_df['rolling_meanstd_t1_t360'] = (forex_df['rolling_mean_t1_t7'] - forex_df['rolling_mean_t1_t360']) / forex_df['rolling_std_t1_t360']


# Cardinality is the number of unique values, e.g. [0,0,0,1,1] has 2 unique values

# In[ ]:


#cardinality
forex_df['lag_t1_card_180'] = forex_df['lag_t1_round_1'].rolling(window=180,min_periods=1).apply(lambda x: np.unique(x).shape[0])
forex_df['lag_t1_card_360'] = forex_df['lag_t1_round_1'].rolling(window=360,min_periods=1).apply(lambda x: np.unique(x).shape[0])


# 

# #### 4.5 Trends

# If the shorter moving average crosses over a longer one, it could be a trend indicator

# In[ ]:


#moving average crossover trends, 1 = positive, 0 = negative
forex_df['lag_t1_trend_7'] = np.where(forex_df['lag_t1'] >= forex_df['rolling_ema_t1_t7'],1,0)
forex_df['lag_t1_trend_14'] = np.where(forex_df['rolling_ema_t1_t7'] >= forex_df['rolling_ema_t1_t14'],1,0)
forex_df['lag_t1_trend_28'] = np.where(forex_df['rolling_ema_t1_t7'] >= forex_df['rolling_ema_t1_t28'],1,0)
forex_df['lag_t1_trend_90'] = np.where(forex_df['rolling_ema_t1_t7'] >= forex_df['rolling_ema_t1_t90'],1,0)
forex_df['lag_t1_trend_180'] = np.where(forex_df['rolling_ema_t1_t7'] >= forex_df['rolling_ema_t1_t180'],1,0)
forex_df['lag_t1_trend_360'] = np.where(forex_df['rolling_ema_t1_t7'] >= forex_df['rolling_ema_t1_t360'],1,0)


# In[ ]:


#number of crossovers last n days
forex_df['lag_t1_no_crossover_7'] = forex_df['lag_t1_trend_7'].rolling(window=7,min_periods=1).sum()
forex_df['lag_t1_no_crossover_14'] = forex_df['lag_t1_trend_14'].rolling(window=14,min_periods=1).sum()
forex_df['lag_t1_no_crossover_28'] = forex_df['lag_t1_trend_28'].rolling(window=28,min_periods=1).sum()
forex_df['lag_t1_no_crossover_90'] = forex_df['lag_t1_trend_90'].rolling(window=90,min_periods=1).sum()
forex_df['lag_t1_no_crossover_180'] = forex_df['lag_t1_trend_180'].rolling(window=180,min_periods=1).sum()
forex_df['lag_t1_no_crossover_360'] = forex_df['lag_t1_trend_360'].rolling(window=360,min_periods=1).sum()


# In[ ]:


#decay


# In[ ]:


#slope or 1st derivative
forex_df['lag_t1_slope_7'] = forex_df['lag_t1'].rolling(7).apply(lambda x: np.polyfit(range(7), x, 1)[0]).values
forex_df['lag_t1_slope_14'] = forex_df['lag_t1'].rolling(14).apply(lambda x: np.polyfit(range(14), x, 1)[0]).values
forex_df['lag_t1_slope_28'] = forex_df['lag_t1'].rolling(28).apply(lambda x: np.polyfit(range(28), x, 1)[0]).values
forex_df['lag_t1_slope_90'] = forex_df['lag_t1'].rolling(90).apply(lambda x: np.polyfit(range(90), x, 1)[0]).values
forex_df['lag_t1_slope_180'] = forex_df['lag_t1'].rolling(180).apply(lambda x: np.polyfit(range(180), x, 1)[0]).values
forex_df['lag_t1_slope_360'] = forex_df['lag_t1'].rolling(360).apply(lambda x: np.polyfit(range(360), x, 1)[0]).values


# In[ ]:


#2nd derivative, slope of the 1st derivative, again for detecting trend changes
forex_df['lag_t1_deriv2_7'] = forex_df['lag_t1_slope_7'].rolling(7).apply(lambda x: np.polyfit(range(7), x, 1)[0]).values
forex_df['lag_t1_deriv2_14'] = forex_df['lag_t1_slope_7'].rolling(14).apply(lambda x: np.polyfit(range(14), x, 1)[0]).values
forex_df['lag_t1_deriv2_28'] = forex_df['lag_t1_slope_7'].rolling(28).apply(lambda x: np.polyfit(range(28), x, 1)[0]).values
forex_df['lag_t1_deriv2_90'] = forex_df['lag_t1_slope_7'].rolling(90).apply(lambda x: np.polyfit(range(90), x, 1)[0]).values
forex_df['lag_t1_deriv2_180'] = forex_df['lag_t1_slope_7'].rolling(180).apply(lambda x: np.polyfit(range(180), x, 1)[0]).values
forex_df['lag_t1_deriv2_360'] = forex_df['lag_t1_slope_7'].rolling(360).apply(lambda x: np.polyfit(range(360), x, 1)[0]).values


# In[ ]:


forex_df.shape


# ### 5. Prepare Dataset for Model

# In[ ]:


#We have a heap of features:
list(forex_df.columns)


# #### 5.1 Drop unnecessary columns

# In[ ]:



#Create a list of the features to drop, as previously mentioned we can't use the feature from today else it would cause target leakage - the model knows something that it can't know in advance.
useless_cols = ['Unnamed: 0', 
                "date", 
                'AUSTRALIA - AUSTRALIAN DOLLAR/US$',
                'Time Serie', 
                'EURO AREA - EURO/US$',
                 'NEW ZEALAND - NEW ZELAND DOLLAR/US$',
                 'UNITED KINGDOM - UNITED KINGDOM POUND/US$',
                 'BRAZIL - REAL/US$',
                 'CANADA - CANADIAN DOLLAR/US$',
                 'CHINA - YUAN/US$',
                 'HONG KONG - HONG KONG DOLLAR/US$',
                 'INDIA - INDIAN RUPEE/US$',
                 'KOREA - WON/US$',
                 'MEXICO - MEXICAN PESO/US$',
                 'SOUTH AFRICA - RAND/US$',
                 'SINGAPORE - SINGAPORE DOLLAR/US$',
                 'DENMARK - DANISH KRONE/US$',
                 'JAPAN - YEN/US$',
                 'MALAYSIA - RINGGIT/US$',
                 'NORWAY - NORWEGIAN KRONE/US$',
                 'SWEDEN - KRONA/US$',
                 'SRI LANKA - SRI LANKAN RUPEE/US$',
                 'SWITZERLAND - FRANC/US$',
                 'TAIWAN - NEW TAIWAN DOLLAR/US$',
                 'THAILAND - BAHT/US$']

#define train columns to use in model
train_cols = forex_df.columns[~forex_df.columns.isin(useless_cols)]

#Let's simply use historical data up until Oct 2019
x_train = forex_df[forex_df['date'] <= '2019-10-31']
#The variable we want to predict is AUD to USD rate.
y_train = x_train['AUSTRALIA - AUSTRALIAN DOLLAR/US$']

#The LGBM model needs a train and validation dataset to be fed into it, let's use Nov 2019
x_val = forex_df[(forex_df['date'] > '2019-10-31') & (forex_df['date'] <= '2019-11-30')]
y_val = x_val['AUSTRALIA - AUSTRALIAN DOLLAR/US$']

#We shall test the model on data it hasn't seen before or been used in the training process
test = forex_df[(forex_df['date'] > '2019-12-01')]

#Setup the data in the necessary format the LGB requires
train_set = lgb.Dataset(x_train[train_cols], y_train)
val_set = lgb.Dataset(x_val[train_cols], y_val)
 


# ### 6.Model

# Light GBM will be used to build this model.

# Light GBM is a gradient boosting framework that uses tree based learning algorithms. It is designed to be distributed and efficient with the following advantages:
# 
#     Faster training speed and higher efficiency.
# 
#     Lower memory usage.
# 
#     Better accuracy.
# 
#     Support of parallel and GPU learning.
# 
#     Capable of handling large-scale data.
# 

# In[ ]:


#Set the model parameters
params = {
        "objective" : "regression", # regression is the type of business case we are running
        "metric" :"rmse", #root mean square error is a standard metric to use
#         "force_row_wise" : True,
        "learning_rate" : 0.05, #the pace at which the model is allowed to reach it's objective of minimising the rsme.
#         "sub_feature" : 0.8,
#         "sub_row" : 0.75,
#         "bagging_freq" : 1,
        "lambda_l1" : 0.1, #lambda_l1 worked better than l2 in this case, as we have high number of features this makes sense (L1 or Lasso reduces some terms to 0 weight, whereas L2 or ridge includes all)
#         "nthread" : 4
        'verbosity': 1, 
#         'num_iterations' : 300,
        'num_leaves': 100, # minimum number of leaves in each boosting round
        "min_data_in_leaf": 25, #minimum amount of data in the leaf nodes or last value of the tree
        "early_stopping": 50, #if the model does not improve after this many consecutive rounds, call a halt to training
#         "max_bin" = 
        "sub_sample" : 0.025, #sampling feature to reduce overfitting
#         "boosting":"dart",
}


# In[ ]:


#Run the model
m_lgb = lgb.train(params, train_set, num_boost_round = 2500, valid_sets = [train_set, val_set], verbose_eval = 50)


# ### 7. Model Interpretation and Performance

# #### 7.1 Model Interpretation

# In[ ]:


#plot feature importance
feature_imp = pd.DataFrame({'Value':m_lgb.feature_importance(),'Feature':train_cols})
plt.figure(figsize=(20, 10))
sns.set(font_scale = 1)
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", 
                                                    ascending=False)[0:40])
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.savefig('lgbm_importances-01.png')
plt.show()


# This shows how many times the feature was used by the model. 
# <br>We can see that the model did like a lot of the slope and derivative type features generated at the end.
# <br>The model also liked the ratio of other currencies a fair amount. NB when I ran with these excluded it did not make much difference to performance. 
# 

# #### 7.2 Predictions on Test Data for out of time performance

# In[ ]:


#generate predictions on test data
y_pred = m_lgb.predict(test[train_cols])
test['AUSTRALIA - AUSTRALIAN DOLLAR/US$_pred'] = y_pred


# In[ ]:


test.head()


# In[ ]:


#view the test data in chart form
df_long=pd.melt(test, id_vars=['date'], value_vars=['AUSTRALIA - AUSTRALIAN DOLLAR/US$', 'AUSTRALIA - AUSTRALIAN DOLLAR/US$_pred'])

# plotly 
fig = px.line(df_long, x='date', y='value', color='variable')

# Show plot 
fig.show()


# We can see that the model generally captures trends well, there will always be spikes and volatility that is hard to predict.

# #### 7.3 Model Performance

# In[ ]:


#RSME metric
rms = sqrt(mean_squared_error(test['AUSTRALIA - AUSTRALIAN DOLLAR/US$'], test['AUSTRALIA - AUSTRALIAN DOLLAR/US$_pred']))


# In[ ]:


rms


# ### Summary

# Thanks for viewing this notebook if you made it this far, I hope you got some ideas and please share in comments below any suggestions or questions you may have to improve and learn on your journey!

# In[ ]:




