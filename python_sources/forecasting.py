#!/usr/bin/env python
# coding: utf-8

# # Introduction

# I have just joined this compete for a week. In my observation, almost of kernel which has better score using Light GBM with lag and aggregate feature. Some kernel used NN but not effective.
# 
# Light GBM is good approach. There are many kernels forking, optimizing. In my opinion, it has not enough room to improve because of time and memory to train model limitation. I submit a new but very simple, clear and all in one approach. It is quick and easy to optimize because we can optimize per each series. Currently, it takes more than 3 hours on CPU to complete.
# 
# * Why Prophet: It can detect abnormal, trend and seasonality very well. It support exogenous feature. 
# * Why simple MA: There are many id, which have zero or low sale item in last day. So do not need complex algorithm. We need save time for other id forecast.
# * Why not ARIMA: I cannot choose suite orders (p, d, q) for all series. Cannot use Auto ARIMA due to runtime limitation
# 
# Because of WRMSSE evaluation, I tried algorithm selection by amount of sale but it is no better. I will try again later.
# 
# I hope there are many kernel will fork, optimize this approach. So, I can learn more than from community. Please comment your opinion about this approach. It help me turning better. Thanks.
# 
# EDA
# * https://www.kaggle.com/binhlc/m5-forecasting-accuracy-eda
# * https://www.kaggle.com/binhlc/prophet-hyperparameter-tuning
# 
# My deep learning approach (working only on GPU and not good score)
# * https://www.kaggle.com/binhlc/high-dimensional-time-series-forecasting-with-tf2
# 
# Reference
# * Time series with exogenous features https://mc.ai/time-series-analysis-arima-or-prophet/
# * https://facebook.github.io/prophet/docs/quick_start.html
# 
# To do:
# * Using exogenous feature (sale, event, store information)
# * Data cleaning
# * Turning both Prophet and MA (base on comparing result to Light GBM)
# * Optimize algorithm by series characteristic (algorithm selection)
# 
# Change log:
# * Version 31: Reset algorithm
# * Version 30: Apply 4 algorithm
# * Version 28: Change MA by Prophet Top - Down with hyper parameter
# * Version 26: Rollback to MA, apply hyper parameter on prophet with holiday, reset manual tunning
# * Version 25: Change MA by Prophet Top - Down
# * Version 24: Turning by series
# * Version 23: Add manual select MA algorithm
# * Version 22: Change MA forecast by week day
# * Version 21: Fixed error on static ids list
# * Version 20: Turning by series
# * Version 18: Turning by series
# * Version 14: Add plot forecast for random id
# * Version 13: Apply changepoint_range to adjust trend. Increase Prophet forecast (est 5 hours to compleate)
# * Version 12: Reduce prophet forecast, apply uncertainty_samples
# * Version 11: Apply holiday and changepoint_prior_scale
# * Version 10: Keep noise on Prophet (for score)
# * Version 09: Rollback yearly_seasonality = True. Select algorithm by last 14 day sale
# * Version 08: Select algorithm by last 28 day sale
# * Version 07: Change yearly_seasonality = false and reduce prophet forecast, fix negative forecast
# * Version 06: Fixed MA forecast
# * Version 05: Remove noise (try with z score)
# * Version 04: Decrease prophet forecast
# * Version 03: Remove zero sale
# * Version 02: Turning model selection
# * Version 01: Apply simple MA 
# * Version 00: Prophet predict
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df_sale = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')
df_calendar = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')
df_price = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv')


# In[ ]:


from fbprophet import Prophet
columns = df_sale.columns
date_columns = columns[columns.str.contains("d_")]
dates_s = [pd.to_datetime(df_calendar.loc[df_calendar['d'] == str_date,'date'].values[0]) for str_date in date_columns]

tmp = df_sale[date_columns].sum()
ignore_date = df_calendar[df_calendar['d'].isin(tmp[tmp < 10000].index.values)]['date'].values

df_ev_1 = pd.DataFrame({'holiday': 'Event 1', 'ds': df_calendar[~df_calendar['event_name_1'].isna()]['date']})
df_ev_2 = pd.DataFrame({'holiday': 'Event 2', 'ds': df_calendar[~df_calendar['event_name_2'].isna()]['date']})
df_ev_3 = pd.DataFrame({'holiday': 'snap_CA', 'ds': df_calendar[df_calendar['snap_CA'] == 1]['date']})
df_ev_4 = pd.DataFrame({'holiday': 'snap_TX', 'ds': df_calendar[df_calendar['snap_TX'] == 1]['date']})
df_ev_5 = pd.DataFrame({'holiday': 'snap_WI', 'ds': df_calendar[df_calendar['snap_WI'] == 1]['date']})
holidays = pd.concat((df_ev_1, df_ev_2, df_ev_3, df_ev_4, df_ev_5))

def CreateTimeSeries(id):
    item_series =  df_sale[df_sale['id'] == id]
    dates = pd.DataFrame({'ds': dates_s}, index=range(len(dates_s)))
    dates['y'] = item_series[date_columns].values.transpose()
    # Remove chirstmas date
    #dates = dates[~dates['ds'].isin(ignore_date)]
    # Remove zero day
    #dates = dates[dates['y'] > 0]        
    return dates


# In[ ]:


#%%time

#df_price_ext = df_price.merge(df_calendar[['date','wm_yr_wk','d']], on='wm_yr_wk', how='left')
#df_price_ext['id'] = df_price_ext['item_id'] + '_' + df_price_ext['store_id'] + '_validation'
#df_price_ext = df_price_ext.pivot(index='id', columns='d', values='sell_price').reset_index()
#df_price_ext = df_price_ext.fillna(0)


# In[ ]:


#last_month = list(date_columns[-28:])
#df_sale_lastmonth = df_sale[['id'] + last_month].set_index('id')
#df_price_lastmonth = df_price_ext[['id'] + last_month].set_index('id')
#for col in last_month:
#    df_sale_lastmonth[col] = df_sale_lastmonth[col].mul(df_price_lastmonth[col], axis=0)


# In[ ]:


#forecast with prophet without holiday
ids1 = np.array(['FOODS_3_090_WI_3_validation','FOODS_3_785_CA_1_validation'])
#forecast with prophet with holiday
ids2 = np.array(['FOODS_3_362_CA_3_validation'])
#forecast with prophet top down
ids3 = df_sale['id'].values
#forecast with ma
ids4 = np.array(['FOODS_3_090_CA_3_validation'])

#other predict
ids = np.concatenate((ids1, ids2, ids4), axis=0)
#Default prophet top down
ids3 = ids3[~np.isin(ids3,ids)]


print(f'Prophet without: {ids1.shape[0]}, Prophet bu: {ids2.shape[0]}, Prophet td: {ids3.shape[0]}, MA : {ids4.shape[0]}')
print(f'Total ids: {ids1.shape[0] + ids2.shape[0] + ids3.shape[0] + ids4.shape[0]} , Total sample: {df_sale.shape[0]}')


# In[ ]:


# Check duplicate
np.setdiff1d(ids1,np.setdiff1d(ids1,ids2))


# # Prophet forecast

# In[ ]:


get_ipython().run_cell_magic('time', '', "\n# Without holiday\n\ndef run_prophet(id):\n    timeserie = CreateTimeSeries(id)\n    model = Prophet(uncertainty_samples=False)\n    model.fit(timeserie)\n    forecast = model.make_future_dataframe(periods=28, include_history=False)\n    forecast = model.predict(forecast)\n    return np.append(np.array([id]),forecast['yhat'].values.transpose())\n\n\nfrom multiprocessing import Pool, cpu_count\nprint(f'Parallelism on {cpu_count()} CPU')\nwith Pool(cpu_count()) as p:\n    predictions1  = list(p.map(run_prophet, ids1))")


# In[ ]:


get_ipython().run_cell_magic('time', '', "\n# With holiday\n\ndef run_prophet2(id):\n    timeserie = CreateTimeSeries(id)\n    model = Prophet(holidays = holidays, uncertainty_samples = False, n_changepoints = 50, changepoint_range = 0.8, changepoint_prior_scale = 0.7)\n    # changepoint_prior_scale default is 0.5. Increasing it will make the trend more flexible    \n    model.fit(timeserie)\n    forecast = model.make_future_dataframe(periods=28, include_history=False)\n    forecast = model.predict(forecast)\n    return np.append(np.array([id]),forecast['yhat'].values.transpose())\n\n\nfrom multiprocessing import Pool, cpu_count\nprint(f'Parallelism on {cpu_count()} CPU')\nwith Pool(cpu_count()) as p:\n    predictions2  = list(p.map(run_prophet2, ids2))")


# In[ ]:


get_ipython().run_cell_magic('time', '', "\ndf_sale_group_item = df_sale[np.hstack([['dept_id','store_id'],date_columns])].groupby(['dept_id','store_id']).sum()\ndf_sale_group_item = df_sale_group_item.reset_index()\n\ndef CreateTimeSeries3(dept_id, store_id):\n    item_series =  df_sale_group_item[(df_sale_group_item.dept_id == dept_id) & (df_sale_group_item.store_id == store_id)]\n    dates = pd.DataFrame({'ds': dates_s}, index=range(len(dates_s)))\n    dates['y'] = item_series[date_columns].values[0].transpose()     \n    return dates\n\ndef run_prophet3(dept_id, store_id):\n    timeserie = CreateTimeSeries3(dept_id, store_id)\n    # Tunned by one id\n    #model = Prophet(holidays = holidays, uncertainty_samples = False, n_changepoints = 50, changepoint_range = 0.8, changepoint_prior_scale = 0.7)\n    # Tunned by level 9    \n    model = Prophet(holidays = holidays, uncertainty_samples = False, n_changepoints = 50, changepoint_range = 0.8, changepoint_prior_scale = 0.7, seasonality_mode = 'multiplicative')\n    model.fit(timeserie)\n    forecast = model.make_future_dataframe(periods=28, include_history=False)\n    forecast = model.predict(forecast)\n    return np.append(np.array([dept_id,store_id]),forecast['yhat'].values.transpose())\n\n# create list param\nids = []\nfor i in range(0,df_sale_group_item.shape[0]):\n    ids = ids + [(df_sale_group_item[i:i+1]['dept_id'].values[0],df_sale_group_item[i:i+1]['store_id'].values[0])]\n\nprint(f'Parallelism on {cpu_count()} CPU')\nwith Pool(cpu_count()) as p:\n    predictions3  = list(p.starmap(run_prophet3, ids))")


# In[ ]:


df_prophet_forecast_3 = pd.DataFrame()
for k in range(0, len(predictions3)):
    dept_id = predictions3[k][0]
    store_id = predictions3[k][1]

    df_item = df_sale.loc[(df_sale.dept_id == dept_id) & (df_sale.store_id == store_id)][['id']]
    df_item['val'] = df_sale[(df_sale.dept_id == dept_id) & (df_sale.store_id == store_id)].iloc[:, np.r_[0,-28:0]].sum(axis = 1)
    for i in range(1,29):
        df_item[f'F{i}'] = (df_item['val'] * float(predictions3[k][i+1]) / df_item['val'].sum())
    df_prophet_forecast_3 = pd.concat([df_prophet_forecast_3, df_item])

df_prophet_forecast_3 = df_prophet_forecast_3.drop('val',axis=1)


# # Simple MA forecast

# In[ ]:


df_sample = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sample_submission.csv')


# In[ ]:


# Forecast with MA
df_last_sale = df_sale.iloc[:, np.r_[0,-28:0]].melt('id', var_name='d', value_name='sale')
df_last_sale = df_last_sale.merge(df_calendar.loc[:,['d','date','wday']])
last_date = int(df_last_sale.d.max()[2:])
df_last_sale = df_last_sale.groupby(['id','wday'])['sale'].mean()


df_ma_forecast = df_sample.copy()
df_ma_forecast.columns = ['id'] + ['d_' + str(last_date + x) for x in range(1, 29)]
df_ma_forecast = df_ma_forecast.loc[df_ma_forecast.id.str.contains('validation')]

df_ma_forecast = df_ma_forecast.melt('id', var_name='d', value_name='sale')
df_ma_forecast = df_ma_forecast.drop('sale',axis = 1)
df_ma_forecast = df_ma_forecast.merge(df_calendar.loc[:,['d','date','wday']])
df_ma_forecast = df_ma_forecast.join(df_last_sale, on=['id', 'wday'])

df_ma_forecast = df_ma_forecast.pivot(index='id', columns='d', values='sale')
df_ma_forecast = df_ma_forecast.reset_index()


# # Submission

# In[ ]:


if len(ids4) > 0:
    df_ma_forecast = df_ma_forecast[df_ma_forecast.id.isin(ids4)]
    df_ma_forecast.columns = df_sample.columns
else:
    df_ma_forecast = pd.DataFrame()

if len(ids1) > 0:
    df_prophet_forecast_1 = pd.DataFrame(predictions1)
    df_prophet_forecast_1.columns = df_sample.columns
else:
    df_prophet_forecast_1 = pd.DataFrame()

if len(ids2) > 0:
    df_prophet_forecast_2 = pd.DataFrame(predictions2)
    df_prophet_forecast_2.columns = df_sample.columns
else:
    df_prophet_forecast_2 = pd.DataFrame()


df_prophet_forecast_3.columns = df_sample.columns
df_prophet_forecast_3 = df_prophet_forecast_3[df_prophet_forecast_3.id.isin(ids3)]


df_sub_val = pd.concat([df_ma_forecast,df_prophet_forecast_1, df_prophet_forecast_2,df_prophet_forecast_3], sort=False)

df_sub_eval = df_sub_val.copy()
df_sub_eval['id'] = df_sub_eval['id'].str.replace("validation", "evaluation")

df_sub = pd.concat([df_sub_val,df_sub_eval], sort=False)
df_sub = df_sub.sort_values('id')

# Fix negative forecast
num = df_sub._get_numeric_data()
num[num < 0] = 0

df_sub.to_csv('submission.csv', index=False)

print(f'Submission shape: {df_sub.shape}')
print(f'Prophet without holiday: {df_prophet_forecast_1.shape[0]}, Prophet bu: {df_prophet_forecast_2.shape[0]}, Prophet td: {df_prophet_forecast_3.shape[0]}, MA: {df_ma_forecast.shape[0]}')


# # Random check

# In[ ]:


import matplotlib.pyplot as plt
import random
get_ipython().run_line_magic('matplotlib', 'inline')

def plotForecast(item_id):
    his_step = 100
    plt.plot(dates_s[-his_step:] + [dates_s[-1:][0] + pd.DateOffset(days=x) for x in range(28)], np.append(df_sale[df_sale['id'] == item_id][date_columns].values[0][-his_step:],df_sub[df_sub['id'] == item_id].values[0][1:]))
    plt.plot(dates_s[-his_step:], df_sale[df_sale['id'] == item_id][date_columns].values[0][-his_step:])
    if item_id in ids1:
        plt.title(f' Prophet without holiday forecast: {item_id}')    
    elif item_id in ids2:
        plt.title(f' Prophet bu forecast: {item_id}')
    elif item_id in ids3:
        plt.title(f' Prophet td forecast: {item_id}')
    else:
        plt.title(f' MA forecast: {item_id}')
    plt.gcf().autofmt_xdate()


# In[ ]:


if len(ids1)>0:
    item_id = ids1[random.randint(0, len(ids1) - 1)]
    plotForecast(item_id)


# In[ ]:


if len(ids2)>0:
    item_id = ids2[random.randint(0, len(ids2) - 1)]
    plotForecast(item_id)


# In[ ]:


if len(ids3)>0:
    item_id = ids3[random.randint(0, len(ids3) - 1)]
    plotForecast(item_id)


# In[ ]:


if len(ids4):
    item_id = ids4[random.randint(0, len(ids4) - 1)]
    plotForecast(item_id)


# In[ ]:




