#!/usr/bin/env python
# coding: utf-8

# Very fast solution (score: 0.75). 2 minutes to complete without turning. Score: 0.63, 4 minutes with some parameter tuning
# This kernel use Top - Down approach:
# - Predict 70 time series group by Department, Category, Store and State (aggregation level 9)
# - Predict each of item on time series by weight (total sale in last 28 days)
# 
# To do:
# - Try Auto ARIMA with exogenous variables
# 
# My other approach:
# - Prophet bu  (score 0.71): https://www.kaggle.com/binhlc/forecasting-multiple-time-series-using-prophet
# - Simple Exp  (score >1.0): https://www.kaggle.com/binhlc/m5-forecasting-accuracy-simple-exp-smoothing
# - Tensorflow  (score >2.0): https://www.kaggle.com/binhlc/high-dimensional-time-series-forecasting-with-tf2
# - Prophet parametertunning: https://www.kaggle.com/binhlc/prophet-hyperparameter-tuning
# 
# Reference:
# - https://otexts.com/fpp2/forecasting-regression.html
# 
# Change log:
# - Verison 3: Change seasonality_mode = 'multiplicative' 
# - Verison 2: Rollback Prophet with new parameter
# - Verison 1: Auto Arima without event (score: 0.78, 1 hours)
# - Version 0: Prophet with event (score: 0.75, 2.5 minutes)
# 

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


# In[ ]:


df_sale = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')
df_sale_eval = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_evaluation.csv')
df_calendar = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')
df_price = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv')


# # Validation

# In[ ]:


date_columns = df_sale.columns[df_sale.columns.str.contains("d_")]
dates_s = [pd.to_datetime(df_calendar.loc[df_calendar['d'] == str_date,'date'].values[0]) for str_date in date_columns]

df_ev_1 = pd.DataFrame({'holiday': 'Event 1', 'ds': df_calendar[~df_calendar['event_name_1'].isna()]['date']})
df_ev_2 = pd.DataFrame({'holiday': 'Event 2', 'ds': df_calendar[~df_calendar['event_name_2'].isna()]['date']})
df_ev_3 = pd.DataFrame({'holiday': 'snap_CA', 'ds': df_calendar[df_calendar['snap_CA'] == 1]['date']})
df_ev_4 = pd.DataFrame({'holiday': 'snap_TX', 'ds': df_calendar[df_calendar['snap_TX'] == 1]['date']})
df_ev_5 = pd.DataFrame({'holiday': 'snap_WI', 'ds': df_calendar[df_calendar['snap_WI'] == 1]['date']})
holidays = pd.concat((df_ev_1, df_ev_2, df_ev_3, df_ev_4, df_ev_5))

df_sale_group_item = df_sale[np.hstack([['dept_id','store_id'],date_columns])].groupby(['dept_id','store_id']).sum()
df_sale_group_item = df_sale_group_item.reset_index()


# In[ ]:


get_ipython().run_cell_magic('time', '', "\nfrom fbprophet import Prophet\nfrom multiprocessing import Pool, cpu_count\n\ndef CreateTimeSeries(dept_id, store_id):\n    item_series =  df_sale_group_item[(df_sale_group_item.dept_id == dept_id) & (df_sale_group_item.store_id == store_id)]\n    dates = pd.DataFrame({'ds': dates_s}, index=range(len(dates_s)))\n    dates['y'] = item_series[date_columns].values[0].transpose()     \n    return dates\n\ndef run_prophet(dept_id, store_id):\n    timeserie = CreateTimeSeries(dept_id, store_id)\n    # Tunned by one id\n    #model = Prophet(holidays = holidays, uncertainty_samples = False, n_changepoints = 50, changepoint_range = 0.8, changepoint_prior_scale = 0.7)\n    # Tunned by level 9    \n    model = Prophet(holidays = holidays, uncertainty_samples = False, n_changepoints = 50, changepoint_range = 0.8, changepoint_prior_scale = 0.7, seasonality_mode = 'multiplicative')\n    model.fit(timeserie)\n    forecast = model.make_future_dataframe(periods=28, include_history=False)\n    forecast = model.predict(forecast)\n    return np.append(np.array([dept_id,store_id]),forecast['yhat'].values.transpose())\n\n# create list param\nids = []\nfor i in range(0,df_sale_group_item.shape[0]):\n    ids = ids + [(df_sale_group_item[i:i+1]['dept_id'].values[0],df_sale_group_item[i:i+1]['store_id'].values[0])]\n\nprint(f'Parallelism on {cpu_count()} CPU')\nwith Pool(cpu_count()) as p:\n    predictions  = list(p.starmap(run_prophet, ids))")


# In[ ]:


#!pip install pmdarima


# In[ ]:


'''
%%time

import pmdarima as pm
def run_arima(dept_id, store_id):
    timeserie = CreateTimeSeries(dept_id, store_id)
    model = pm.auto_arima(timeserie['y'], suppress_warnings=True, seasonal=True, error_action="ignore")
    y_hat = model.predict(n_periods=28)
    return np.append(np.array([dept_id,store_id]),y_hat)

print(f'Parallelism on {cpu_count()} CPU')
with Pool(cpu_count()) as p:
    predictions  = list(p.starmap(run_arima, ids))
'''


# In[ ]:


#Submission

df_sub_val = pd.DataFrame()
for k in range(0, len(predictions)):
    dept_id = predictions[k][0]
    store_id = predictions[k][1]

    df_item = df_sale.loc[(df_sale.dept_id == dept_id) & (df_sale.store_id == store_id)][['id']]
    df_item['val'] = df_sale[(df_sale.dept_id == dept_id) & (df_sale.store_id == store_id)].iloc[:, np.r_[0,-28:0]].sum(axis = 1)
    for i in range(1,29):
        df_item[f'F{i}'] = (df_item['val'] * float(predictions[k][i+1]) / df_item['val'].sum())
    df_sub_val = pd.concat([df_sub_val, df_item])

df_sub_val = df_sub_val.drop('val',axis=1)


# # Evaluation

# In[ ]:


df_sale = df_sale_eval
date_columns = df_sale.columns[df_sale.columns.str.contains("d_")]
dates_s = [pd.to_datetime(df_calendar.loc[df_calendar['d'] == str_date,'date'].values[0]) for str_date in date_columns]

df_ev_1 = pd.DataFrame({'holiday': 'Event 1', 'ds': df_calendar[~df_calendar['event_name_1'].isna()]['date']})
df_ev_2 = pd.DataFrame({'holiday': 'Event 2', 'ds': df_calendar[~df_calendar['event_name_2'].isna()]['date']})
df_ev_3 = pd.DataFrame({'holiday': 'snap_CA', 'ds': df_calendar[df_calendar['snap_CA'] == 1]['date']})
df_ev_4 = pd.DataFrame({'holiday': 'snap_TX', 'ds': df_calendar[df_calendar['snap_TX'] == 1]['date']})
df_ev_5 = pd.DataFrame({'holiday': 'snap_WI', 'ds': df_calendar[df_calendar['snap_WI'] == 1]['date']})
holidays = pd.concat((df_ev_1, df_ev_2, df_ev_3, df_ev_4, df_ev_5))

df_sale_group_item = df_sale[np.hstack([['dept_id','store_id'],date_columns])].groupby(['dept_id','store_id']).sum()
df_sale_group_item = df_sale_group_item.reset_index()


# In[ ]:


ids = []
for i in range(0,df_sale_group_item.shape[0]):
    ids = ids + [(df_sale_group_item[i:i+1]['dept_id'].values[0],df_sale_group_item[i:i+1]['store_id'].values[0])]

print(f'Parallelism on {cpu_count()} CPU')
with Pool(cpu_count()) as p:
    predictions  = list(p.starmap(run_prophet, ids))


# In[ ]:


df_sub_eval = pd.DataFrame()
for k in range(0, len(predictions)):
    dept_id = predictions[k][0]
    store_id = predictions[k][1]

    df_item = df_sale.loc[(df_sale.dept_id == dept_id) & (df_sale.store_id == store_id)][['id']]
    df_item['val'] = df_sale[(df_sale.dept_id == dept_id) & (df_sale.store_id == store_id)].iloc[:, np.r_[0,-28:0]].sum(axis = 1)
    for i in range(1,29):
        df_item[f'F{i}'] = (df_item['val'] * float(predictions[k][i+1]) / df_item['val'].sum())
    df_sub_eval = pd.concat([df_sub_eval, df_item])

df_sub_eval = df_sub_eval.drop('val',axis=1)


# In[ ]:


df_sub = pd.concat([df_sub_val,df_sub_eval], sort=False)
df_sub = df_sub.sort_values('id').reset_index(drop = True)

df_sub.to_csv('submission.csv', index=False)
df_sub


# In[ ]:


#Random check

import matplotlib.pyplot as plt
import random
get_ipython().run_line_magic('matplotlib', 'inline')
pd.plotting.register_matplotlib_converters()
def plotForecast(item_id):
    his_step = 100
    plt.plot(dates_s[-his_step:] + [dates_s[-1:][0] + pd.DateOffset(days=x) for x in range(28)], np.append(df_sale[df_sale['id'] == item_id][date_columns].values[0][-his_step:],df_sub[df_sub['id'] == item_id].values[0][1:]))
    plt.plot(dates_s[-his_step:], df_sale[df_sale['id'] == item_id][date_columns].values[0][-his_step:])
    plt.title(f' Prophet top down forecast: {item_id}')
    plt.gcf().autofmt_xdate()

item_id = df_sale['id'][random.randint(0, len(df_sale['id']) - 1)]
#item_id = 'FOODS_3_090_CA_3_validation'
plotForecast(item_id)


# In[ ]:


ids = df_sale.loc[df_sale.iloc[:,-14:].mean(axis=1).sort_values(ascending = False)[0:10].index,'id'].values
item_id = ids[random.randint(0, len(ids) - 1)]
plotForecast(item_id)

