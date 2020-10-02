#!/usr/bin/env python
# coding: utf-8

# * Fork + improved prophet params
# 
# ORIG description follows: ## Goal of the notebook 
# - Use Facebook's Prophet to predict each time series 
# 
# 
# ### The problems 
# FBprophet is a great tool for predicting univariate times series that can take into consideration external factors such as calendar effects, special events, etc. However the large number of time series to predict in the competition leads to increased running times due to a) the sequential nature of the for loop and b) the fact that FBprophet does not really perform parallelization across all cpu threads by default. 
# 
# ### The solutions 
# To reduce computational times there are some tricks we might use: 
# 1. Avoid producing confidence intervals for our prediction. Since we do not have to provide a confidence in our predictions to our bosses, simply dropping them speeds up training a lot (kudos to @tita1708 for that). 
# 2. Avoid producing in-sample prediction. By default, FBprophet produces a prediction that covers your entire training sample plus whatever horizon you tell it to predict (e.g. 28 days). Since we dont neccessarily need the in-sample values (at least not to submit them), we can drop them to save some time. 
# 3. Avoid using the full length of the time series. In this notebook I start from observation 800, dropping the ones before that point. Maybe not the optimal, it is strictly selected just to curtail running times. Could possibly be the correct choice if the was a data shift between old and recent data but that needs to be examined
# 4. Parallelize the process. I have use the standard 'multiprocessing' python library here to force all available CPU threads to work on each prediction. CAUTION: this only parallelizes each single iteration of the for loop across CPUs, it does not parallelize the entire for loop itself. 

# In[ ]:


import pandas as pd
import numpy as np
from fbprophet import Prophet
from tqdm import tqdm, tnrange, trange
from multiprocessing import Pool, cpu_count


# In[ ]:


calendar_df = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')
sales_train =  pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')
sell_prices = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv')
submission = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sample_submission.csv')


# In[ ]:


Prophet(uncertainty_samples=False,weekly_seasonality = True, yearly_seasonality = True,)


# In[ ]:


def run_prophet(timeserie):
    model = Prophet(uncertainty_samples=False,weekly_seasonality = True, yearly_seasonality = True,) # changed to add seasonality - will it wokr with numeric row extract? 
    # optional: add usa holidays
    model.add_country_holidays(country_name='US')
    
    model.fit(timeserie)
    future = model.make_future_dataframe(periods=28, include_history=False)
    forecast = model.predict(future)
    return forecast


# In[ ]:


# start_from_ob = 800 # orig
start_from_ob = 700
for i in trange(sales_train.shape[0]):
    temp_series = sales_train.iloc[i,start_from_ob:]
    temp_series.index = calendar_df['date'][start_from_ob:start_from_ob+len(temp_series)]
    temp_series =  pd.DataFrame(temp_series)
    temp_series = temp_series.reset_index()
    temp_series.columns = ['ds', 'y']

    with Pool(cpu_count()) as p:
        forecast1 = p.map(run_prophet, [temp_series])

    submission.iloc[i,1:] = forecast1[0]['yhat'].values

submission.iloc[:,1:][submission.iloc[:,1:]<0]=0


# In[ ]:


submission.to_csv('submission.csv', index=False)

