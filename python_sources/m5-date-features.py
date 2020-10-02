#!/usr/bin/env python
# coding: utf-8

# ## M5 Encoder Decoder model - Generating Date features
# This is with reference to the Notebook M5 Encoder Decoder model with attention.
# 
# Link : https://www.kaggle.com/josephjosenumpeli/m5-forecasting-encoder-decoder-with-attention 
# The features generated are fairly simple
# 
# This Notebook contains the basic approach and is just the start

# In[ ]:


import pandas as pd
import os
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt
import datetime as dt
calender = pd.read_csv('../input/m5-forecasting-accuracy/calendar.csv')


# In[ ]:


calender.head()


# In[ ]:


# Removed event_name_1 and event_2 as they were to sparse
date_features = calender.loc[:,['d','date','wm_yr_wk','wday','event_type_1', 'snap_CA', 'snap_TX', 'snap_WI']]


# In[ ]:


date_features


# In[ ]:


calender['date'] = pd.to_datetime(calender['date'])
calender['month'] = calender['date'].dt.month
calender['wk_year'] = calender['wm_yr_wk'].apply(str).str[-2:].apply(int)
calender['yr-month'] = calender['date'].dt.to_period('M')
calender['day_of_month']=calender['date'].dt.day


# In[ ]:


calender.columns


# In[ ]:


dates = pd.concat([calender.loc[:,['date','wm_yr_wk','d','snap_CA','snap_TX','snap_WI','day_of_month','wk_year']],
           pd.get_dummies(calender['wday']),
           pd.get_dummies(calender['month']),
           pd.get_dummies(calender['event_type_1'])], axis =1)


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
dates['day_of_month'] = MinMaxScaler().fit_transform(dates['day_of_month'].to_numpy().reshape(-1,1))


# In[ ]:


dates['wk_year'] = MinMaxScaler().fit_transform(dates['wk_year'].to_numpy().reshape(-1,1))


# In[ ]:


dates.columns = [ 'date','wm_yr_wk','d','CA','TX','WI','day_of_month', 'wk_year', 
                 '1', '2', '3', '4','5','6','7',
                 'Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec',
                 'Cultural','National','Religious','Sporting']


# In[ ]:


column_names = [ 'date','wm_yr_wk','d','day_of_month', 'wk_year', 
                 '1', '2', '3', '4','5','6','7',
                 'Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec',
                 'Cultural','National','Religious','Sporting',
                 'CA','TX','WI']


# In[ ]:


dates= dates.reindex(columns=column_names)


# In[ ]:


dates


# In[ ]:


dates.to_csv('data/date_features.csv')

