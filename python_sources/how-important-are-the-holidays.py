#!/usr/bin/env python
# coding: utf-8

# # This notebook investigates holiday sales
# Quick & dirty charts and tables to demonstrate differnces in average consumer purchasing behavior for the period extending from two weeks before to one week after each holiday.

# In[ ]:


# import libraries
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


# In[ ]:


datapath = '../input/m5-forecasting-accuracy'

# import data files
calendar = pd.read_csv(f'{datapath}/calendar.csv', parse_dates=['date'])
sales_train_validation = pd.read_csv(f'{datapath}/sales_train_validation.csv')


# In[ ]:


# tags = 'id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'
# data = 'd_1' -> 'd_1913'
tags, data =sales_train_validation.iloc[:, :6], sales_train_validation.iloc[:, 6:]


# In[ ]:


# plot daily sales average
data_means = pd.DataFrame(data.mean(), columns=['mean'])
data_means.plot(subplots=True, figsize=(15,4))


# In[ ]:


# Strip holidays from calendar and create unique list
holidays = calendar[['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']]
uholidays = pd.unique(holidays[['event_name_1', 'event_name_2']].values.ravel())[1:]
holidays_shifted = holidays.shift(-14).loc[:data_means.shape[0]-1, :]
uholidays


# Plot the three week window around each holiday. The holiday itself occurs on day 0. The reference line is the average of all days not included in the window.

# In[ ]:


# align dates to each holiday
for holiday in uholidays:
    dayno = 0
    daynos = []
    for index, row in holidays_shifted.iterrows():
        if dayno > 0:
            dayno += 1
        if dayno>21:
            dayno = 0
        if (row.event_name_1==holiday) | (row.event_name_2==holiday): # upcoming holiday
            dayno = 1
        daynos.append(dayno)

    data_means['dayno'] = daynos
    data_means['dayno'] -= 15
    df = data_means.groupby('dayno', sort=True).mean() #.plot(figsize=(15,2))
    df.columns = [holiday]
    df['ref'] = df.iloc[0, 0] # all zero dayno's go here
    ax = df[1:].plot(figsize=(15,4))
    ax.locator_params(integer=True)
    ax.axvline(x=0)
    plt.show()


# Some of the graphs may be a suprise. Although causation cannot be infered by these charts, consider that the categories are food, houshold, and hobbies. It might be interesting to split these plots by category. For example, more food items may be sold before holidays associated with large family gatherings, but they are not as popular on Black Friday. Also, Monday holidays are always comming off a weekend, when sales normally peak. 

# In[ ]:




