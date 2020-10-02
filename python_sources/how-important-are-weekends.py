#!/usr/bin/env python
# coding: utf-8

# # This notebook investigates an overlooked seasonality
# Quick & dirty charts to demonstrate the daily trend each month

# Import libraries

# In[ ]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


# Define workspace

# In[ ]:


datapath = '../input/m5-forecasting-accuracy'

# import data files
calendar = pd.read_csv(f'{datapath}/calendar.csv', parse_dates=['date'])
sales_train_validation = pd.read_csv(f'{datapath}/sales_train_validation.csv')


# > Split labels from numeric data

# In[ ]:


# tags = 'id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'
# data = 'd_1' -> 'd_1913'
tags, data = sales_train_validation.iloc[:, :6], sales_train_validation.iloc[:, 6:]
data.columns = calendar.loc[:data.columns.size-1,'date']


# Let's plot daily sales.

# In[ ]:


df = data.sum(axis=0)
df.plot(figsize=(15,4))


# Let's plot the day-of-week seasonality.

# In[ ]:


data.columns = calendar.loc[:data.columns.size-1,'wday']
data.transpose().groupby('wday').mean().transpose().sum().plot()


# Generally, more items are sold on the weekend.  
#   
# Let's plot the day-of-month seasonality.

# In[ ]:


data.columns = calendar.date.dt.day.loc[:data.columns.size-1]
data.transpose().groupby('date', sort=True).mean().transpose().sum().plot(figsize=(15,4))


# Generally, more items are sold in the first half of the month than in the second. More items are sold on the 3rd than any other day. The 25th takes a big hit. Although rumors abound, no causation is implied by this chart.  
#   
# What if we align the dates to the first Saturday of each month?

# In[ ]:


calendar['weekno'] = ((calendar.date.dt.day-1)/7).astype(int) # zero based: 0 = week 1
dayno = 0
daynos = []
for index, row in calendar.iterrows():
    if dayno == 0:
        dayno = row.wday+row.weekno*7
    if (row.wday==1) & (row.weekno==0): # first Saturday
        dayno = 1
    daynos.append(dayno)
    dayno = dayno+1

calendar['dayno'] = daynos


# Let's plot the aligned day-of-month seasonality.

# In[ ]:


data.columns = calendar.loc[:data.columns.size-1,'dayno']
data.transpose().groupby('dayno', sort=True).mean().transpose().sum().plot(figsize=(15,4))


# The third and fourth weekends are generally lower than the first two. But, why does the fifth weekend look wanky?

# In[ ]:


df = calendar.groupby('dayno', sort=True)['date'].count().plot(figsize=(15,4))


# The fifth weekend is rare, only 23 occurances in the 65 months plotted. This may influence the uncertainty in the fifth week.

# In[ ]:




