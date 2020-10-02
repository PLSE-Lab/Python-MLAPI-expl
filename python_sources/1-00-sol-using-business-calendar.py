#!/usr/bin/env python
# coding: utf-8

# # (Open) Shopee Code League - Logistics
# 
# We use business_calender to calculate working days.
# This scores 1.00000 with late submission.

# ## 0. Set environment

# In[ ]:


# install business_calendar

get_ipython().system('pip install business_calendar')


# In[ ]:


# import packages

import numpy as np
import pandas as pd
import os
import csv
from business_calendar import Calendar, MO, TU, WE, TH, FR, SA
from datetime import datetime


# ## 1. Data Preprocess

# In[ ]:


# load data and have a look

data = pd.read_csv('/kaggle/input/open-shopee-code-league-logistic/delivery_orders_march.csv')
data.head()


# In[ ]:


# find out the city of origin and destination each order, and bind them together

data['origin'] = [address.split()[-1].lower() for address in data['selleraddress']]
data['destination'] = [address.split()[-1].lower() for address in data['buyeraddress']]

data['from_to'] = [i+" "+j for i, j in zip(data['origin'], data['destination'])]
set(data['from_to'])  # check the possible set of origin and destination


# In[ ]:


# use the information in SLA_matrix and create the dictionary of corresponding SLA limit days

sla = {'luzon luzon': 5,
       'manila luzon': 5,
       'manila manila': 3,
       'manila mindanao': 7,
       'manila visayas': 7}

data['SLA_time'] = [sla[i] for i in data['from_to']]
data = data[['orderid', 'pick', '1st_deliver_attempt', '2nd_deliver_attempt', 'SLA_time']]

data.head()


# ## 2. Data Analysis

# In[ ]:


# create the calendar
# Add holiday 2020/1/1 and 2020/12/31 to avoid warning

cal = Calendar(workdays=[MO, TU, WE, TH, FR, SA], holidays=[datetime(2020, 1, 1), datetime(2020, 3, 25), datetime(2020, 3, 30), datetime(2020, 3, 31), datetime(2020, 12, 31)])

# check the time_zone
# timestamp = 1583137548
# datetime.fromtimestamp(timestamp)

time_zone = 28800  # add 8 hours


# In[ ]:


# make the solution!
# columns are 'orderid', 'pick', '1st_deliver_attempt', '2nd_deliver_attempt', 'SLA_time'

data = data.to_numpy()
f = open(os.path.join('/kaggle/working', 'output.csv'), 'w')
f.write('orderid,is_late\n')

for row in data:
    start_time = datetime.fromtimestamp(float(row[1]) + time_zone)  # add the time_zone
    first_time = datetime.fromtimestamp(float(row[2]) + time_zone)
    is_delay = first_time.date() > cal.addbusdays(start_time, int(row[4])).date()  # check if late

    if not is_delay and not pd.isna(row[3]):  # the case of 2nd delivery attempt
        second_time = datetime.fromtimestamp(float(row[3]) + time_zone)
        is_delay = second_time.date() > cal.addbusdays(first_time, 3).date()
    f.write(f'{int(float(row[0]))},{int(is_delay)}\n')
    
f.close()


# Welcome to leave a message ~T&T~
