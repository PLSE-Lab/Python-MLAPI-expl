#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import time
from datetime import datetime
import pytz


# In[ ]:


# Read data
data = pd.read_csv("../input/logistics-shopee-code-league/delivery_orders_march.csv")
data.head()


# In[ ]:


# Make an SLA dictionary from SLA matrix
sla_dict = {
    ('metro manila', 'metro manila') : 3,
    ('metro manila', 'luzon') : 5,
    ('metro manila', 'visayas') : 7,
    ('metro manila', 'mindanao') : 7,
    ('luzon', 'metro manila') : 5,
    ('luzon', 'luzon') : 5,
    ('luzon', 'visayas') : 7,
    ('luzon', 'mindanao') : 7,
    ('visayas', 'metro manila') : 7,
    ('visayas', 'luzon') : 7,
    ('visayas', 'visayas') : 7,
    ('visayas', 'mindanao') : 7,
    ('mindanao', 'metro manila') : 7,
    ('mindanao', 'luzon') : 7,
    ('mindanao', 'visayas') : 7,
    ('mindanao', 'mindanao') : 7,
}


# In[ ]:


# Add a new 'route' column from buyeraddress and selleraddress
def get_city(addr):
    city = addr.split(',')[-1].lower()
    for city_name in ['metro manila', 'luzon', 'visayas', 'mindanao']:
        if city_name in city:
            return city_name
    raise ValueError
        
data['buyeraddress'] = data['buyeraddress'].apply(get_city)
data['selleraddress'] = data['selleraddress'].apply(get_city)
data['route'] = tuple(zip(data['buyeraddress'], data['selleraddress']))
data.head()


# In[ ]:


# Add a 'required' delivery time column
def get_required(route):
    return sla_dict[route]

data['required'] = data['route'].apply(get_required)
data.head()


# In[ ]:


# Convert epoch times into date
def convert_time(t):
    if (np.isnan(t)):
        return np.nan
    t = datetime.fromtimestamp(t, tz=pytz.timezone('Asia/Singapore'))
    month = int(t.strftime('%m'))
    day = int(t.strftime('%d'))
    if month == 3:
        return day
    elif month == 4:
        return day + 31
    else:
        raise ValueError
        
data['pick'] = data['pick'].apply(convert_time)
data['1st_deliver_attempt'] = data['1st_deliver_attempt'].apply(convert_time)
data['2nd_deliver_attempt'] = data['2nd_deliver_attempt'].apply(convert_time)
data.head()


# In[ ]:


# Check if a delivery is late or not
off_days = [1, 8, 15, 22, 25, 29, 30, 31, 36, 43, 50, 57]
def check_delivery(pick, first, second, required):
    first_deliver_time = first - pick
    for i in range(pick, first+1):
        if i in off_days:
            first_deliver_time -= 1
    if (first_deliver_time > required):
        return 1
    if (np.isnan(second)):
        return 0
    second = int(second)
    second_deliver_time = second - first
    for i in range(first, second+1):
        if i in off_days:
            second_deliver_time -= 1
    if (second_deliver_time > 3):
        return 1
    return 0

data['is_late'] = data.apply(lambda x: check_delivery(x['pick'], x['1st_deliver_attempt'], x['2nd_deliver_attempt'], x['required']), axis=1)
data.head()


# In[ ]:


# Export result
data[['orderid', 'is_late']].to_csv('../input/output/result.csv', index=False, header=True)
# Accuracy: 0.99996


# In[ ]:




