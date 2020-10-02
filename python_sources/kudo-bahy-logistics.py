#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import re
from datetime import datetime
import numpy as np
pd.options.mode.chained_assignment = None  # |default='warn'


# This algorithm scores **0.99441** on Shopee Code League - Logistics competition.

# In[ ]:


df_sla = pd.read_excel("../input/open-shopee-code-league-logistic/SLA_matrix.xlsx")


# In[ ]:


## SLA Matrix
df_sla


# In[ ]:


## Delivery data
df_delivery = pd.read_csv("../input/open-shopee-code-league-logistic/delivery_orders_march.csv")


# In[ ]:


df_delivery.shape


# In[ ]:


df_delivery.head()


# In[ ]:


## SLA matrix
matrix = df_sla.iloc[1:5, 2:].values
city_order = ["metro manila", "luzon", "visayas", "mindanao"]


# In[ ]:


matrix


# In[ ]:


## SLA 2nd attempt
print(df_sla.iloc[8:,:1].values[0][0])


# #### Create sample df

# In[ ]:


## Sample orders
df_sample = df_delivery.sample(n=1000)
df_sample.head()


# #### Find origin, destination, SLA

# In[ ]:


def find_city(x):
    for city in city_order:
        if city.lower() in x:
            return city.lower()


# In[ ]:


## Get Origin
def get_origin(df):
    origin = find_city(df['selleraddress'].lower())
    return origin


# In[ ]:


## Get Destination
def get_destination(df):
    destination = find_city(df['buyeraddress'].lower())
    return destination


# In[ ]:


df_sample['origin'] = df_sample.apply(get_origin, axis=1)
df_sample['destination'] = df_sample.apply(get_destination, axis=1)


# In[ ]:


df_sample.head()


# In[ ]:


def get_sla(df):
    sla = matrix[city_order.index(df['origin']), city_order.index(df['destination'])]
    days = int(re.search(r'\d', sla).group(0))
    return days


# In[ ]:


df_sample['sla'] = df_sample.apply(get_sla, axis=1)


# In[ ]:


df_sample.head()


# #### Check for first SLA fullfilment

# In[ ]:


## Convert all date column to date
pick = pd.to_datetime(df_sample['pick'], unit='s').dt.date
first_deliver = pd.to_datetime(df_sample['1st_deliver_attempt'], unit='s').dt.date
second_deliver = pd.to_datetime(df_sample['2nd_deliver_attempt'], unit='s').dt.date


# In[ ]:


df_sample['pick'] = pick
df_sample['1st_deliver_attempt'] = first_deliver
df_sample['2nd_deliver_attempt'] = second_deliver


# In[ ]:


df_sample.head()


# In[ ]:


## Initiate public holidays
public_holidays = ["2020-03-08", "2020-03-25", "2020-03-30", "2020-03-31"]


# In[ ]:


def get_busday_first(df):
    create_date = str(df['pick'])
    resolve_date = str(df['1st_deliver_attempt'])

    create_datetime = datetime.strptime(create_date, '%Y-%m-%d')
    resolve_datetime = datetime.strptime(resolve_date, '%Y-%m-%d')

    busday = np.busday_count(create_date, resolve_date, holidays=public_holidays, weekmask=[1,1,1,1,1,1,0])

    return busday    


# In[ ]:


df_sample['1st_deliver_days'] = df_sample.apply(get_busday_first, axis=1)


# In[ ]:


df_sample.head()


# In[ ]:


## Saving checkpoint
# import pickle
# pickle.dump(df_sample, open("1st_fullfilment.pickle", "wb"))


# #### Check for 2nd SLA fullfilment

# In[ ]:


def get_busday_second(df):
    create_date = str(df['1st_deliver_attempt'])
    resolve_date = str(df['2nd_deliver_attempt'])
    
    if resolve_date == 'NaT':
        resolve_date = create_date

    create_datetime = datetime.strptime(create_date, '%Y-%m-%d')
    resolve_datetime = datetime.strptime(resolve_date, '%Y-%m-%d')

    busday = np.busday_count(create_date, resolve_date, holidays=public_holidays, weekmask=[1,1,1,1,1,1,0])

    return busday   


# In[ ]:


df_sample['2nd_deliver_days'] = df_sample.apply(get_busday_second, axis=1)


# In[ ]:


df_sample.head()


# In[ ]:


## Saving checkpoint
# import pickle
# pickle.dump(df_sample, open("2nd_fullfilment.pickle", "wb"))


# #### Decide late/not late

# In[ ]:


not_late_df = df_sample[(df_sample['1st_deliver_days'] <= df_sample['sla']) & (df_sample['2nd_deliver_days'] <= 3)][['orderid']]
not_late_df['is_late'] = 0
not_late_df.head()


# In[ ]:


late_df = df_sample[~df_sample['orderid'].isin(not_late_df['orderid'])][['orderid']]
late_df['is_late'] = 1
late_df.head()


# In[ ]:


not_late_df.orderid.nunique() + late_df.orderid.nunique()


# In[ ]:


res_df = pd.concat([late_df, not_late_df], axis=0)
res_df.head()


# ### Run all below cells as "Code" to export the solution for sample/all data

# #### Export solution
# res_df.to_csv("solution_logistics_bahy_sample.csv", index=False)

# test = pd.read_csv("solution_logistics_bahy_sample.csv")
# test.head()

# test.shape

# #### Test to all data
# Just replace all the cells below to a Code cell to run it on all data

# df_delivery.head()

# df_delivery['origin'] = df_delivery.apply(get_origin, axis=1)
# df_delivery['destination'] = df_delivery.apply(get_destination, axis=1)

# df_delivery['sla'] = df_delivery.apply(get_sla, axis=1)

# pick = pd.to_datetime(df_delivery['pick'], unit='s').dt.date
# first_deliver = pd.to_datetime(df_delivery['1st_deliver_attempt'], unit='s').dt.date
# second_deliver = pd.to_datetime(df_delivery['2nd_deliver_attempt'], unit='s').dt.date

# df_delivery['pick'] = pick
# df_delivery['1st_deliver_attempt'] = first_deliver
# df_delivery['2nd_deliver_attempt'] = second_deliver

# public_holidays = ["2020-03-08", "2020-03-25", "2020-03-30", "2020-03-31"]

# df_delivery['1st_deliver_days'] = df_delivery.apply(get_busday_first, axis=1)

# df_delivery['2nd_deliver_days'] = df_delivery.apply(get_busday_second, axis=1)

# not_late_df = df_sample[(df_sample['1st_deliver_days'] <= df_sample['sla']) & (df_sample['2nd_deliver_days'] <= 3)][['orderid']]
# not_late_df['is_late'] = 0

# late_df = df_sample[~df_sample['orderid'].isin(not_late_df['orderid'])][['orderid']]
# late_df['is_late'] = 1

# res_df = pd.concat([late_df, not_late_df], axis=0)

# #### Export solution
# res_df.to_csv("solution_logistics_bahy.csv", index=False)
