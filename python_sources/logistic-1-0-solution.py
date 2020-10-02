#!/usr/bin/env python
# coding: utf-8

# We decide to release our solution to score 1.0 after the competition for your reference!

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


PATH = "/kaggle/input/logistics-shopee-code-league/"


# In[ ]:


df = pd.read_csv(PATH+'delivery_orders_march.csv')
df


# In[ ]:


def SLA(origin, dest):
    if origin == 'manila':
        if dest == 'manila': return 3
        elif dest == 'luzon': return 5
        else: return 7
    elif origin == 'luzon':
        if dest == 'manila': return 5
        elif dest == 'luzon': return 5
        else: return 7


# In[ ]:


buyer = []
seller = []
pick = []
st = []
nd = []
day = []

buyer_addr = df['buyeraddress'].values
seller_addr = df['selleraddress'].values

p = df['pick'].values
p1 = df['1st_deliver_attempt'].values
p2 = df['2nd_deliver_attempt'].values

for i in range(len(df)):
    buyer.append(buyer_addr[i].lower().split(' ')[-1])
    seller.append(seller_addr[i].lower().split(' ')[-1])
    day.append(SLA(seller_addr[i].lower().split(' ')[-1],buyer_addr[i].lower().split(' ')[-1]))
    pick.append(datetime.datetime.fromtimestamp(p[i]).date())
    st.append(datetime.datetime.fromtimestamp(p1[i]).date())
    if np.isnan(p2[i]):
        nd.append(0)
    else:
        nd.append(datetime.datetime.fromtimestamp(p2[i]).date())


# In[ ]:


df_new = pd.DataFrame()
df_new['orderid'] = df['orderid'].values
df_new['pick'] = pick
df_new['SLA'] = day
df_new['1st_deliver_attempt'] = st
df_new['2nd_deliver_attempt'] = nd
df_new['buyeraddress'] = buyer
df_new['selleraddress'] = seller


# In[ ]:


df_new.head(20)


# In[ ]:


buyer = df_new['buyeraddress'].values
seller = df_new['selleraddress'].values
pick = df_new['pick'].values
t1 = df_new['1st_deliver_attempt'].values
t2 = df_new['2nd_deliver_attempt'].values
SLA = df_new['SLA'].values


# In[ ]:


holiday = [datetime.datetime(2020, 3, 1).date(),datetime.datetime(2020, 3, 8).date(),datetime.datetime(2020, 3, 15).date(),datetime.datetime(2020, 3, 22).date(),datetime.datetime(2020, 3, 25).date(),datetime.datetime(2020, 3, 29).date(),datetime.datetime(2020, 3, 30).date(),datetime.datetime(2020, 3, 31).date(),datetime.datetime(2020, 4, 5).date()]
holiday


# In[ ]:


ans = []
for i in range(len(df_new)):
    is_late = False
    time = (t1[i] - pick[i]).days
    for h in holiday:
        if pick[i]<=h<=t1[i]:
            time-=1
    if time>SLA[i]:
        is_late = True
    
    if t2[i]!=0 and is_late!=True:
        time = (t2[i] - t1[i]).days
        for h in holiday:
            if t1[i]<=h<=t2[i]:
                time-=1
        if time>3:
            is_late = True
    ans.append(is_late)


# In[ ]:


df_new['is_late'] = ans
df_new['is_late'] = df_new['is_late'].map({False:0,True:1})
df_new


# In[ ]:


df_new[['orderid','is_late']].to_csv('submission.csv',index = False)

