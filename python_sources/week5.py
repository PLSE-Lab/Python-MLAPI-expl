#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('reset', '-sf')


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import functools, collections

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


from tqdm.notebook import tqdm
tqdm.pandas()


# In[ ]:


df = pd.read_csv("/kaggle/input/logistics-shopee-code-league/delivery_orders_march.csv")


# In[ ]:


# Pick Up Time 1583137548 (Converted to 2020-03-02 4:25:48 PM Local Time)
(1583137548%(24*60*60))/(60*60), ((1583137548+(8*60*60))%(24*60*60))/(60*60)


# In[ ]:


df["pick"] = df["pick"]+(8*60*60)
df["1st_deliver_attempt"] = df["1st_deliver_attempt"]+(8*60*60)
df["2nd_deliver_attempt"] = df["2nd_deliver_attempt"]+(8*60*60)


# In[ ]:


df["t1"] = df["pick"]//(24*60*60)
df["t2"] = df["1st_deliver_attempt"].astype(int)//(24*60*60)
df["t3"] = df["2nd_deliver_attempt"]//(24*60*60)


# In[ ]:


# assuming same day delivery for t3 if NaN (i.e. succesful first delivery)
df.loc[df["t3"] != df["t3"], "t3"] = df.loc[df["t3"] != df["t3"], "t2"]
df["t3"] = df["t3"].astype(int)


# # Get SLA

# In[ ]:


@functools.lru_cache(maxsize=None)
def get_sla(x,y):
    if x == y == 0:
        return 3
    if x <= 1 and y <= 1:
        return 5
    return 7


# In[ ]:


df["buyeraddress"] = df["buyeraddress"].str.lower()
df["selleraddress"] = df["selleraddress"].str.lower()

df["buyer_region"] = -1
df["seller_region"] = -1


# In[ ]:


addresses = ["metro manila", "luzon", "visayas", "mindanao"]
for i,address in enumerate(addresses):
    df.loc[df["buyeraddress"].str.endswith(address, na=False), "buyer_region"] = i
    df.loc[df["selleraddress"].str.endswith(address, na=False), "seller_region"] = i


# In[ ]:


collections.Counter(df["buyer_region"])


# In[ ]:


collections.Counter(df["seller_region"])


# In[ ]:


df["sla"] = df[['buyer_region','seller_region']].progress_apply(lambda x: get_sla(*x), axis=1)
collections.Counter(df["sla"])


# # Get required deadline

# In[ ]:


import time
pattern = '%Y-%m-%d'
public_holidays = [int(time.mktime(time.strptime('2020-03-08', pattern)))//(24*60*60),
                   int(time.mktime(time.strptime('2020-03-25', pattern)))//(24*60*60),
                   int(time.mktime(time.strptime('2020-03-30', pattern)))//(24*60*60),
                   int(time.mktime(time.strptime('2020-03-31', pattern)))//(24*60*60)]
public_holidays = set(public_holidays)
public_holidays


# In[ ]:


(18329-3)%7   # '2020-03-08' is a Sunday


# In[ ]:


import functools

@functools.lru_cache(maxsize=None)
def working(epoch_day):
    epoch_day = int(epoch_day)
    if (epoch_day-3)%7 == 0:
        return False
    if (epoch_day) in public_holidays:
        return False
    return True


# In[ ]:


min_date, max_date = min(min(df["t1"]), min(df["t2"]), min(df["t3"])), max(max(df["t1"]), max(df["t2"]), max(df["t3"]))
non_working_days = set()
for date in range(int(min_date), int(max_date)+20):
    if not(working(date)):
        non_working_days.add(date)
non_working_days


# In[ ]:


@functools.lru_cache(maxsize=None)
def required(start, sla):
    start = int(start)
    while sla:
        if not start in non_working_days:
            sla -= 1
        start += 1
    while start in non_working_days:
        start += 1
    return start


# In[ ]:


[required(x,3) for x in range(18345,18355)]


# In[ ]:


[required(x,3) for x in range(18345,18355)]


# In[ ]:


df["t2_required"] = df[['t1','sla']].progress_apply(lambda x: required(*x), axis=1)
collections.Counter(df["t2_required"] - df["t1"])


# In[ ]:


df["t3_required"] = df['t2'].progress_apply(lambda x: required(x, 3))
collections.Counter(df["t3_required"] - df["t2"])


# # Combine judgement

# In[ ]:


df["t2_fail"] = df["t2"] > df["t2_required"]
collections.Counter(df["t2_fail"])


# In[ ]:


df["t3_fail"] = df["t3"] > df["t3_required"]
collections.Counter(df["t3_fail"])


# In[ ]:


df["result"] = df["t2_fail"] | df["t3_fail"]
df["is_late"] = df["result"].astype(int)
collections.Counter(df["is_late"])


# In[ ]:


df[["orderid", "is_late"]].to_csv("submission.csv", index=False)


# In[ ]:


df.head(5)


# In[ ]:


get_ipython().system('head submission.csv')


# In[ ]:


df.to_csv("not_the_submission.csv", index=False)


# # Adhoc

# In[ ]:


# on why our old submission failed to get 1.0 (version 16)


# In[ ]:


@functools.lru_cache(maxsize=None)
def g(xx):
    xx = xx.lower()
    for x in xx.split(",")[::-1]:
        res = ""
        if 'metro manila' in x:
            res += "0"
        if "luzon" in x:
            res += "1"
        if "visayas" in x:
            res += "2"
        if "mindanao" in x:
            res += "3"
        if res == "":
            continue
        return res
    return -1


# In[ ]:


df["bb"] = df["buyeraddress"].progress_apply(g)  # assume all this place first
collections.Counter(df["bb"])


# In[ ]:


df["ss"] = df["selleraddress"].progress_apply(g)  # assume all this place first
collections.Counter(df["ss"])


# In[ ]:




