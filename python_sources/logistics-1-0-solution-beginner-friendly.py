#!/usr/bin/env python
# coding: utf-8

# A Big Thanks to the notebook contributer to let me get the ideas to solve this question as a beginner!
# 1. [3 hooman 1 cat - Logistics](http://www.kaggle.com/xuanquang1999/3-hooman-1-cat-logistics)
# 2. [week5](http://www.kaggle.com/huikang/week5)

# # Tips to get 1.0
# There are some tips to get 1.0 from this disscussion. Another big thanks!(https://www.kaggle.com/c/open-shopee-code-league-logistic/discussion/165972):
# > 1. You can easily count the weekday difference with the help of numpy.busday_count function! Check the "weekmask" and "holidays" in numpy document.
# > 2. We use pd.to_datetime to process the epoch time from "deliveryordersmarch.csv". However, don't forget we need to convert the time to GMT+8.
# > 3. I learned from the discussion forum [1] that if the weekday of "pick" is Sunday, we need to shift it to next day.
# > 4. You can use datetime.timedelta() to add a day or a hour to a datetime object.
# > 
# > [1]. https://www.kaggle.com/c/open-shopee-code-league-logistic/discussion/165829

# # Set environment

# In[ ]:


import numpy as np
import pandas as pd


# # Data manipulate

# In[ ]:


data = pd.read_csv('../input/logistics-shopee-code-league/delivery_orders_march.csv')


# In[ ]:


# look at data
data.head()


# In[ ]:


data.info()


# In[ ]:


# get the state of the address by getting the last str in the address
data['buyeraddress'] = data['buyeraddress'].apply(lambda x: x.split()[-1]).str.lower()
data['selleraddress'] = data['selleraddress'].apply(lambda x: x.split()[-1]).str.lower()
data.head()


# In[ ]:


# calculate the business days to deliver before late 
temp = []
for i,j in data[['buyeraddress','selleraddress']].itertuples(index=False):
  if (i == 'manila' and j == 'manila'):
    temp.append(3)
  elif (
      (i == 'manila' and j == 'luzon')
      or (i == 'luzon' and j == 'manila')
      or (i == 'luzon' and j == 'luzon')
      ):
    temp.append(5)
  else:
    temp.append(7)

data['days'] = temp
data.head()


# In[ ]:


# change unix time to date
# need to convert to gmt+8
data[['pick','1st_deliver_attempt','2nd_deliver_attempt']] += 8*60*60
data['pick'] = pd.to_datetime(data['pick'],unit='s').dt.date
data['1st_deliver_attempt'] = pd.to_datetime(data['1st_deliver_attempt'],unit='s').dt.date
data['2nd_deliver_attempt'] = data['2nd_deliver_attempt'].replace(np.nan,0) # change nan to 0 or else can't be process
data['2nd_deliver_attempt'] = pd.to_datetime(data['2nd_deliver_attempt'],unit='s').dt.date
data.head()


# In[ ]:


# count how many days of business day taken for the 1pick and 2pick
holiday = ['2020-03-08','2020-03-25','2020-03-30','2020-03-31']
data['1st_pick'] = np.busday_count(data['pick'], data['1st_deliver_attempt'], weekmask='1111110', holidays=holiday)
data['2nd_pick'] = np.busday_count(data['1st_deliver_attempt'], data['2nd_deliver_attempt']	, weekmask='1111110', holidays=holiday)
data.head()


# In[ ]:


# check if is late
data['is_late'] = (data['1st_pick'] > data['days']) | (data['2nd_pick'] > 3)
data.head()


# # Prepare Submit

# In[ ]:


# prepare submission df and change is_late column to int using .apply(int)
submission = pd.DataFrame({'orderid':data['orderid'], 'is_late':data['is_late'].apply(int)})
submission


# In[ ]:


submission.to_csv('submission.csv', header=True, index=False)

