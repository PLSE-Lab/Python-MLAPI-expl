#!/usr/bin/env python
# coding: utf-8

# I often start any competition with first benchmark result without any machine learning by applying average most of the time. Here by exploring train and test data it is very much clear that we can just carry forward 2016's data to 2017 and 2018 with just simple join of building and time.
# 
# Let's try to do it with minimal code.

# In[ ]:


import pandas as pd
train = pd.read_csv('/kaggle/input/ashrae-energy-prediction/train.csv')
test = pd.read_csv('/kaggle/input/ashrae-energy-prediction/test.csv')
train['timestamp'] = train['timestamp'].str[5:]
test['timestamp'] = test['timestamp'].str[5:]
submission = test.merge(train, how='left', on =['building_id','timestamp','meter'])
submission[['row_id','meter_reading']].fillna(0).to_csv('submission.csv',index=False)


# we are gettign 1.78 for this simple piece of code. In my opinion, couple of more lines will improve score significantly:
# 1) Introducing join by weekdays
# 2) Filling null with average value of meter rather than 0
# 
# At first glance it is very much clear building id , date and month , weekday will be most important features.
