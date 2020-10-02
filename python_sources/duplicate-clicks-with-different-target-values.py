#!/usr/bin/env python
# coding: utf-8

# I wondered how many prediction we might get wrong with duplicate values in the test data. These duplicate clicks might be diiffer in microsecond values. So I comapred day 9 train data duplicate click's target values which have different target with test data for percent of wrong predictions. 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
import time
import gc
from datetime import datetime
import os

os.listdir("../input")


# In[ ]:


dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }
print('loading train data...')
train_df = pd.read_csv("../input/train.csv", dtype=dtypes,skiprows=range(1,131886954),
                       usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'])
len_train = len(train_df)


# In[ ]:


train_df['click_time'] = pd.to_datetime(train_df['click_time'])
dup = train_df[train_df.duplicated(["ip", 'app', 'channel', 'device', 'os', 'click_time'], False)]
group = dup.groupby(['ip', 'app', 'channel', 'device', 'os', 'click_time']).is_attributed.mean().reset_index().rename(index=str, columns={'is_attributed': 'mean'})
dup = dup.merge(group, on=['ip', 'app', 'channel', 'device', 'os', 'click_time'], how='left')
del train_df
del group
gc.collect()
len_dup = len(dup)
print('Number of Duplicate clicks in train data: ', len_dup)


# Get the dupplicate clicks with different target values. Cannot simply drop duplicate from 'dup' because there are some clicks with more than 2 duplicates.

# In[ ]:


dup_diff_target = dup[(dup['mean']!=0.0) & (dup['mean'] !=1.0)]
len_dup_diff_target = len(dup_diff_target)
print('NUmber of duplicate clicks with different target values in train data: ', len_dup_diff_target)


# In[ ]:


dup_diff_target.head()


# Percent of duplicate clicks with different target values

# In[ ]:


len_dup_diff_target/len_dup


# In[ ]:


print('loading test supplement data...')
test_sup_df = pd.read_csv("../input/test_supplement.csv", dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])
test_sup_df['click_time'] = pd.to_datetime(test_sup_df['click_time'])
test_sup_dup = test_sup_df[test_sup_df.duplicated(["ip", 'app', 'channel', 'device', 'os', 'click_time'], False)]
len_test_sup = len(test_sup_df)
del test_sup_df
gc.collect()
len_test_sup_dup = len(test_sup_dup)
print('Number of Duplicate clicks in test supplement data: ', len_test_sup_dup)


# In[ ]:


print('loading test data...')
test_df = pd.read_csv("../input/test.csv", dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])
test_df['click_time'] = pd.to_datetime(test_df['click_time'])
test_dup = test_df[test_df.duplicated(["ip", 'app', 'channel', 'device', 'os', 'click_time'], False)]
del test_df
gc.collect()
len_test_dup = len(test_dup)
print('Number of Duplicate clicks in test(For submission) data: ', len_test_dup)


# In[ ]:


percent_train_dup = len_dup/len_train
print("percent of dupplicate in train data: ", percent_train_dup)

percent_test_dup = len_test_sup_dup/len_test_sup
print("percent of dupplicate in test data: ", percent_test_dup)

percent_of_test_csv_dup = len_test_dup/len_test_sup_dup
print("Percent of test.csv duplicates in test_supplement.csv: ", percent_of_test_csv_dup)

maybe_worng_preds = (len_test_sup_dup*len_dup_diff_target)/len_dup
print("Number of wrong(may be) predictions for test supplement: ", maybe_worng_preds)

maybe_worng_preds_for_submission = percent_of_test_csv_dup*maybe_worng_preds
print("Number of wrong(may be) predictions for submission: ", maybe_worng_preds_for_submission)


# In[ ]:


group = dup_diff_target.groupby(["ip", 'app', 'channel', 'device', 'os', 'click_time'])
first = group.nth(0).is_attributed.value_counts()
second = group.nth(1).is_attributed.value_counts()
third = group.nth(2).is_attributed.value_counts()
print('First click target value counts:\n', first)
print('Second click target value counts:\n', second)
print('Third click target value counts:\n', third)


# About 2/3rd of the taget values of 'duplicate clicks with diffrent tagets' have there taget values as "1" for lower index. which means first click has more probability than next one which is microseconds appart.  We will have more chances if we can find those duplicates which are going to have different target values among all the duplicate clicks.

# I think I made so many assumptions here. One of it is distribution of duplicate clicks for test and test_supplement.
# 
# I have done it for only day 9 of train data
# 
# and i might be wrong somewhere, if so please let me know

# In[ ]:




