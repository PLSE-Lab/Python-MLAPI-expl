#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


# In[2]:


train = pd.read_csv('../input/train_5clrc8b/train.csv', index_col='reservation_id', parse_dates=True)
test = pd.read_csv('../input/test_jwt0mqh/test.csv', index_col='reservation_id', parse_dates=True)


# In[7]:


train_ = pd.read_csv('../input/train_5clrc8b/train.csv', index_col='reservation_id', parse_dates=True)
test_ = pd.read_csv('../input/test_jwt0mqh/test.csv', index_col='reservation_id', parse_dates=True)


# In[3]:


print (train.shape)
print (test.shape)


# Checking for null values in the data set.

# In[4]:


print (train.isnull().sum()*100/train.shape[0])
print (test.isnull().sum()*100/test.shape[0])


# In[5]:


import datetime
import pandas as pd
from pandas_datareader import data
import re

def add_datepart(df, fldname, drop=True):
    fld = df[fldname]
    if not np.issubdtype(fld.dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)
    targ_pre = re.sub('[Dd]ate$', '', fldname)
    for n in ('Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
            'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start'):
        df[targ_pre+n] = getattr(fld.dt,n.lower())
    df[targ_pre+'Elapsed'] = fld.astype(np.int64) // 10**9
    if drop: df.drop(fldname, axis=1, inplace=True)


# In[6]:


add_datepart(train, 'booking_date')
add_datepart(test, 'booking_date')
add_datepart(train, 'checkin_date')
add_datepart(test, 'checkin_date')
add_datepart(train, 'checkout_date')
add_datepart(test, 'checkout_date')


# In[8]:


def convert_year(x):
    temp = x.split('/')
    temp[2] = '20'+temp[2]
    return '-'.join(temp)


# In[9]:


train_['checkin_date'] = train_['checkin_date'].astype('str').apply(lambda x: convert_year(x))
train_['checkout_date'] = train_['checkout_date'].astype('str').apply(lambda x: convert_year(x))
train_['booking_date'] = train_['booking_date'].astype('str').apply(lambda x: convert_year(x))


# In[43]:


test_['checkin_date'] = test_['checkin_date'].astype('str').apply(lambda x: convert_year(x))
test_['checkout_date'] = test_['checkout_date'].astype('str').apply(lambda x: convert_year(x))
test_['booking_date'] = test_['booking_date'].astype('str').apply(lambda x: convert_year(x))


# In[10]:


train_['checkin_date'] = pd.to_datetime(train_['checkin_date'])
train_['checkout_date'] = pd.to_datetime(train_['checkout_date'])
train_['booking_date'] = pd.to_datetime(train_['booking_date'])


# In[44]:


test_['checkin_date'] = pd.to_datetime(test_['checkin_date'])
test_['checkout_date'] = pd.to_datetime(test_['checkout_date'])
test_['booking_date'] = pd.to_datetime(test_['booking_date'])


# In[11]:


train_.head()


# In[22]:


train['chkout_chkin_diff'] = (train_['checkout_date']-train_['checkin_date']).dt.days
train['chkin_book_diff'] = (train_['checkin_date']-train_['booking_date']).dt.days


# In[45]:


test['chkout_chkin_diff'] = (test_['checkout_date']-test_['checkin_date']).dt.days
test['chkin_book_diff'] = (test_['checkin_date']-test_['booking_date']).dt.days


# In[23]:


import seaborn as sns
import matplotlib.pyplot as plt

fig, axs = plt.subplots(ncols=2)
sns.boxplot(x=train['chkout_chkin_diff'], orient='v', ax=axs[0])
sns.boxplot(x=train['chkin_book_diff'], orient='v', ax=axs[1])


# In[46]:


train.loc[train.chkout_chkin_diff < 0, 'chkout_chkin_diff'] = 0
train.loc[train.chkin_book_diff < 0, 'chkin_book_diff'] = 0
test.loc[test.chkout_chkin_diff < 0, 'chkout_chkin_diff'] = 0
test.loc[test.chkin_book_diff < 0, 'chkin_book_diff'] = 0


# In[25]:


fig, axs = plt.subplots(ncols=2)
sns.boxplot(x=train['chkout_chkin_diff'], orient='v', ax=axs[0])
sns.boxplot(x=train['chkin_book_diff'], orient='v', ax=axs[1])


# **Creating a new features - total number of people and total people not travelling******

# In[16]:


train['total_people'] = train['numberofadults'] + train['numberofchildren']
test['total_people'] = test['numberofadults'] + test['numberofchildren']


# In[34]:


train['not_travelling'] = train['total_people'] - train['total_pax']
test['not_travelling'] = test['total_people'] - test['total_pax']


# In[36]:


train.loc[train.not_travelling < 0, 'not_travelling'] = 0
train.loc[train.not_travelling < 0, 'not_travelling'] = 0
test.loc[test.not_travelling < 0, 'not_travelling'] = 0
test.loc[test.not_travelling < 0, 'not_travelling'] = 0


# In[35]:


sns.boxplot(x='not_travelling', data=train)


# In[29]:


fig, axs = plt.subplots(ncols=2)
sns.boxplot(x=train['total_people'], orient='v', ax=axs[0])
sns.boxplot(x=test['total_people'], orient='v', ax=axs[1])


# In[26]:


sns.scatterplot(x='total_people', y='amount_spent_per_room_night_scaled', data=train)


# In[32]:


sns.boxplot(x='numberofadults', orient='h',data=train)


# In[27]:


sns.scatterplot(x='numberofadults', y='amount_spent_per_room_night_scaled', data=train)


# In[33]:


sns.boxplot(x='numberofchildren', orient='h',data=train)


# In[28]:


sns.scatterplot(x='numberofchildren', y='amount_spent_per_room_night_scaled', data=train)


# Removing Memberid as it contains too many categories!!

# In[17]:


cat_vars = ['channel_code','main_product_code','resort_region_code','resort_type_code','room_type_booked_code','season_holidayed_code','state_code_residence','state_code_resort','member_age_buckets','booking_type_code','cluster_code','reservationstatusid_code',
            'resort_id', 'persontravellingid']
for col in cat_vars:
    print ('Processing ', col)
    print ('Train uniques', train[col].unique().shape)
    print ('Test uniques', test[col].unique().shape)
    train[col] = train[col].astype('str')
    test[col] = test[col].astype('str')


# ### Categorical variables Label Encoding.
# As tree based models have a lesser impact of encoding we first go with label encoding and check the results.****

# In[18]:


from sklearn.preprocessing import LabelEncoder
encoder = {}
for col in cat_vars:
    print ('Processing ', col)
    le = LabelEncoder()
    le.fit(train[col])
    train[col] = le.transform(train[col])
    for attr in test[col].unique().tolist():
        if attr not in le.classes_:
            le.classes_ = np.append(le.classes_, values=attr)
    encoder[col] = le
    test[col] = le.transform(test[col])


# In[19]:


coe_train = train['checkout_Elapsed'][0]
cie_train = train['checkin_Elapsed'][0]
boe_train = train['booking_Elapsed'][0]
train['checkout_Elapsed'] = train['checkout_Elapsed'] / coe_train
test['checkout_Elapsed'] = test['checkout_Elapsed'] / coe_train
train['checkin_Elapsed'] = train['checkin_Elapsed'] / cie_train
test['checkin_Elapsed'] = test['checkin_Elapsed'] / cie_train
train['booking_Elapsed'] = train['booking_Elapsed'] / boe_train
test['booking_Elapsed'] = test['booking_Elapsed'] / boe_train


# In[ ]:


train.shape


# #### Splitting data set into train/valid/test

# In[37]:


train_idx = range(0, round(0.8*len(train)))
valid_idx = range(round(0.8*len(train)), round(0.9*len(train)))
test_idx = range(round(0.9*len(train)), len(train))

tr = train.iloc[train_idx, :]
val = train.iloc[valid_idx, :]
tst = train.iloc[test_idx, :]

train_params = ['channel_code',
'main_product_code', 
'numberofadults', 
'numberofchildren', 
'persontravellingid', 
'resort_region_code', 
'resort_type_code', 
'room_type_booked_code', 
'roomnights', 
'season_holidayed_code', 
'state_code_residence', 
'state_code_resort', 
'total_pax', 
'member_age_buckets', 
'booking_type_code', 
'cluster_code', 
'reservationstatusid_code', 
'resort_id', 
'booking_Year', 
'booking_Month', 
'booking_Week', 
'booking_Day', 
'booking_Dayofweek', 
'booking_Dayofyear', 
'booking_Is_month_end', 
'booking_Is_month_start', 
'booking_Is_quarter_end', 
'booking_Is_quarter_start', 
'booking_Is_year_end', 
'booking_Is_year_start', 
'booking_Elapsed', 
'checkin_Year', 
'checkin_Month', 
'checkin_Week', 
'checkin_Day', 
'checkin_Dayofweek', 
'checkin_Dayofyear', 
'checkin_Is_month_end', 
'checkin_Is_month_start', 
'checkin_Is_quarter_end', 
'checkin_Is_quarter_start', 
'checkin_Is_year_end', 
'checkin_Is_year_start', 
'checkin_Elapsed', 
'checkout_Year', 
'checkout_Month', 
'checkout_Week', 
'checkout_Day', 
'checkout_Dayofweek', 
'checkout_Dayofyear', 
'checkout_Is_month_end', 
'checkout_Is_month_start', 
'checkout_Is_quarter_end', 
'checkout_Is_quarter_start', 
'checkout_Is_year_end', 
'checkout_Is_year_start', 
'checkout_Elapsed', 
'total_people',
'not_travelling',
'chkout_chkin_diff',
'chkin_book_diff']

X_train = tr[train_params]
y_train = tr['amount_spent_per_room_night_scaled'].ravel()
X_val = val[train_params]
y_val = val['amount_spent_per_room_night_scaled'].ravel()
X_test = tst[train_params]
y_test = tst['amount_spent_per_room_night_scaled'].ravel()


# ### Trying First Model with basic configuration

# In[38]:


from xgboost import XGBRegressor
""""
x1 = XGBRegressor()
x1.fit(X_train, y_train)"""


# In[39]:


x4 = XGBRegressor(n_estimators=1000)
from sklearn.metrics import mean_squared_error, r2_score
x4.fit(X_train, y_train, early_stopping_rounds=2, eval_set=[(X_train, y_train),(X_val, y_val)])


# In[40]:


y_train_pred4 = x4.predict(X_train)
y_pred4 = x4.predict(X_test)

print('Train r2 score: ', r2_score(y_train_pred4, y_train))
print('Test r2 score: ', r2_score(y_test, y_pred4))
train_mse4 = mean_squared_error(y_train_pred4, y_train)
test_mse4 = mean_squared_error(y_pred4, y_test)
train_rmse4 = np.sqrt(train_mse4)
test_rmse4 = np.sqrt(test_mse4)
print('Train RMSE: %.4f' % train_rmse4)
print('Test RMSE: %.4f' % test_rmse4)


# In[ ]:



""""
y_train_pred1 = x1.predict(X_train)
y_pred1 = x1.predict(X_test)

print('Train r2 score: ', r2_score(y_train_pred1, y_train))
print('Test r2 score: ', r2_score(y_test, y_pred1))
train_mse1 = mean_squared_error(y_train_pred1, y_train)
test_mse1 = mean_squared_error(y_pred1, y_test)
train_rmse1 = np.sqrt(train_mse1)
test_rmse1 = np.sqrt(test_mse1)
print('Train RMSE: %.4f' % train_rmse1)
print('Test RMSE: %.4f' % test_rmse1)"""


# #### Second Model

# In[ ]:


""""x2 = XGBRegressor(n_estimators=1000)
x2.fit(X_train, y_train, early_stopping_rounds=2, eval_set=[(X_train, y_train),(X_val, y_val)])"""


# In[ ]:


""""y_train_pred2 = x2.predict(X_train)
y_pred2 = x2.predict(X_test)

print('Train r2 score: ', r2_score(y_train_pred2, y_train))
print('Test r2 score: ', r2_score(y_test, y_pred2))
train_mse2 = mean_squared_error(y_train_pred2, y_train)
test_mse2 = mean_squared_error(y_pred2, y_test)
train_rmse2 = np.sqrt(train_mse2)
test_rmse2 = np.sqrt(test_mse2)
print('Train RMSE: %.4f' % train_rmse2)
print('Test RMSE: %.4f' % test_rmse2)"""


# #### Third Model Best Attributes

# In[ ]:





# In[ ]:


""""x2 = XGBRegressor(n_estimators=1000)
x2.fit(X_train, y_train, early_stopping_rounds=2, eval_set=[(X_train, y_train),(X_val, y_val)])"""


# In[ ]:


""""most_relevant_features = ['total_people',
 'total_pax',
 'roomnights',
 'persontravellingid',
 'resort_region_code',
 'channel_code',
 'resort_id',
 'numberofadults',
 'season_holidayed_code',
 'booking_Elapsed',
 'state_code_resort',
 'main_product_code',
 'checkin_Elapsed',
 'checkin_Dayofyear',
 'resort_type_code',
 'state_code_residence',
 'checkout_Elapsed',
 'room_type_booked_code',
 'checkin_Day',
 'member_age_buckets',
 'checkout_Day',
 'checkout_Month',
 'booking_Day',
 'cluster_code',
 'numberofchildren',
 'checkout_Dayofweek',
 'booking_Dayofyear',
 'checkin_Dayofweek',
 'checkin_Year',
 'booking_type_code',
 'booking_Dayofweek',
 'checkout_Dayofyear']"""


# In[41]:


sub = pd.read_csv('../input/sample_submission_dlc0jkw/sample_submission.csv')


# In[47]:


sub['amount_spent_per_room_night_scaled'] = x4.predict(test[train_params])


# In[48]:


sub.to_csv('sub2.csv', index=False)


# In[ ]:


#from collections import OrderedDict
#attributes = OrderedDict(sorted(x2.get_booster().get_fscore().items(), key=lambda t: t[1], reverse=True))


# In[ ]:


#most_relevant_features


# In[ ]:


#


# In[ ]:


#most_relevant_features= list( dict((k, v) for k, v in x2.get_booster().get_fscore().items() if v >= 4).keys())

