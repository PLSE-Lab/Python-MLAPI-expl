#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor, FeaturesData, Pool
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import datetime
from datetime import timedelta
import os
print(os.listdir("../input"))
get_ipython().run_line_magic('matplotlib', 'inline')
# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().run_line_magic('time', '')
# ,parse_dates=['booking_date', 'checkin_date', 'checkout_date']
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
print(train.shape)
print(test.shape)


# In[ ]:


data = train.append(test,ignore_index = True) 
print(data.shape)
data.head()


# In[ ]:


# %time
# data.info()


# In[ ]:


for column in data:
    if data[column].dtype=='int64':
        data[column] = data[column].astype(np.int16)
for column in data:
    if data[column].dtype=='float64':
        data[column] = data[column].astype(np.float32)


# In[ ]:


# data.info()


# In[ ]:


# %time
# data.describe()


# In[ ]:


# %time
# data.isna().sum()


# In[ ]:


# %time
# data.nunique()


# flight-specific forecasts using a time series method for historical bookings, a regression method for advanced bookings,
# and a combined model with both advanced bookings and historical data.
# He found that the combined approach worked
# better, but he did not consider unconstrained demand (i.e., demand unconstrained by the capacity of the plane or hotel).
# In addition, his forecasts were made on monthly data, and therefore do not provide the necessary level of detail.
# Sa (1987) used multiple regression to develop a combined forecast. The dependent variable used was reservations
# remaining while the independent variables included the number of reservations on hand, a seasonal index, a weekly index,
# and an average of historical reservations remaining. The regression method was run for various days before departure (t
# = 7, 14, 21, and 28). Unfortunately Sa did not test the accuracy of his methods and did not consider the impact of
# unconstrained demand.
# Wickham (1995) presented a simple linear regression method in which the independent variable was the number of
# reservations on hand for a flight at a particular reading day and the dependent variable was the final number of seats sold. 
#  proprietary hotel industry methods advocate a weighted average of the historical forecast and the
# advanced booking forecast. When the day of arrival is far in the future, more weight is put on the historical forecast,
# whereas when the day of arrival is imminent, more weight is put on the advanced booking forecast.
# Weatherford (1998) compared additive methods, multiplicative methods, and regression in an airline context and
# found that additive methods and regression out-performed multiplicative methods.

# In[ ]:


get_ipython().run_line_magic('time', '')
def feature_engineering(df):
    
    df.loc[:,'booking_date'] = pd.to_datetime(df['booking_date'], format="%d/%m/%y",infer_datetime_format=True)
    df.loc[:,'checkin_date'] = pd.to_datetime(df['checkin_date'], format="%d/%m/%y",infer_datetime_format=True)
    df.loc[:,'checkout_date'] = pd.to_datetime(df['checkout_date'], format="%d/%m/%y",infer_datetime_format=True)
    
    df.loc[:,'checkin_day'] = df['checkin_date'].apply(lambda x : x.day)
    df.loc[:,'checkin_month'] = df['checkin_date'].apply(lambda x : x.month) 
    df.loc[:,'checkin_year'] = df['checkin_date'].apply(lambda x : x.year)
    df.loc[:,'checkin_day_of_year'] = df['checkin_date'].apply(lambda x : (x - datetime.datetime(x.year, 1, 1)).days + 1) 
    df.loc[:,'checkin_weekday'] = df['checkin_date'].apply(lambda x : x.weekday())    
    
    df.loc[:,'checkout_day'] = df['checkout_date'].apply(lambda x : x.day)
    df.loc[:,'checkout_month'] = df['checkout_date'].apply(lambda x : x.month) 
    df.loc[:,'checkout_year'] = df['checkout_date'].apply(lambda x : x.year)
    df.loc[:,'checkout_day_of_year'] = df['checkout_date'].apply(lambda x : (x - datetime.datetime(x.year, 1, 1)).days + 1) 
    df.loc[:,'checkout_weekday'] = df['checkout_date'].apply(lambda x : x.weekday()) 
    
    df.loc[:,'no_of_rooms'] = (df['numberofadults'] + df['numberofchildren'])/df['total_pax']
    df.loc[:,'trip_length'] = df['checkout_date'] - df['checkin_date']
    df.loc[:,'days_before_planning'] = df['checkin_date'] - df['booking_date']
    df.loc[:,'trip_length'] = df['trip_length'].apply(lambda x : x.days)
    df.loc[:,'days_before_planning'] = df['days_before_planning'].apply(lambda x : x.days)
    trip_count = df.groupby(['memberid'])['reservation_id'].agg(['count'])
    df.loc[:,'trip_count'] = df['memberid'].apply(lambda i: trip_count.loc[i][0])
    
    #handle data irregularity
    df['days_before_planning'] = df['days_before_planning'].apply(lambda x : x if x>=0 else 0)
    
    for column in df:
        if df[column].dtype=='int64':
            df[column] = df[column].astype(np.int16)
    for column in df:
        if df[column].dtype=='float64':
            df[column] = df[column].astype(np.float32)
    return df
    
# def drop(df):
#     to_drop = ['reservation_id','memberid','booking_date', 'checkin_date', 'checkout_date']
#     return df.drop(to_drop,axis=1)

def create_data(df):
    df = feature_engineering(df)
#     df = drop(df)
    return df


# In[ ]:


data.shape


# In[ ]:


get_ipython().run_line_magic('time', '')
dataset = create_data(data)


# In[ ]:


dic = dict()
for i,r in dataset.iterrows():
    print(i)
#     ,r['resort_id'],r['checkin_date'].date(),r['checkout_date'].date(),r['trip_length']
    for k in range(r['trip_length']+1):
        if r['resort_id'] in dic.keys():
            if str((r['checkin_date'] + timedelta(days=k)).date()) in dic[r['resort_id']].keys():
                dic[r['resort_id']][str((r['checkin_date'] + timedelta(days=k)).date())] += 1
            else:
                dic[r['resort_id']][str((r['checkin_date'] + timedelta(days=k)).date())] = 1
        else:
            dic[r['resort_id']] = {}
            if str((dataset.loc[i,'checkin_date'] + timedelta(days=k)).date()) in dic[r['resort_id']].keys():
                dic[r['resort_id']][str((r['checkin_date'] + timedelta(days=k)).date())] += 1
            else:
                dic[r['resort_id']][str((r['checkin_date'] + timedelta(days=k)).date())] = 1


# In[ ]:


print(len(dic))


# In[ ]:


maxdic = dict()
for key, value in dic.items():
    maxdic[key] = max(dic[key].values())
#     print(key, len(dic[key]))
print(maxdic)


# In[ ]:


def get_booking(row):
#     print(row)
#     print(row['resort_id'],str(row['checkin_date'].date()))
    return dic[row['resort_id']][str(row['checkin_date'].date())]


# In[ ]:


dataset.loc[:,'resort_max_bookings'] = dataset['resort_id'].map(maxdic) 


# In[ ]:


dataset.loc[:,'current_occupancy']=dataset.apply(lambda x : get_booking(x),axis=1)


# In[ ]:


dataset.columns


# In[ ]:


dataset.head()


# In[ ]:


f, axs = plt.subplots(2,2,figsize=(10,10))
plt.subplot(2,2, 1)
sns.boxplot(dataset['numberofadults'])
plt.xlabel('numberofadults')

plt.subplot(2,2, 2)
sns.boxplot(dataset['numberofchildren'])
plt.xlabel('numberofchildren')

plt.subplot(2,2, 3)
sns.boxplot(dataset['total_pax'])
plt.xlabel('total_pax')

plt.subplot(2,2, 4)
sns.boxplot(dataset['roomnights'])
plt.xlabel('roomnights')


# In[ ]:


# fig, ax = plt.subplots(2, 3, sharex='col', sharey='row')
f, axs = plt.subplots(4,3,figsize=(20,20))
plt.subplot(4, 3, 1)
sns.countplot(dataset['channel_code'])
plt.xlabel('channel_code')
plt.ylabel('count')

plt.subplot(4, 3, 2)
sns.countplot(dataset['booking_type_code'])
plt.xlabel('booking_type_code')
plt.ylabel('count')

plt.subplot(4, 3, 3)
sns.countplot(dataset['cluster_code'])
plt.xlabel('cluster_code')
plt.ylabel('count')

plt.subplot(4, 3, 4)
sns.countplot(dataset['main_product_code'])
plt.xlabel('main_product_code')
plt.ylabel('count')

plt.subplot(4, 3, 5)
sns.countplot(dataset['member_age_buckets'])
plt.xlabel('member_age_buckets')
plt.ylabel('count')

plt.subplot(4, 3, 6)
sns.countplot(dataset['reservationstatusid_code'])
plt.xlabel('reservationstatusid_code')
plt.ylabel('count')

plt.subplot(4, 3, 7)
sns.countplot(dataset['resort_region_code'])
plt.xlabel('resort_region_code')
plt.ylabel('count')

plt.subplot(4, 3, 8)
sns.countplot(dataset['resort_type_code'])
plt.xlabel('resort_type_code')
plt.ylabel('count')

plt.subplot(4, 3, 9)
sns.countplot(dataset['room_type_booked_code'])
plt.xlabel('room_type_booked_code')
plt.ylabel('count')

plt.subplot(4, 3, 10)
sns.countplot(dataset['season_holidayed_code'])
plt.xlabel('season_holidayed_code')
plt.ylabel('count')

plt.subplot(4, 3, 11)
sns.countplot(dataset['state_code_residence'])
plt.xlabel('state_code_residence')
plt.ylabel('count')

plt.subplot(4, 3, 12)
sns.countplot(dataset['state_code_resort'])
plt.xlabel('state_code_resort')
plt.ylabel('count')



# In[ ]:



f, axs = plt.subplots(4,3,figsize=(20,20))
plt.subplot(3, 3, 1)
sns.distplot(train['amount_spent_per_room_night_scaled'],kde=False)
plt.xlabel('amount_spent_per_room_night_scaled')
# plt.ylabel('count')

plt.subplot(3, 3, 2)
sns.distplot(dataset['numberofadults'],kde=False)
plt.title('numberofadults')
# plt.ylabel('count')

plt.subplot(3, 3, 3)
sns.distplot(dataset['numberofchildren'],kde=False)
plt.title('cluster_code')
# plt.ylabel('count')

plt.subplot(3, 3, 4)
sns.distplot(dataset['roomnights'],kde=False)
plt.title('roomnights')
# plt.ylabel('count')

plt.subplot(3, 3, 5)
sns.distplot(dataset['trip_length'],kde=False)
plt.title('trip_length')
# plt.ylabel('count')

plt.subplot(3, 3, 6)
sns.distplot(dataset['total_pax'],kde=False)
plt.title('total_pax')
# plt.ylabel('count')


plt.subplot(3, 3, 7)
sns.distplot(dataset['trip_count'],kde=False)
plt.title('trip_count')

plt.subplot(3, 3, 8)
sns.distplot(dataset['days_before_planning'],kde=False)
plt.title('days_before_planning')


# In[ ]:


sns.distplot(train[train['season_holidayed_code']==1]['amount_spent_per_room_night_scaled'],label='1',color='r')
sns.distplot(train[train['season_holidayed_code']==2]['amount_spent_per_room_night_scaled'],label='2',color='y')
sns.distplot(train[train['season_holidayed_code']==3]['amount_spent_per_room_night_scaled'],label='3',color='g')
sns.distplot(train[train['season_holidayed_code']==4]['amount_spent_per_room_night_scaled'],label='4',color='b')
plt.xlabel('channel_code')


# In[ ]:


sns.distplot(train[train['room_type_booked_code']==1]['amount_spent_per_room_night_scaled'],label='1',color='r')
sns.distplot(train[train['room_type_booked_code']==2]['amount_spent_per_room_night_scaled'],label='2',color='y')
sns.distplot(train[train['room_type_booked_code']==3]['amount_spent_per_room_night_scaled'],label='3',color='g')
sns.distplot(train[train['room_type_booked_code']==4]['amount_spent_per_room_night_scaled'],label='4',color='b')
sns.distplot(train[train['room_type_booked_code']==5]['amount_spent_per_room_night_scaled'],label='4',color='b')
plt.xlabel('room_type_booked_code')


# In[ ]:


dataset.columns


# In[ ]:


cat =[]
for column in dataset:
    if 'id' in column or 'code' in column:
        cat.append(column)
cat.append('member_age_buckets')
print(cat)


# In[ ]:


for col in cat:
    dataset[col] = dataset[col].astype(str)


# In[ ]:


# Function to determine if column in dataframe is string.
def is_str(col):
    for i in col:
        if pd.isnull(i):
            continue
        elif isinstance(i, str):
            return True
        else:
            return False
# Splits the mixed dataframe into categorical and numerical features.
def split_features(df):
    cfc = []
    nfc = []
    for column in df:
        if is_str(df[column]):
            cfc.append(column)
        else:
            nfc.append(column)
    return df[cfc], df[nfc]


# In[ ]:


def preprocess(cat_features, num_features):
    cat_features = cat_features.fillna("None")
    for column in num_features:
        num_features[column].fillna(np.nanmean(num_features[column]), inplace=True)
    return cat_features, num_features


# In[ ]:


train_df = dataset[dataset['amount_spent_per_room_night_scaled'].isnull()!=True]
test_df = dataset[dataset['amount_spent_per_room_night_scaled'].isnull()==True]


# In[ ]:


train_df = train_df[(train_df['numberofadults']<7)]
train_df = train_df[(train_df['numberofchildren']<=3)]
train_df = train_df[(train_df['total_pax']<7)]


# In[ ]:


print(train_df.shape)


# In[ ]:


y_train = train_df['amount_spent_per_room_night_scaled']
to_drop=['amount_spent_per_room_night_scaled','reservation_id','memberid','booking_date', 'checkin_date', 'checkout_date']
X_train = train_df.drop(to_drop, axis=1)
X_test = test_df.drop(to_drop,axis=1)
# dftrain=dataset[dataset['amount_spent_per_room_night_scaled'].isnull()!=True]
# dftest=dataset[dataset['amount_spent_per_room_night_scaled'].isnull()==True]
# dftrain.head()


# In[ ]:


# # Apply the "split_features" function on the data.
cat_tmp_train, num_tmp_train = split_features(X_train)
cat_tmp_test, num_tmp_test = split_features(X_test)


# In[ ]:


# cat_tmp_train.head()


# In[ ]:


# Now to apply the "preprocess" function.
# Getting a "SettingWithCopyWarning" but I usually ignore it.
cat_features_train, num_features_train = preprocess(cat_tmp_train, num_tmp_train)
cat_features_test, num_features_test = preprocess(cat_tmp_test, num_tmp_test)


# In[ ]:


train_pool = Pool(
    data = FeaturesData(num_feature_data = np.array(num_features_train.values, dtype=np.float32), 
                    cat_feature_data = np.array(cat_features_train.values, dtype=object), 
                    num_feature_names = list(num_features_train.columns.values), 
                    cat_feature_names = list(cat_features_train.columns.values)),
    label =  np.array(y_train, dtype=np.float32)
)


# In[ ]:


test_pool = Pool(
    data = FeaturesData(num_feature_data = np.array(num_features_test.values, dtype=np.float32), 
                    cat_feature_data = np.array(cat_features_test.values, dtype=object), 
                    num_feature_names = list(num_features_test.columns.values), 
                    cat_feature_names = list(cat_features_test.columns.values))
)


# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold


# In[ ]:


# model = CatBoostRegressor(loss_function = 'RMSE') 
# Fit model iterations=4000,, learning_rate=0.05, depth=7
# model.fit(train_pool,early_stopping_rounds=3000)
# # Get predictions
# preds = model.predict(test_pool)


# In[ ]:


cat_features = cat_tmp_train.columns
params = {'depth': [4, 6, 8],
          'random_seed' : [400],
          'learning_rate' : [0.01, 0.05, 0.1],
         'iterations'    : [4000, 5000, 8000]}
# params = {'depth': [4],
#           'random_seed' : [400],
#           'learning_rate' : [ 0.05],
#          'iterations'    : [400]}
cb = CatBoostRegressor(loss_function='RMSE', iterations = 5000, learning_rate=0.03)
cb_model = GridSearchCV(cb, params, scoring='neg_mean_squared_error', cv = 5)
cb_model.fit(X_train, y_train, cat_features = cat_features)


# In[ ]:


cb_model.best_params_


# In[ ]:


model = CatBoostRegressor(**cb_model.best_params_,loss_function = 'RMSE') 
# Fit model
model.fit(train_pool,early_stopping_rounds=3000)
# Get predictions
preds = model.predict(test_pool)


# In[ ]:


res_id = test['reservation_id']


# In[ ]:


df = pd.DataFrame({'reservation_id': res_id, 'amount_spent_per_room_night_scaled': preds}, columns=['reservation_id', 'amount_spent_per_room_night_scaled'])
df.to_csv("submission.csv", index=False)


# In[ ]:




