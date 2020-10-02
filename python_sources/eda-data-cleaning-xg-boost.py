#!/usr/bin/env python
# coding: utf-8

# ### Importing Data and Libraries

# In[ ]:


# load some default Python modules
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

plt.style.use('seaborn-whitegrid')

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


train = pd.read_csv("../input/train.csv", nrows = 5_000_000)
print("shape of train data", train.shape)
train.head()


# In[ ]:


# datatypes
train.dtypes


# In[ ]:


# Basic Stats of the data set
train.describe()


# **following thing that i can notice**
# - Fare amount is negative and it doesn't seem to be realistic
# - few longitude and lattitude entries are off
# - maximum passanger count is 208 which looks odd
# 

# In[ ]:


print("old size: %d" % len(train))
train = train[train.fare_amount >=0]
print("New size: %d" % len(train))


# In[ ]:


# check missing data
train.isnull().sum()


# **There are 36 records where longitud and latitude are missing, we will drop them from the data**

# In[ ]:


print("old size: %d" % len(train))
train = train.dropna(how='any', axis=0)
print("New size after dropping missing value: %d" % len(train))


# In[ ]:


# Lets see the distribution of fare amount 
train.fare_amount.hist(bins=100, figsize = (16,8))
plt.xlabel("Fare Amount")
plt.ylabel("Frequency")


# - Looks like the distribution is highly skewed and frequency above 100 is very less
# - we will plot below 100 and above 100 separately 

# In[ ]:


# Lets see the distribution of fare amount less than 100
train[train.fare_amount <100 ].fare_amount.hist(bins=100, figsize = (16,8))
plt.xlabel("Fare Amount")
plt.ylabel("Frequency")


# - There are few points between 40 and 60 dollars which has slightly high frequency and that could be airport trips

# In[ ]:


train[train.fare_amount >100 ].shape


# - we can see here that there are total 1977 trips which are above 100 dollars
# - some of them might be outliers or few of them might be long distance trip from/to airport, we will see it in later sectionn

# In[ ]:


# Lets see the distribution of fare amount more than 100
train[train.fare_amount >100 ].fare_amount.hist(bins=100, figsize = (16,8))
plt.xlabel("Fare Amount")
plt.ylabel("Frequency")


# #### Lets also check passanger count distribution

# In[ ]:


# checking for passanger count greater than 7
train[train.passenger_count >7].passenger_count.hist(bins=10, figsize = (16,8))
plt.xlabel("Passanger Count")
plt.ylabel("Frequency")


# In[ ]:


# data for passanger count greater than 7
train[train.passenger_count >7]


# - There is only 11  entry for  passanger count above 7 , rest are below 7
# - we will drop this record in data cleaning section

# In[ ]:


# checking for passanger count less than 7
train[train.passenger_count <7].passenger_count.hist(bins=10, figsize = (16,8))
plt.xlabel("Passanger Count")
plt.ylabel("Frequency")


# - Most of the trips are taken by single passanger
# - we will try to see if there is any relation between passanger count and fare amout

# In[ ]:


# checking for records where passanger count is 0
train[train.passenger_count ==0].shape


# **We have 17K such cases where passanger count is zero, there can be two possibility**
#    - Passanger count is incorrectly populated
#    - Taxi was not carrying any passanger, may be taxi was used for goods
# 
# **we will look into test data set and finalize whether we should drop these cases or not.**

# In[ ]:


plt.figure(figsize= (16,8))
sns.boxplot(x = train[train.passenger_count< 7].passenger_count, y = train.fare_amount)


# - As we can see from the box plot median price of each passanger counts looks similar except one record, There are few outliers we wil treat in cleaning section
# - we will try to see if there is any relationship between passanger count and fare amount using correlation factor

# In[ ]:


train[train.passenger_count <7][['fare_amount','passenger_count']].corr()


# **There is very weak correlation (0.013) between fare amount and passanger count**

# ### Lets read the test data

# In[ ]:


test = pd.read_csv("../input/test.csv")
print("shape of test data", test.shape)
test.head()


# In[ ]:


#check for missing value
test.isnull().sum()


# There are no missing value in test data set

# In[ ]:


# checking for basic stats
test.describe()


# We will store the minimum and maximum of the longitude and latitude from test data set and filter the train data set for those data points

# In[ ]:


min(test.pickup_longitude.min(),test.dropoff_longitude.min()), max(test.pickup_longitude.max(),test.dropoff_longitude.max())


# In[ ]:


min(test.pickup_latitude.min(),test.dropoff_latitude.min()), max(test.pickup_latitude.max(),test.dropoff_latitude.max())


# In[ ]:


# this function will also be used with the test set below
def select_within_test_boundary(df, BB):
    return (df.pickup_longitude >= BB[0]) & (df.pickup_longitude <= BB[1]) &            (df.pickup_latitude >= BB[2]) & (df.pickup_latitude <= BB[3]) &            (df.dropoff_longitude >= BB[0]) & (df.dropoff_longitude <= BB[1]) &            (df.dropoff_latitude >= BB[2]) & (df.dropoff_latitude <= BB[3])


# In[ ]:


BB = (-74.5, -72.8, 40.5, 41.8)
print('Old size: %d' % len(train))
train = train[select_within_test_boundary(train, BB)]
print('New size: %d' % len(train))


# - Now we have sliced the train data records as per the coordinates of the test data

# ### Manual Feature Engineering
# - Adding distance metrics
# - few time based variables

# In[ ]:


def prepare_time_features(df):
    df['pickup_datetime'] = df['pickup_datetime'].str.slice(0, 16)
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], utc=True, format='%Y-%m-%d %H:%M')
    df['hour_of_day'] = df.pickup_datetime.dt.hour
#     df['week'] = df.pickup_datetime.dt.week
    df['month'] = df.pickup_datetime.dt.month
    df["year"] = df.pickup_datetime.dt.year
#     df['day_of_year'] = df.pickup_datetime.dt.dayofyear
#     df['week_of_year'] = df.pickup_datetime.dt.weekofyear
    df["weekday"] = df.pickup_datetime.dt.weekday
#     df["quarter"] = df.pickup_datetime.dt.quarter
#     df["day_of_month"] = df.pickup_datetime.dt.day
    
    return df


# In[ ]:


train = prepare_time_features(train)
test = prepare_time_features(test)


# In[ ]:


# calculate-distance-between-two-latitude-longitude-points-haversine-formula 
# Returns distance in miles
def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295 # Pi/180
    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a))   # 2*R*asin...


# In[ ]:


train['distance_miles'] = distance(train.pickup_latitude, train.pickup_longitude,                                       train.dropoff_latitude, train.dropoff_longitude)


# In[ ]:


test['distance_miles'] = distance(test.pickup_latitude, test.pickup_longitude,                                       test.dropoff_latitude, test.dropoff_longitude)


# #### Calculating pickup and drop distance from all 3 airports of Air Ports

# In[ ]:


def transform(data):
    # Distances to nearby airports, 
    jfk = (-73.7781, 40.6413)
    ewr = (-74.1745, 40.6895)
    lgr = (-73.8740, 40.7769)

    data['pickup_distance_to_jfk'] = distance(jfk[1], jfk[0],
                                         data['pickup_latitude'], data['pickup_longitude'])
    data['dropoff_distance_to_jfk'] = distance(jfk[1], jfk[0],
                                           data['dropoff_latitude'], data['dropoff_longitude'])
    data['pickup_distance_to_ewr'] = distance(ewr[1], ewr[0], 
                                          data['pickup_latitude'], data['pickup_longitude'])
    data['dropoff_distance_to_ewr'] = distance(ewr[1], ewr[0],
                                           data['dropoff_latitude'], data['dropoff_longitude'])
    data['pickup_distance_to_lgr'] = distance(lgr[1], lgr[0],
                                          data['pickup_latitude'], data['pickup_longitude'])
    data['dropoff_distance_to_lgr'] = distance(lgr[1], lgr[0],
                                           data['dropoff_latitude'], data['dropoff_longitude'])
    
    return data

train = transform(train)
test = transform(test)


# - Delete these 15 record where distance covered and fare amount are zero, which won't help our model

# In[ ]:


train[(train['distance_miles']==0)&(train['fare_amount']==0)]


# In[ ]:


print("old size: %d" % len(train))
train = train.drop(index= train[(train['distance_miles']==0)&(train['fare_amount']==0)].index, axis=0)
print("New size: %d" % len(train))


# - There are 24 records where fare amount is zero but distance travelled is greater than 0, we will drop such case

# In[ ]:


train[train['fare_amount']==0].shape


# In[ ]:


print("old size: %d" % len(train))
train = train.drop(index= train[train['fare_amount']==0].index, axis=0)
print("New size: %d" % len(train))


# - There are 53 record where fare amount is less than 2.5 dollars (most of them are 0.01 dollars except 3 cases of fare between 1 and 2 dollar ) which doesn't make sense as the base fare for any taxi in new york is 2.5 dollars, we will drop those cases

# In[ ]:


train[train['fare_amount'] < 2.5].shape


# In[ ]:


print("old size: %d" % len(train))
train = train.drop(index= train[train['fare_amount'] < 2.5].index, axis=0)
print("New size: %d" % len(train))


# - There are stil 6 records left where passanger count is greater than 7
# - we will delete such cases because we don't  have passanger count greater than 7 in test record

# In[ ]:


train[train.passenger_count >= 7]


# In[ ]:


print("old size: %d" % len(train))
train = train.drop(index= train[train.passenger_count >= 7].index, axis=0)
print("New size: %d" % len(train))


# In[ ]:


train.describe().T


# ### Lets check the distribution of distance in miles covered for train and test data set

# In[ ]:


#train data set
pd.cut(train['distance_miles'],np.linspace(0, 70, num = 8)).value_counts()


# In[ ]:


# test data set
pd.cut(test['distance_miles'],np.linspace(0, 70, num = 8)).value_counts()


# In[ ]:


# we will deal with it later 
fare_100 = train[train.fare_amount > 100]
fare_100.shape


# **There are 1669 records where fare amount is higher than 100 dollars**
#    - I also see than 527 records are such where distance covered  is even less than 1 miles
#    - While researching through few websites (https://www.introducingnewyork.com/taxis,  https://www.taxi-calculator.com/taxi-rate-new-york-city/259) i got to know that if taxi waits for passanger additional $30 dollars per hour will be charged depending on the location of the pickup.
#    - Since we don't have such variable where we can get to know the waiting time, we can drop such cases where distance is very less in copare to fare amount charged for the ride

# In[ ]:


fare_100[fare_100.distance_miles <1].shape


# In[ ]:


# #dropping cases where fare is above 100 dollars and distance is less than 1 miles
# print("old size: %d" % len(train))
# train = train.drop(index= train[(train.distance_miles <1) & (train.fare_amount > 100)].index, axis=0)
# print("New size: %d" % len(train))


# In[ ]:


train.columns


# In[ ]:


# create copy of the data set
df_train = train.drop(columns= ['key','pickup_datetime'], axis= 1).copy()
df_test = test.drop(columns= ['key','pickup_datetime'], axis= 1).copy()
print(df_train.shape)
print(df_test.shape)


# ### Train Test Split

# In[ ]:


from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(df_train.drop('fare_amount', axis=1),
                                                    df_train['fare_amount'], test_size=0.2, random_state = 42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# ## XG Boost Model

# In[ ]:


import xgboost as xgb


# In[ ]:


params = {
   
    'max_depth': 7,
    'gamma' :0,
    'eta':.03, 
    'subsample': 1,
    'colsample_bytree': 0.9, 
    'objective':'reg:linear',
    'eval_metric':'rmse',
    'silent': 0
}


# In[ ]:


def XGBmodel(X_train,X_test,y_train,y_test,params):
    matrix_train = xgb.DMatrix(X_train,label=y_train)
    matrix_test = xgb.DMatrix(X_test,label=y_test)
    model=xgb.train(params=params,
                    dtrain=matrix_train,num_boost_round=5000, 
                    early_stopping_rounds=10,evals=[(matrix_test,'test')])
    return model

model = XGBmodel(X_train,X_test,y_train,y_test,params)


# In[ ]:


prediction = model.predict(xgb.DMatrix(df_test), ntree_limit = model.best_ntree_limit).tolist()


# In[ ]:


test = pd.read_csv("../input/test.csv")
holdout = pd.DataFrame({'key': test['key'], 'fare_amount': prediction})
holdout.to_csv('xgb_4m_utc_with_cleaning.csv', index=False)


# In[ ]:


import matplotlib.pyplot as plt
fscores = pd.DataFrame({'X': list(model.get_fscore().keys()), 'Y': list(model.get_fscore().values())})
fscores.sort_values(by='Y').plot.bar(x='X')


# ## Stay Tuned,  Hyperparameter tuning coming soon
