#!/usr/bin/env python
# coding: utf-8

# # 1) Define Problem

# # New York Taxi Fare Prediction: 

#  our  tasked is  predicting the fare amount (inclusive of tolls) for a taxi ride in New York City given the pickup and dropoff locations. While we can get a basic estimate based on just the distance between the two points, this will result in an RMSE of $5-$8, depending on the model used . our  challenge is to do better than this using Machine Learning techniques!
# 
# 

# # 2) Specify input and output

# # Data Field:

# 1)**ID**
# key - Unique string identifying each row in both the training and test sets. Comprised of pickup_datetime plus a unique integer, but this doesn't matter, it should just be used as a unique ID field. Required in your submission CSV. Not necessarily needed in the training set, but could be useful to simulate a 'submission file' while doing cross-validation within the training set.
# 

# # Features
# 

# **pickup_datetime** - timestamp value indicating when the taxi ride started.
# 

# **pickup_longitude** - float for longitude coordinate of where the taxi ride started.
# 

# **pickup_latitude** - float for latitude coordinate of where the taxi ride started.
# 

# **dropoff_longitude** - float for longitude coordinate of where the taxi ride ended.
# 

# **dropoff_latitude** - float for latitude coordinate of where the taxi ride ended.
# 

# **passenger_count** - integer indicating the number of passengers in the taxi ride.
# 

# # Target
# 

# **fare_amount** - float dollar amount of the cost of the taxi ride. This value is only in the training set; this is what you are predicting in the test set and it is required in your submission CSV.

# # 3) Select Framework(libraries)

# In[ ]:


import os
import numpy as np#linear algebra   
import pandas as pd #data preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train =  pd.read_csv('../input/train.csv', nrows = 100000, parse_dates=["pickup_datetime"])  # 55m rows,but we import 10m rows


# In[ ]:


test = pd.read_csv('../input/test.csv')   #10k rows 


# In[ ]:


train.head()  # first 5 record of train 


# # 4) EDA(Exploratery Data Analysis)

# #  Data collection

# In[ ]:


train.describe() 


# In[ ]:


train.columns


# In[ ]:


train.info()


# train  has total 8 column in that   5 float64 values, 1 int value , 1 object ,and 1  datetime64 . 

# # Data Preprocessing & Data cleaning

# In[ ]:


print(train.isnull().sum())  # check anu null value is available or not .


# In[ ]:


print('Old size: %d' % len(train))
train = train.dropna(how = 'any', axis = 'rows')
print('New size: %d' % len(train))
# if gives 20million data then NaN values comes.


# In[ ]:


print(train.isnull().sum())


# In[ ]:


sns.distplot(train['fare_amount']);


# in between 0-50 there are 95% 'fare_amount' located.

# In[ ]:


train.loc[train['fare_amount']<0].shape


# There are 9 records with negative fare, we will remove these record from the data.

# there are lots of cases where lat and longitude is 0 , check how many such cases are?

# In[ ]:


train[(train.pickup_latitude==0) | (train.pickup_longitude)==0 | (train.dropoff_latitude==0) | (train.dropoff_longitude==0)].shape


# 1918 values are** 0 in train.
# Based on just look at the data, we can see that its not 100% clean and
# some entries will contribute to higher error rates. 

# In[ ]:


sns.distplot(train['passenger_count'])


# In[ ]:


train.describe()


# In[ ]:


#clean up the train dataset to eliminate out of range values
train = train[train['fare_amount'] > 0]
train = train[train['pickup_longitude'] < -72]
train = train[(train['pickup_latitude'] > 40) &(train
                                               ['pickup_latitude'] < 44)]
train = train[train['dropoff_longitude'] < -72]
train = train[(train['dropoff_latitude'] >40) & (train
                                                ['dropoff_latitude'] < 44)]
train = train[(train['passenger_count']>0) &(train['passenger_count'] < 10)]


# Now we can see there are no obvious inconstitencies with the data.

# In[ ]:


train.describe()


# #  Same operation perform on 'test'

# In[ ]:


test.head()  # first 5 record of test 


# In[ ]:


test.describe()


# In[ ]:


test.info()


# In[ ]:


print(test.isnull().sum())


# In[ ]:


test[(test.pickup_latitude==0) | (test.pickup_longitude)==0 | (test.dropoff_latitude==0) | (test.dropoff_longitude==0)].shape


# In[ ]:


print(test.isnull().sum())


# In[ ]:


print('Old size: %d' % len(test))
test = test.dropna(how = 'any', axis = 'rows')
print('New size: %d' % len(test))


# In[ ]:


#clean up the train dataset to eliminate out of range values
test = test[test['pickup_longitude'] < -72]
test = test[(test['pickup_latitude'] > 40) &(train
                                               ['pickup_latitude'] < 44)]
test = test[test['dropoff_longitude'] < -72]
test = test[(test['dropoff_latitude'] >40) & (train
                                                ['dropoff_latitude'] < 44)]
test = test[(test['passenger_count']>0) &(train['passenger_count'] < 10)]
train.head()


# we clean the dataset.

# #  Transforming Feature

# In[ ]:


#pickup_datetime 

train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'])
type(train['pickup_datetime'].iloc[0])


# conver Datetime var into single column as year, month,day_of_week, and hour 

# In[ ]:


combine = [test, train]
for dataset in combine:
        # Features: hour of day (night vs day), month (some months may be in higher demand) 
    dataset['pickup_datetime'] = pd.to_datetime(dataset['pickup_datetime'])
    dataset['hour_of_day'] = dataset.pickup_datetime.dt.hour
    dataset['day'] = dataset.pickup_datetime.dt.day
    dataset['week'] = dataset.pickup_datetime.dt.week
    dataset['month'] = dataset.pickup_datetime.dt.month
    dataset['day_of_year'] = dataset.pickup_datetime.dt.dayofyear
    dataset['week_of_year'] = dataset.pickup_datetime.dt.weekofyear

    
#dataset['Year'] = dataset['pickup_datetime'].apply(lambda time: time.year)
#dataset['Month'] = dataset['pickup_datetime'].apply(lambda time: time.month)
#ataset['Day of Week'] = dataset['pickup_datetime'].apply(lambda time: time.dayofweek)
#dataset['Hour'] = dataset['pickup_datetime'].apply(lambda time: time.hour)

train.head()


# In[ ]:


test.head()


# In[ ]:


# Given a dataframe, add two new features 'abs_diff_longitude' and
# 'abs_diff_latitude' reprensenting the "Manhattan vector" from
# the pickup location to the dropoff location.
def add_travel_vector_features(df):
    df['abs_diff_longitude'] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df['abs_diff_latitude'] = (df.dropoff_latitude - df.pickup_latitude).abs()

add_travel_vector_features(train) 
train.head(1)


# In[ ]:


# Given a dataframe, add two new features 'abs_diff_longitude' and
# 'abs_diff_latitude' reprensenting the "Manhattan vector" from
# the pickup location to the dropoff location.
def add_travel_vector_features(df):
    df['abs_diff_longitude'] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df['abs_diff_latitude'] = (df.dropoff_latitude - df.pickup_latitude).abs()

add_travel_vector_features(test) 
test.head(1)


# In[ ]:


# remove unnessary column that not requred for modeling.
train = train.drop(['key','pickup_datetime'],axis = 1) 
test = test.drop('pickup_datetime',axis = 1)
#train.info()


# #  Feature encoding

# In[ ]:


x_train = train.drop(['fare_amount'], axis=1)
y_train = train['fare_amount']
x_test = test.drop('key', axis=1)


# #  5) Model Design

# #  i)Linear Regression

# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score


# In[ ]:


linmodel = LinearRegression()
linmodel.fit(x_train, y_train)


# In[ ]:


linmodel_pred = linmodel.predict(x_test)  # prediction on train 


# # ii)Random Forest Regressor
# 

# In[ ]:


# Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()
rfr.fit(x_train, y_train)
rfr_pred = rfr.predict(x_test)


# references
# https://www.kaggle.com/danpavlov/ny-taxi-fare-comprehensive-and-simple-analysis

# # iii) XG-BOOST Model
# 

# Now that we have the dataFrame that we wanted we can start to train the XGBoost model. First we will split the dataset into train(99%) and test(1%). with this amont of data 1% should be enough to test performance.

# In[ ]:


from sklearn.model_selection import train_test_split
import xgboost as xgb

#Let's prepare the test set
x_pred = test.drop('key', axis=1)


# In[ ]:


#feature selection
y = train['fare_amount']    
train_df = train.drop(['fare_amount'],axis = 1)


# In[ ]:



# Let's run XGBoost and predict those fares
x_train,x_test,y_train,y_test = train_test_split(train_df,y,random_state=123,test_size=0.2)


# # Parameter Tunning(selecting best parameter for model)
# 

# In[ ]:


params = {
      #parameters that we are going to tune
    'max_depth' :8 ,#result of tuning with cv
    'eta' :.03, #result of tuning with cv
    'subsample' : 1, # result of tuning with cv
    'colsample_bytree' : 0.8, #result of tuning with cv
    #other parameter
    'objective': 'reg:linear',
    'eval_metrics':'rmse',
    'silent': 1
}


# In[ ]:


#Block of code used for hypertuning parameters. Adapt to each round of parameter tuning.
CV=False
if CV:
    dtrain = xgb.DMatrix(train,label=y)
    gridsearch_params = [
        (eta)
        for eta in np.arange(.04, 0.12, .02)
    ]

    # Define initial best params and RMSE
    min_rmse = float("Inf")
    best_params = None
    for (eta) in gridsearch_params:
        print("CV with eta={} ".format(
                                 eta))

        # Update our parameters
        params['eta'] = eta

        # Run CV
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=1000,
            nfold=3,
            metrics={'rmse'},
            early_stopping_rounds=10
        )

        # Update best RMSE
        mean_rmse = cv_results['test-rmse-mean'].min()
        boost_rounds = cv_results['test-rmse-mean'].argmin()
        print("\tRMSE {} for {} rounds".format(mean_rmse, boost_rounds))
        if mean_rmse < min_rmse:
            min_rmse = mean_rmse
            best_params = (eta)

    print("Best params: {}, RMSE: {}".format(best_params, min_rmse))
else:
    #Print final params to use for the model
    params['silent'] = 0 #Turn on output
    print(params)


# In[ ]:



def XGBmodel(x_train,x_test,y_train,y_test):
    matrix_train = xgb.DMatrix(x_train,label=y_train)
    matrix_test = xgb.DMatrix(x_test,label=y_test)
    model=xgb.train(params=params
                                  ,dtrain=matrix_train,num_boost_round=200, 
                    early_stopping_rounds=20,evals=[(matrix_test,'test')],)
    return model

model=XGBmodel(x_train,x_test,y_train,y_test)
xgb_pred = model.predict(xgb.DMatrix(x_pred), ntree_limit = model.best_ntree_limit)


# In[ ]:


linmodel_pred, rfr_pred, xgb_pred


# In[ ]:


# Assigning weights. More precise models gets higher weight.
linmodel_weight = 1
rfr_weight = 1
xgb_weight = 3
prediction = (linmodel_pred * linmodel_weight + rfr_pred * rfr_weight + xgb_pred * xgb_weight) / (linmodel_weight + rfr_weight + xgb_weight)


# In[ ]:


prediction


# # 6)Submission
# 

# In[ ]:


# Add to submission
submission = pd.DataFrame({
        "key": test['key'],
        "fare_amount": prediction.round(2)
})

submission.to_csv('sub_fare.csv',index=False)


# In[ ]:


submission


# # 7)Conclusion

# i have tried all the parts related to the proccess of machin learning with a variety of python package and i know there are still some problem then i hope to get your feedback to improve it.
