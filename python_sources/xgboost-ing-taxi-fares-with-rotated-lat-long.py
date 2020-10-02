#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd
import datetime as dt
from sklearn.model_selection import train_test_split
import xgboost as xgb
import os

print(os.listdir("../input"))


# **Load dataset**
# 
# First we will load the train.csv dataset. Since this dataset has 55M rows, we will only use the first 1M to build our model to prevent memory issues and speed up preprocessing and model building.

# In[ ]:


train_df =  pd.read_csv('../input/train.csv', nrows = 1_000_000)
train_df.dtypes


# **Data exploration**
# 
# Now we will explore the loaded data to identify outliers and other problems that might need fixing such as null values.

# In[ ]:


#Identify null values
print(train_df.isnull().sum())


# We have a few rows with null values so it is safe to remove them.

# In[ ]:


#Drop rows with null values
train_df = train_df.dropna(how = 'any', axis = 'rows')


# Now let's explore the variables in the dataset. First we will look at the first rows to get an idea of the format of the values and then we will plot them to get a sense of their distribution and identify outliers.

# In[ ]:


#Look at the first rows
train_df.head()


# In[ ]:


#Plot variables using only 1000 rows for efficiency
train_df.iloc[:1000].plot.scatter('pickup_longitude', 'pickup_latitude')
train_df.iloc[:1000].plot.scatter('dropoff_longitude', 'dropoff_latitude')

#Get distribution of values
train_df.describe()


# Okay, that was interesting. We learned a few things about the dataset:
# - Fare_amount has negative values. We will remove those.
# - Latitudes and longitudes have values near 0 that cannot be correct since NYC is at (40,-74) aprox. We will remove points not near these coordinates.
# - Passenger_count has values of 0 and as high as 200, which are also unrealistic. We will remove those.
# 

# In[ ]:


#Clean dataset
def clean_df(df):
    return df[(df.fare_amount > 0) & 
            (df.pickup_longitude > -80) & (df.pickup_longitude < -70) &
            (df.pickup_latitude > 35) & (df.pickup_latitude < 45) &
            (df.dropoff_longitude > -80) & (df.dropoff_longitude < -70) &
            (df.dropoff_latitude > 35) & (df.dropoff_latitude < 45) &
            (df.passenger_count > 0) & (df.passenger_count < 10)]

train_df = clean_df(train_df)
print(len(train_df))


# **Feature engineering**
# 
# Now that we have cleaned some extreme values, we will add some interesting features in the dataset.
# - total_distance: distance from pickup to dropoff
# - Extract information from datetime (day of week, month, hour, day)

# In[ ]:


def sphere_dist(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):
    #Define earth radius (km)
    R_earth = 6371
    #Convert degrees to radians
    pickup_lat, pickup_lon, dropoff_lat, dropoff_lon = map(np.radians,
                                                             [pickup_lat, pickup_lon, 
                                                              dropoff_lat, dropoff_lon])
    #Compute distances along lat, lon dimensions
    dlat = dropoff_lat - pickup_lat
    dlon = dropoff_lon - pickup_lon
    
    #Compute haversine distance
    a = np.sin(dlat/2.0)**2 + np.cos(pickup_lat) * np.cos(dropoff_lat) * np.sin(dlon/2.0)**2
    
    return 2 * R_earth * np.arcsin(np.sqrt(a))

def add_datetime_info(dataset):
    #Convert to datetime format
    dataset['pickup_datetime'] = pd.to_datetime(dataset['pickup_datetime'])
    
    dataset['hour'] = dataset.pickup_datetime.dt.hour
    dataset['day'] = dataset.pickup_datetime.dt.day
    dataset['month'] = dataset.pickup_datetime.dt.month
    dataset['weekday'] = dataset.pickup_datetime.dt.weekday
    
    return dataset

train_df['distance'] = sphere_dist(train_df['pickup_latitude'], train_df['pickup_longitude'], 
                                   train_df['dropoff_latitude'] , train_df['dropoff_longitude'])

train_df = add_datetime_info(train_df)

train_df.head()


# Now we need to drop the columns that we will not use to train our model.
# - key
# - pickup_datetime

# In[ ]:


train_df.drop(columns=['key', 'pickup_datetime'], inplace=True)
train_df.head()


# ### Add rotational latitude and longitudes

# In[ ]:


#y' = y*cos(a) - x*sin(a)
#x' = y*sin(a) + x*cos(a)
train_df['pickup_long_15'] = train_df['pickup_longitude']*np.cos(15* np.pi / 180) - train_df['pickup_latitude']*np.sin(15* np.pi/180)
train_df['pickup_long_30'] = train_df['pickup_longitude']*np.cos(30* np.pi / 180) - train_df['pickup_latitude']*np.sin(30* np.pi/180)
train_df['pickup_long_45'] = train_df['pickup_longitude']*np.cos(45* np.pi / 180) - train_df['pickup_latitude']*np.sin(45* np.pi/180)
train_df['pickup_long_60'] = train_df['pickup_longitude']*np.cos(60* np.pi / 180) - train_df['pickup_latitude']*np.sin(60* np.pi/180)
train_df['pickup_long_75'] = train_df['pickup_longitude']*np.cos(75* np.pi / 180) - train_df['pickup_latitude']*np.sin(75* np.pi/180)

train_df['pickup_lat_15'] = train_df['pickup_longitude']*np.sin(15* np.pi / 180) + train_df['pickup_latitude']*np.cos(15* np.pi/180)
train_df['pickup_lat_30'] = train_df['pickup_longitude']*np.sin(30* np.pi / 180) + train_df['pickup_latitude']*np.cos(30* np.pi/180)
train_df['pickup_lat_45'] = train_df['pickup_longitude']*np.sin(45* np.pi / 180) + train_df['pickup_latitude']*np.cos(45* np.pi/180)
train_df['pickup_lat_60'] = train_df['pickup_longitude']*np.sin(60* np.pi / 180) + train_df['pickup_latitude']*np.cos(60* np.pi/180)
train_df['pickup_lat_75'] = train_df['pickup_longitude']*np.sin(75* np.pi / 180) + train_df['pickup_latitude']*np.cos(75* np.pi/180)

train_df['dropoff_long_15'] = train_df['dropoff_longitude']*np.cos(15* np.pi / 180) - train_df['dropoff_latitude']*np.sin(15* np.pi/180)
train_df['dropoff_long_30'] = train_df['dropoff_longitude']*np.cos(30* np.pi / 180) - train_df['dropoff_latitude']*np.sin(30* np.pi/180)
train_df['dropoff_long_45'] = train_df['dropoff_longitude']*np.cos(45* np.pi / 180) - train_df['dropoff_latitude']*np.sin(45* np.pi/180)
train_df['dropoff_long_60'] = train_df['dropoff_longitude']*np.cos(60* np.pi / 180) - train_df['dropoff_latitude']*np.sin(60* np.pi/180)
train_df['dropoff_long_75'] = train_df['dropoff_longitude']*np.cos(75* np.pi / 180) - train_df['dropoff_latitude']*np.sin(75* np.pi/180)

train_df['dropoff_lat_15'] = train_df['dropoff_longitude']*np.sin(15* np.pi / 180) + train_df['dropoff_latitude']*np.cos(15* np.pi/180)
train_df['dropoff_lat_30'] = train_df['dropoff_longitude']*np.sin(30* np.pi / 180) + train_df['dropoff_latitude']*np.cos(30* np.pi/180)
train_df['dropoff_lat_45'] = train_df['dropoff_longitude']*np.sin(45* np.pi / 180) + train_df['dropoff_latitude']*np.cos(45* np.pi/180)
train_df['dropoff_lat_60'] = train_df['dropoff_longitude']*np.sin(60* np.pi / 180) + train_df['dropoff_latitude']*np.cos(60* np.pi/180)
train_df['dropoff_lat_75'] = train_df['dropoff_longitude']*np.sin(75* np.pi / 180) + train_df['dropoff_latitude']*np.cos(75* np.pi/180)


# **Model training**
# 
# Now that we have the dataframe that we wanted we can start to train the XGBoost model. First we will split the dataset into train (80%)  and test (20%). 

# In[ ]:


y = train_df['fare_amount']
train = train_df.drop(columns=['fare_amount'])

x_train,x_test,y_train,y_test = train_test_split(train,y,random_state=0,test_size=0.2)


# In[ ]:


def XGBmodel(x_train,x_test,y_train,y_test):
    matrix_train = xgb.DMatrix(x_train,label=y_train)
    matrix_test = xgb.DMatrix(x_test,label=y_test)
    model=xgb.train(params={'objective':'reg:linear','eval_metric':'rmse'},
                    dtrain=matrix_train,num_boost_round=100, 
                    early_stopping_rounds=100,evals=[(matrix_test,'test')])
    return model

model = XGBmodel(x_train,x_test,y_train,y_test)


# **Prediction**
# 
# Finally we can use our trained model to predict the submission. First we will need to load and preprocess the test dataset just like we did for the training dataset.

# In[ ]:


#Read and preprocess test set
test_df =  pd.read_csv('../input/test.csv')
test_df['distance'] = sphere_dist(test_df['pickup_latitude'], test_df['pickup_longitude'], 
                                   test_df['dropoff_latitude'] , test_df['dropoff_longitude'])
test_df = add_datetime_info(test_df)
#y' = y*cos(a) - x*sin(a)
#x' = y*sin(a) + x*cos(a)
test_df['pickup_long_15'] = test_df['pickup_longitude']*np.cos(15* np.pi / 180) - test_df['pickup_latitude']*np.sin(15* np.pi/180)
test_df['pickup_long_30'] = test_df['pickup_longitude']*np.cos(30* np.pi / 180) - test_df['pickup_latitude']*np.sin(30* np.pi/180)
test_df['pickup_long_45'] = test_df['pickup_longitude']*np.cos(45* np.pi / 180) - test_df['pickup_latitude']*np.sin(45* np.pi/180)
test_df['pickup_long_60'] = test_df['pickup_longitude']*np.cos(60* np.pi / 180) - test_df['pickup_latitude']*np.sin(60* np.pi/180)
test_df['pickup_long_75'] = test_df['pickup_longitude']*np.cos(75* np.pi / 180) - test_df['pickup_latitude']*np.sin(75* np.pi/180)

test_df['pickup_lat_15'] = test_df['pickup_longitude']*np.sin(15* np.pi / 180) + test_df['pickup_latitude']*np.cos(15* np.pi/180)
test_df['pickup_lat_30'] = test_df['pickup_longitude']*np.sin(30* np.pi / 180) + test_df['pickup_latitude']*np.cos(30* np.pi/180)
test_df['pickup_lat_45'] = test_df['pickup_longitude']*np.sin(45* np.pi / 180) + test_df['pickup_latitude']*np.cos(45* np.pi/180)
test_df['pickup_lat_60'] = test_df['pickup_longitude']*np.sin(60* np.pi / 180) + test_df['pickup_latitude']*np.cos(60* np.pi/180)
test_df['pickup_lat_75'] = test_df['pickup_longitude']*np.sin(75* np.pi / 180) + test_df['pickup_latitude']*np.cos(75* np.pi/180)

test_df['dropoff_long_15'] = test_df['dropoff_longitude']*np.cos(15* np.pi / 180) - test_df['dropoff_latitude']*np.sin(15* np.pi/180)
test_df['dropoff_long_30'] = test_df['dropoff_longitude']*np.cos(30* np.pi / 180) - test_df['dropoff_latitude']*np.sin(30* np.pi/180)
test_df['dropoff_long_45'] = test_df['dropoff_longitude']*np.cos(45* np.pi / 180) - test_df['dropoff_latitude']*np.sin(45* np.pi/180)
test_df['dropoff_long_60'] = test_df['dropoff_longitude']*np.cos(60* np.pi / 180) - test_df['dropoff_latitude']*np.sin(60* np.pi/180)
test_df['dropoff_long_75'] = test_df['dropoff_longitude']*np.cos(75* np.pi / 180) - test_df['dropoff_latitude']*np.sin(75* np.pi/180)

test_df['dropoff_lat_15'] = test_df['dropoff_longitude']*np.sin(15* np.pi / 180) + test_df['dropoff_latitude']*np.cos(15* np.pi/180)
test_df['dropoff_lat_30'] = test_df['dropoff_longitude']*np.sin(30* np.pi / 180) + test_df['dropoff_latitude']*np.cos(30* np.pi/180)
test_df['dropoff_lat_45'] = test_df['dropoff_longitude']*np.sin(45* np.pi / 180) + test_df['dropoff_latitude']*np.cos(45* np.pi/180)
test_df['dropoff_lat_60'] = test_df['dropoff_longitude']*np.sin(60* np.pi / 180) + test_df['dropoff_latitude']*np.cos(60* np.pi/180)
test_df['dropoff_lat_75'] = test_df['dropoff_longitude']*np.sin(75* np.pi / 180) + test_df['dropoff_latitude']*np.cos(75* np.pi/180)

test_key = test_df['key']
x_pred = test_df.drop(columns=['key', 'pickup_datetime'])

#Predict from test set
prediction = model.predict(xgb.DMatrix(x_pred), ntree_limit = model.best_ntree_limit)


# In[ ]:


#Create submission file
submission = pd.DataFrame({
        "key": test_key,
        "fare_amount": prediction.round(2)
})

submission.to_csv('taxi_fare_submission.csv',index=False)
submission


# **Possible improvements**
# 
# - Right now, converting the 'pickup_datetime' column to datetime format is a real bottleneck. Try to find a way to better scale this part to be able to train with a larger number of traning samples.
# - Use cross-validation to tune the hyperparameters of the model for better performance.
