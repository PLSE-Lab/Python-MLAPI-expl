#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv('../input/train.csv', nrows = 1000000)
# out of the 55million + data, i have picked a sample and will be working with it

test = pd.read_csv('../input/test.csv')

train.head()


# In[ ]:


test.head()


# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


#We check the datatypes 
train.dtypes.value_counts()


# In[ ]:


test.dtypes.value_counts()


# As we can see from above, we have 2 columns with the object datatype. We will have to take a closer look at them later. 
# First we take a look at the data to see if there are any null or NAN values
# 

# In[ ]:


train.isnull().sum()


# In[ ]:


#I will simply drop the nan columns
train = train.dropna()
train.isnull().sum()


# In[ ]:


#The we check for zeros in our train and test data
(train ==0).astype(int).sum()


#  we are simply going to drop the zero rows becasue they may result in wrong outpur down the road'

# In[ ]:


train = train.loc[~(train==0).any(axis =1)]
#we take a look at what we have done
#(train ==0).astype(int).sum()


# In[ ]:


(train ==0).astype(int).sum()


# In[ ]:


#we started out with 1 million data point. Lets see what we have now.
train.shape


# 
# 
# WHAT we do next is to see how the data points are distributed. By using the the .describe method, we check the min and the max value of the dataset.
# 
# 

# In[ ]:


train.describe()


# In[ ]:





# waouw!!! checkout the difference/ range between the min and the max . it is ver large. This means we could have a lot of outlier in out dataset, this may lead to wrong out;puts

# Lets find the absolut distance travelled by each passenger, by taking the difference between their pickup latitudes and dropoff latitudes. We do the same for latitude

# In[ ]:





# In[ ]:


train.describe()


# In[ ]:


train.dtypes.value_counts()


# In[ ]:


#lets take care of the object data first
object_data = train.dtypes == np.object
categoricals = train.columns[object_data]
categoricals


# In[ ]:


#I will drop the key column since i do not really need it 
train.drop('key', axis = 1, inplace = True)
train.head()


# Now we take care of the pickup_datetime column, we extract the dates separately and drop the pickup_datetime column
# 

# In[ ]:


import datetime as dt

def date_extraction(data):
    data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'])
    data['year'] = data['pickup_datetime'].dt.year
    data['month'] = data['pickup_datetime'].dt.month
    data['weekday'] = data['pickup_datetime'].dt.day
    data['hour'] = data['pickup_datetime'].dt.hour
    data = data.drop('pickup_datetime', axis = 1, inplace = True)
    
    return data
    
#Apply this to both the train and the test data
date_extraction(train)
#test = date_extraction(test)
    


# In[ ]:


train.head()


# In[ ]:


date_extraction(test)
test.head()


# Found this on Quora and i am going to use this to find the distance travelled using the longitudes and latitudes given. You can check it out at:
# 
# https://www.quora.com/How-to-measure-the-distance-traveled-using-latitude-and-longitude
# https://community.esri.com/groups/coordinate-reference-systems/blog/2017/10/05/haversine-formula

# In[ ]:




#First i will define the Haversine function
#radii of earth in meters = 6371e3 meters
# def long_lat_distance (x):
#     x['Longitude_distance'] = x['pickup_longitude'] - x['dropoff_longitude']
#     x['Latitude_distance'] = x['pickup_latitude'] - x['dropoff_latitude'] 
    
#     return x   


# In[ ]:


def long_lat_distance (x):
    x['Longitude_distance'] = np.radians(x['pickup_longitude'] - x['dropoff_longitude'])
    x['Latitude_distance'] = np.radians(x['pickup_latitude'] - x['dropoff_latitude']) 
    x['distance_travelled/10e3'] = ((x['Longitude_distance']**2 + x['Latitude_distance']**2)**0.5) *1000
    return x   


# In[ ]:


for x in [train, test]:
    long_lat_distance(x)
    
train.head()


# 

# In[ ]:


def harvesine(x):
    #radii of earth in meters = 
    r = 6371000 
    d = x['distance_travelled/10e3']
    theta_1 = np.radians(x['dropoff_latitude'])
    theta_2 = np.radians(x['pickup_latitude'])
    lambda_1 = np.radians(x['dropoff_longitude'])
    lambda_2 = np.radians(x['dropoff_longitude'])
    theta_diff = x['Longitude_distance']
    lambda_diff = x['Latitude_distance']
    
    a = np.sin(theta_diff/2)**2 + np.cos(theta_1)*np.cos(theta_2)*np.sin(lambda_diff/2)**2
    c = 2 * np.arctan2(a**0.5, (1-a)**0.5)
    x['harvesine/km'] = (r * c)/1000


# In[ ]:


for x in [train, test]:
    harvesine(x)
    
train.head()


# In[ ]:


train.dtypes.value_counts()


# Now that we do not have any categorical features left, we are going to use the standard scaler to mitigate the effect ot the outliers on our data set

# In[ ]:


# #not sure if this is rrally necessary. will have to see

# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# scaler.fit_transform(train)


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train.describe()


# In[ ]:


print('Are there any nulls\nan in the train data: ')
print(train.isnull().sum())

print('\nAre there any nulls\nans in the test data: ')
print(test.isnull().sum())


# In[ ]:


#as we can see there are 3 nulls in the train data. Lets replace them with the mean
train['harvesine/km'] = train['harvesine/km'].fillna(train['harvesine/km'].median())


# ## FEATURE SELECTION  and  RANDOM FOREST

# In[ ]:


from sklearn.ensemble import RandomForestRegressor

#split the train features 
feature_cols = [x for x in train.columns if x!= 'fare_amount']
X = train[feature_cols]
y = train['fare_amount']


# In[ ]:


correlations = X.corrwith(y)
correlations = abs(correlations*100)
correlations.sort_values(ascending = False, inplace= True)

correlations


# In[ ]:


#lets plot and see what we've got
ax = correlations.plot(kind='bar')
ax.set(ylim=[-1, 1], ylabel='pearson correlation');


# In[ ]:


train.head()


# In[ ]:





# In[ ]:


#From the diagram above, i will use the 5 most important features

train_1 = train.drop(['pickup_longitude', 'dropoff_longitude','pickup_latitude','dropoff_latitude',
                    'Longitude_distance', 'Latitude_distance'], axis =1)

train_1.head()


# In[ ]:


train_1['harvesine/km'] =train_1['harvesine/km'].round(2) 
train_1['distance_travelled/10e3'] =train_1['distance_travelled/10e3'].round(2) 

train_1.head()


# In[ ]:



# #not sure if this is rrally necessary. will have to see

# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# scaler.fit_transform(train_1)


# In[ ]:


train_1.describe()


# In[ ]:





# In[ ]:


test_1 = test.drop(['pickup_longitude', 'dropoff_longitude','pickup_latitude','dropoff_latitude',
                    'Longitude_distance', 'Latitude_distance'], axis =1)

test_1.head()


# ## Modelling and Prediction

# 

# In[ ]:


from sklearn.model_selection import train_test_split
feat_cols = [x for x in train_1.columns if x!= 'fare_amount']
X_1 = train_1[feat_cols]
y_1 = train_1['fare_amount']
X_train, X_test, y_train, y_test = train_test_split(X_1, y_1, test_size = 0.25, random_state = 42)


# In[ ]:


#Random forest
rf = RandomForestRegressor(n_estimators = 100, max_features = 5)
rf = rf.fit(X_train, y_train)


# With the previous version 14, i used Linear regression and i got a score of 9.38 but with random forest i got 3.89

# In[ ]:


test.head()


# ## Preparing submission file
# 

# In[ ]:


final_prediction = rf.predict(X_test)


# In[ ]:


test_1.drop('key', axis = 1, inplace = True)
test_1.head()


# In[ ]:


#random forest
final_prediction = rf.predict(test_1)

NYCtaxiFare_submission = pd.DataFrame({'key': test.key, 'fare_amount': final_prediction})
NYCtaxiFare_submission.to_csv('NYCtaxiFare_prediction.csv', index = False)


# In[ ]:


NYCtaxiFare_submission.head()


# In[ ]:





# In[ ]:




