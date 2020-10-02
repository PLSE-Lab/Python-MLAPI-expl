#!/usr/bin/env python
# coding: utf-8

# Basic Features + Location Cells with Avg Location Price
# 
# My best LB score so far comes from a few basic features that were mentioned in these forums (# photos, # features, # description words, price per room) and then my idea of a relative price per room vs the location's avg price per room.
# 
# To build this concept I first rounded off the latitude and longitude to 2 decimal places.  Then combined them into a tuple of (xx.xx, xx.xx) which gave me about 450 unique locations.  I then calculated the mean price per room rate for each location.  Using this I got a ratio of each listing's price per room vs the avg price per room of the location.
# 
# I then encoded some features and ran a random forest.
# 
# This is my first kernel, please point out anything you feel I could have done better or why/why not this feature wouldn't work.

# In[ ]:


import numpy as np 
import pandas as pd

#Load Data
train_df = pd.read_json('../input/train.json')
test_df = pd.read_json('../input/test.json')


# In[ ]:


#Price vs Loc Avg
def add_price_vs_loc_avg(df):
    '''(DataFrame) -> DataFrame
 
    Will add 'PricePerRoomVsLocAvg' to the current DataFrame.
    '''
    #Build Location area
    df['lat_round'] = df.apply(lambda x : round(x['latitude'],2), axis=1)
    df['lon_round'] = df.apply(lambda x : round(x['longitude'],2), axis=1)
    df['loc'] = df.apply(lambda x : tuple([x['lat_round'], x['lon_round']]), axis=1)
    #Calculate Avg PricePerRoom for each Location
    df['AvgLocPricePerRoom'] = df.apply(lambda x: df['PricePerRoom'][df['loc']==x['loc']].mean(), axis=1)
    #Get Ratio
    df['PricePerRoomVsLocAvg'] = df['PricePerRoom'] / df['AvgLocPricePerRoom']
    return df


# In[ ]:


#Add Features
def add_features(df):
    '''(DataFrame) -> DataFrame
    
    Will add new features to the current DataFrame.
    '''
    #Create # of Photos Column
    df['NumPhotos'] = df.photos.str.len()
    #Create # of Features Column
    df['NumFeatures'] = df.features.str.len()
    df['NumDescription'] = df.description.str.len()
    #Total Rooms
    df['TotalRooms'] = df['bathrooms'] + df['bedrooms']
    #Room / Price
    #Add one too all -assume every apartment is at least 1 room (studios)
    #Is adding 1 a safe way to get around divide by 0?
    df['PricePerRoom'] = df['price'] / (df['TotalRooms'] + 1.0)
    df['PricePerBedRoom'] = df['price'] / (df['bedrooms'] + 1.0)
    #Add Price vs Loc
    df = add_price_vs_loc_avg(df)
    return df


# In[ ]:


#Add the features to data - this cell will take a bit of time to process
train_df = add_features(train_df)
test_df = add_features(test_df)


# In[ ]:


#Encode text features
from sklearn.preprocessing import LabelEncoder

#Combine the train and test columns
manager_combo = train_df['manager_id'].append(test_df['manager_id'])
building_combo = train_df['building_id'].append(test_df['building_id'])
loc_combo = train_df['loc'].append(test_df['loc'])
#Encode building_id
le_building = LabelEncoder()
le_building.fit(building_combo)
#Transform Train & Test set
train_df['BuildingID'] = le_building.transform(train_df['building_id'])
test_df['BuildingID'] = le_building.transform(test_df['building_id'])
#Encode manager_id
le_manager = LabelEncoder()
le_manager.fit(manager_combo)
#Transform Train & Test set
train_df['ManagerID'] = le_manager.transform(train_df['manager_id'])
test_df['ManagerID'] = le_manager.transform(test_df['manager_id'])
#Encode loc
le_loc = LabelEncoder()
le_loc.fit(loc_combo)
#Transform Train & Test set
train_df['LocID'] = le_loc.transform(train_df['loc'])
test_df['LocID'] = le_loc.transform(test_df['loc'])


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# In[ ]:


#Select features
feature_cols = ['price', 'PricePerRoom', 'PricePerRoomVsLocAvg', 'BuildingID', 'NumDescription', 'ManagerID', 'NumPhotos',
               'NumFeatures', 'latitude', 'longitude', 'bedrooms', 'bathrooms']

#Prepare data for ML
X_train = train_df[feature_cols].values
X_test = test_df[feature_cols].values

#Encode 'interest_level' to numerical
le_interest = LabelEncoder()
train_df['IL'] = le_interest.fit_transform(train_df['interest_level'])
#Set Train Y
Y = train_df['IL'].values


# In[ ]:


#Cross-Validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score

#Initialize Model
rf = RandomForestClassifier(n_estimators=1000, min_samples_split=20, criterion='entropy', n_jobs=-1)
#Create KFold
kfold = KFold(n_splits=5, random_state=5)
cross_val_results = cross_val_score(rf, X_train, Y, cv=kfold, scoring='neg_log_loss')
print(cross_val_results.mean())


# In[ ]:


#Train RandomForest on more estimators
#Submit


# The random forest parameters I found using grid search.   The features I selected here was a bit of trial and error looking at feature importance.  I don't show the trial and error here because I don't think it is necessary to.
# 
# My current best submission using these features and a random forest scored 0.59337 on LB. 
