#!/usr/bin/env python
# coding: utf-8

# This is going to be a basic exploratory analysis of the dataset only using matplotlib for beginners to follow. We will clean the dataset of outliers and explore the dataset a little bit.

# In[1]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from geopy.distance import vincenty
#get the distance 'as the crow flies" between 2 points
#we could try getting the route distance and maybe even a bit of traffic information, but that can be done later.


# In[2]:


#read the csv file
train = pd.read_csv('../input/train.csv')


# In[ ]:


train.info()
train.head(4)


# Drop the columns we don't want. The store_and_fwd_flag will not effect the ride characteristics, nor do we need the dropoff datetime.

# In[3]:


train.drop('store_and_fwd_flag',axis=1,inplace = True)
train.drop('dropoff_datetime',axis=1,inplace=True)


# Now we want to add a column for the distance between 2 points. We want to see the distribution of ride distances and remove those outliers.

# In[4]:


def distance(start_long,start_lat,stop_long,stop_lat):
    start = (start_long,start_lat)
    stop = (stop_long,stop_lat)
    return vincenty(start,stop).miles


# In[5]:


#this is going to take a while
train['distance'] = train.apply(lambda row: distance(row['pickup_longitude'], row['pickup_latitude'], row['dropoff_longitude'], row['dropoff_latitude']), axis=1)


# In[ ]:


#check that it worked
train.head(4)


# In[ ]:


distance = list(train['distance'])
distance.sort()
plt.scatter(range(len(distance)),distance)
plt.show()


# How can a ride in New York City be shorter than 1000 ft? I don't know much about NYC, but we'll remove those. We should also remove the rides longer than 15 miles as that seems to be the cutoff point.

# In[6]:


train = train.loc[train['distance'] < 15.0]
train = train.loc[train['distance'] > 0.2]


# Now let's remove other outliers based on the ride duration. Let's switch to log scale.

# In[ ]:


duration = list(train['trip_duration'])
duration.sort()
plt.scatter(range(len(duration)),duration)
ax = plt.gca()
ax.set_yscale('log')
plt.show()


# There seem to be a number of rides that last less than a few minutes. We definitely want to get rid of those as there is not a case imaginable that a complete taxi ride in New York would ever last less than 5 minutes.

# In[7]:


train = train.loc[train['trip_duration'] > 300]


# I would like to figure out some details for the rides that last longer than 10^4 seconds (almost 3 hours)

# In[ ]:


outlier = train.loc[train['trip_duration'] > 10000]


# In[ ]:


plt.scatter(outlier['trip_duration'],outlier['distance'])
plt.xlabel('trip_duration')
plt.ylabel('distance')
plt.show()


# These numbers are all over the place. Within 3 hours a taxi can travel anywhere from 0 to 17 miles. This data is not very trustworthy to fit a model. Let's see where the pickup and dropoff locations are.

# In[ ]:


fig = plt.figure()
fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True)
ax[0].scatter(outlier['pickup_longitude'].values, outlier['pickup_latitude'].values,
              color='green', s=1, label='pickup', alpha=0.1)
ax[1].scatter(outlier['dropoff_longitude'].values, outlier['dropoff_latitude'].values,
              color='red', s=1, label='dropoff', alpha=0.1)
ax[0].set_ylabel('latitude')
ax[0].set_xlabel('longitude')
ax[1].set_xlabel('longitude')
plt.ylim([40.6,40.9])
plt.xlim([-74.1,-73.8])
plt.show()


# The pickup and dropoff locations are all within a reasonable distance. Most seem to be happening within Manhattan. Does it take close to 3 hours to go anywhere in manhattan?
# Note: there are other rides outside the area of this plot but those should also be considered outliers in the data as the distance does not warrant a 3 hour drive.

# In[8]:


train = train.loc[train['trip_duration'] < 10000]


# Now let's look at the relationship between distance and ride duration.

# In[ ]:


plt.scatter(train['distance'],train['trip_duration'])
axis = plt.gca()
axis.set_xlabel('distance')
axis.set_ylabel('trip_duration')
plt.show()


# This is going to be hard to detect some outliers with. We can however see that the low distance/high duration rides are questionable. Moving less than 4 miles over 6000 seconds (~1.7 hours) seems very doubtful. I'll also clean up some of the long rides less than 8 miles.

# In[9]:


train = train.loc[~((train['distance'] < 4.0) & (train['trip_duration'] > 6000))]
train = train.loc[~((train['distance'] < 8.0) & (train['trip_duration'] > 8000))]


# In[ ]:


plt.scatter(train['distance'],train['trip_duration'])
axis = plt.gca()
axis.set_xlabel('distance')
axis.set_ylabel('trip_duration')
plt.show()


# That seems better. I do not want to mess with the rides longer than 8 miles until I start to validate a model. Let's also visualize with something we are more familiar with, average speed.

# In[10]:


train['trip_duration_hours'] = train['trip_duration']/3600.0
train['average_speed'] = train['distance']/train['trip_duration_hours']


# In[ ]:


plt.scatter(train['distance'],train['average_speed'])
axis = plt.gca()
axis.set_xlabel('distance')
axis.set_ylabel('average_speed')
plt.show()


# There are some really reckless taxi drivers around. Let's set the max speed threshold at 55 mph. The linear boundary is also curious. I assume it's due to our cleaning of the dataset. 

# In[11]:


train = train.loc[train['average_speed'] < 55.0]


# In[ ]:


plt.scatter(train['trip_duration_hours'],train['average_speed'])
axis = plt.gca()
axis.set_xlabel('trip_duration_hour')
axis.set_ylabel('average_speed')
plt.show()


# We have some rides with very low speeds. Where are these people going while traveling at less than 10 mph for more than an hour? Could be traffic. I don't think that we should try to do anything with that part yet as the regression model we pick should be able to differentiate when we add features to tell the time of day.

# In[ ]:


train.info()


# Overview of what was removed:
# 
# average speed > 55 mph
# 
# trip duration > 10000 seconds (2.8 hours)
# 
# trip duration < 5 minutes
# 
# distance > 15 miles
# 
# distance < 0.2 miles
# 
# removed low distance and high duration rides.
# 
# We've cleaned about 18% of the data from the original dataset. Everything that would have led to a model being thrown off has been removed. Further analysis of some of the rides is still required but I would save that until you select a model and start to validate it. A fun exercise would be to try and detect fraudulent rides from this dataset, but I will leave it at this. If anyone wants to do further analysis then please comment or add a link to your kernel as I would be interested in your insight.

# Let's add features to tell the time.

# In[12]:


train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'])
train['pickup_hour'] = train['pickup_datetime'].dt.hour
train['pickup_weekday'] = train['pickup_datetime'].dt.weekday
train['pickup_month'] = train['pickup_datetime'].dt.month


# In[ ]:


train.head(4)


# In[13]:


prediction_var = list(train.columns)
remove = ['id','pickup_datetime','trip_duration','distance','trip_duration_hours','average_speed']
for i in prediction_var[:]:
    if i in remove:
        prediction_var.remove(i)

prediction_var


# Let's try fitting some models to the data now.

# In[14]:


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor


# In[15]:


dummy_train , dummy_test = train_test_split(train,test_size = 0.2)


# In[ ]:


#let's start with the KNeighbors first
knn = KNeighborsRegressor(n_jobs=-1)


# In[ ]:


knn.fit(dummy_train[prediction_var],dummy_train['trip_duration'])


# In[ ]:


knn.predict(dummy_test[prediction_var])


# In[ ]:


knn.score(dummy_test[prediction_var],dummy_test['trip_duration'])


# Let's see what the RandomForest regressor gives.

# In[18]:


rf = RandomForestRegressor(n_jobs=-1)


# In[19]:


rf.fit(dummy_train[prediction_var],dummy_train['trip_duration'])


# In[20]:


rf.predict(dummy_test[prediction_var])


# In[21]:


rf.score(dummy_test[prediction_var],dummy_test['trip_duration'])


# Definitely better than the KNN regressor. For fun, you can try a neural net regression. The code for that would be:
# 
#     from sklearn.neural_network import MLPRegressor
#     net = MLPRegressor()
#     net.fit(dummy_train[prediction_var],dummy_train['trip_duration'])
#     net.predict(dummy_test[prediction_var])
#     net.score(dummy_test[prediction_var],dummy_test['trip_duration']
# 
# I am not going to run it in this kernel as it would exceed the runetime of 1200 seconds. I will try it on my machine or aws and see how it turned out. For now we should continue with tuning the models. We will use grid search to estimate the best parameters for our models.

# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


def gridsearch(model,param_grid,data_x,data_y):
    clf = GridSearchCV(model,param_grid,cv=10,n_jobs=-1)
    clf.fit(data_x,data_y)
    print(clf.best_params_)
    print(clf.best_score_)
    return clf.best_estimator_


# In[ ]:


knn_param_grid = [{'n_neighbors':[5,10,15,20],'algorithm':['brute'],'weights':['uniform','distance']},
                  {'n_neighbors':[5,10,15,20],'algorithm':['kd_tree'],'weights':['uniform','distance'],'leaf_size':[15,30,45,60]},
                  {'n_neighbors':[5,10,15,20],'algorithm':['ball_tree'],'weights':['uniform','distance'],'leaf_size':[15,30,45,60]}]
rf_param_grid = {'n_estimators':[10,50,100,200],'min_samples_leaf':[10,25,50]}


# This is going to exceed the allowed runtime so I will do this on my own. If you wish to do this on yourown, then I recommend running this on a high memory vm instance. I ran this on google cloud with 4cpu and 15gb memory. You will definitely need more than 8gb. My parameters for random forest were min_samples_leaf = 10 and n_estimators = 50.
# 
# EDIT: I keep getting memory errors everytime I run the predict() function for a knn regressor with different n_neighbors. Not sure why. I will look into it.

# In[16]:


rf = RandomForestRegressor(min_samples_leaf=10,n_estimators=50,n_jobs=-1)
rf.fit(dummy_train[prediction_var],dummy_train['trip_duration'])


# In[17]:


rf.predict(dummy_test[prediction_var])
rf.score(dummy_test[prediction_var],dummy_test['trip_duration'])

