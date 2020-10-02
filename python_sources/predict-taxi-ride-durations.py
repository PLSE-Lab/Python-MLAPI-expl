#!/usr/bin/env python
# coding: utf-8

# ## Predict Taxi Ride Durations

# The model's goal is to accurately predict the travel time of a taxi in New York. The dataset was too large for me to download, so the current dataset was shortened, consisting of the trips within the month of January 2016 that began and ended on Manhattan Island, New York. This dataset now consists of 82,800 different taxi ride trips.
# 
# The dataset includes:
# 
# * `pickup_datetime`: date and time when the meter was engaged
# * `dropoff_datetime`: date and time when the meter was disengaged
# * `pickup_lon`: the longitude where the meter was engaged
# * `pickup_lat`: the latitude where the meter was engaged
# * `dropoff_lon`: the longitude where the meter was disengaged
# * `dropoff_lat`: the latitude where the meter was disengaged
# * `passengers`: the number of passengers in the vehicle (driver entered value)
# * `distance`: trip distance
# * `duration`: duration of the trip in seconds
# 
# The goal is to redict `duration` from the other columns.

# ## Loading the Data

# In[ ]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


manhattan_taxi = pd.read_csv('../input/taxi-data-set/manhattan_taxi.csv')
manhattan_taxi.head(5)


# A scatter diagram of the Manhattan taxi rides. It closely resembles the shape of Manhattan Island. There is an empty white rectangle located where Central Park is because cars are not allowed there.

# In[ ]:


def pickup_scatter(t):
    plt.scatter(t['pickup_lon'], t['pickup_lat'], s=2, alpha=0.2)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Pickup locations')

plt.figure(figsize=(8, 16))
pickup_scatter(manhattan_taxi);


# ## Exploratory Data Analysis

# ### Choosing which days to include as training data in my regression model.

# Adding a column 'date' to the dataframe that contains the date of pickup without the time, formatted as a datetime.date value.

# In[ ]:


manhattan_taxi['date'] = pd.to_datetime(manhattan_taxi['pickup_datetime']).dt.date
manhattan_taxi.head()


# Checking for abnormal days.

# In[ ]:


date_ride_count = manhattan_taxi.groupby('date').count()
date_ride_count = pd.DataFrame(date_ride_count['pickup_datetime']).reset_index().rename(columns={'pickup_datetime': 'ride count'})
plt.plot_date(date_ride_count['date'], date_ride_count['ride count'])
plt.xticks(rotation=70)
plt.xlabel('date')
plt.ylabel('ride count')
plt.title('Ride Count by Date');


# After some research, the abnormality near 2016-01-23 is caused by an extreme blizzard on that date. This visualization allows me to identify which dates the blizzard affected in addiction to 2016-01-23.

# Removing atypical dates.

# In[ ]:


from datetime import date

atypical = [1, 2, 3, 18, 23, 24, 25, 26]
typical_dates = [date(2016, 1, n) for n in range(1, 32) if n not in atypical]
typical_dates

final_taxi = manhattan_taxi[manhattan_taxi['date'].isin(typical_dates)]


# ## Feature Engineering

# #### Creating a feature matrix for my linear regression model

# ### Train Test Split

# In[ ]:


import sklearn.model_selection

train, test = sklearn.model_selection.train_test_split(final_taxi, train_size=0.8, test_size=0.2, random_state=42)
print('Train:', train.shape, 'Test:', test.shape)


# A box plot that compares the distributions of taxi trip durations for each day using `train`.

# In[ ]:


train_sorted_dates = train.sort_values('date')
plt.figure(figsize=(8,4))
sns.boxplot(data=train_sorted_dates, x='date', y='duration')
plt.xticks(rotation=90)
plt.title('Duration by date');


# As especially shown by the Jan 10th, 16th, and 17th weekend dates, the medians are generally lower for weekends compared to weekdays. The upper quartiles of those three date show more distinguishably that the ride durations during weekends are shorter than weekdays. 

# **Using the pickup time to add new columns to the `train` and `test` dataframes. Five new columns are added:**
# 
# * `hour`: The integer hour of the pickup time. E.g., a 3:45pm taxi ride would have 15 as the hour. A 12:20am ride would have 0.
# * `day`: The day of the week with Monday=0, Sunday=6.
# * `weekend`: 1 if and only if the day is Saturday or Sunday.
# * `period`: 1 for early morning (12am-6am), 2 for daytime (6am-6pm), and 3 for night (6pm-12pm).
# * `speed`: Average speed in miles per hour.

# In[ ]:


def speed(t):
    """Return a column of speeds in miles per hour."""
    return t['distance'] / t['duration'] * 60 * 60

def augment(t):
    """Augment a dataframe t with additional columns."""
    u = t.copy()
    pickup_time = pd.to_datetime(t['pickup_datetime'])
    u.loc[:, 'hour'] = pickup_time.dt.hour
    u.loc[:, 'day'] = pickup_time.dt.weekday
    u.loc[:, 'weekend'] = (pickup_time.dt.weekday >= 5).astype(int)
    u.loc[:, 'period'] = np.digitize(pickup_time.dt.hour, [0, 6, 18])
    u.loc[:, 'speed'] = speed(t)
    return u
    
train = augment(train)
test = augment(test)
train.iloc[0,:]


# An overlaid histogram comparing the distributions of average speeds for taxi rides that start in the early morning (period 1), day (period 2), and night (period 3).

# In[ ]:


period_1 = train[train['period'] == 1]['speed']
period_2 = train[train['period'] == 2]['speed']
period_3 = train[train['period'] == 3]['speed']

plt.figure(figsize=(8,6))
sns.distplot(period_1, label='Early Morning')
sns.distplot(period_2, label='Daytime')
sns.distplot(period_3, label='Night')
plt.title('Distribution of Speed per Period');
plt.legend();


# Adding a `region` column to `train` that categorizes each pick-up location as 0, 1, or 2 based on the value of each point's first principal component, such that an equal number of points fall into each region.

# In[ ]:


D = train[['pickup_lon', 'pickup_lat']].to_numpy()
pca_n = len(train)
pca_means = np.mean(D, axis=0)
X = (D - pca_means) / np.sqrt(pca_n)
u, s, vt = np.linalg.svd(X, full_matrices=False)

def add_region(t):
    """Add a region column to t based on vt above."""
    D = t[['pickup_lon', 'pickup_lat']].to_numpy()
    assert D.shape[0] == t.shape[0], 'You set D using the incorrect table'
    X = (D - pca_means) / np.sqrt(pca_n) 
    first_pc = X @ vt.T[0]
    t.loc[:,'region'] = pd.qcut(first_pc, 3, labels=[0, 1, 2])
    
add_region(train)
add_region(test)


# In[ ]:


plt.figure(figsize=(8, 16))
for i in [0, 1, 2]:
    pickup_scatter(train[train['region'] == i])


# Creating a feature matrix with these features, coverting quantitative features to standard units and categorical features to dummy variables using one-hot encoding. `period` is not included because it is a linear combination of `hour`. `weekend` is not included because it is a linear combination of `day`. `speed` is not included because it was computed from `duration`.

# In[ ]:


from sklearn.preprocessing import StandardScaler

num_vars = ['pickup_lon', 'pickup_lat', 'dropoff_lon', 'dropoff_lat', 'distance']
cat_vars = ['hour', 'day', 'region']

scaler = StandardScaler()
scaler.fit(train[num_vars])

def design_matrix(t):
    """Create a design matrix from taxi ride dataframe t."""
    scaled = t[num_vars].copy()
    scaled.iloc[:,:] = scaler.transform(scaled) # Convert to standard units
    categoricals = [pd.get_dummies(t[s], prefix=s, drop_first=True) for s in cat_vars]
    return pd.concat([scaled] + categoricals, axis=1)

design_matrix(train).iloc[0,:]


# ## Model Selection

# #### Selecting a regression model to predict the duration of a taxi ride.

# Root mean squared error (RMSE) on the test set for a constant model that always predicts the mean duration of all training set taxi rides.

# In[ ]:


def rmse(errors):
    """Output: root mean squared error."""
    return np.sqrt(np.mean(errors ** 2))

constant_rmse = rmse(test['duration'] - np.mean(train['duration']))
constant_rmse


# RMSE on the test set for a simple linear regression model that uses only the distance of the taxi ride as a feature and includes an intercept.

# In[ ]:


from sklearn.linear_model import LinearRegression

simple_model = LinearRegression()
simple_model = simple_model.fit(train[['distance']], train.loc[:, 'duration'])

simple_rmse = rmse(test['duration'] - simple_model.predict(test[['distance']]))
simple_rmse


# RMSE on the test set for a linear regression model fitted to the training set without regularization, using the `design_matrix` defined by the previous design_matrix function.

# In[ ]:


linear_model = LinearRegression()
linear_model = linear_model.fit(design_matrix(train), train.loc[:, 'duration'])

linear_rmse = rmse(test['duration'] - linear_model.predict(design_matrix(test)))
linear_rmse


# RMSE on the test set for a model that first chooses linear regression parameters based on the observed period of the taxi ride, then predicts the duration using those parameters. An unregularized linear regression model is fit for each possible value of `period` in the training set. `design_matrix` was used again for features.

# In[ ]:


period_model = LinearRegression()
errors = []

for v in np.unique(train['period']):
    model = period_model.fit(design_matrix(train[train['period'] == v]), train[train['period'] == v].loc[:, 'duration'])
    errors = np.append(errors, test[test['period'] == v]['duration'] - period_model.predict(design_matrix(test[test['period'] == v])))

period_rmse = rmse(np.array(errors))
period_rmse


# The period model could possibly outperform the linear regression model because it focuses on less ranges of hours, so there are less covariates and a less complex model. With a less complex model, overfitting is less likely to be a problem.

# #### Instead of predicting duration directly, an alternative is to predict the average speed of the taxi ride using linear regression, then compute an estimate of the duration from the predicted speed and observed distance for each ride.

# RMSE in the duration predicted by a model that first predicts speed as a linear combination of features from `design_matrix`, fitted on the training set, then predicts duration from the predicted speed and observed distance.

# In[ ]:


speed_model = LinearRegression()
speed_model = model.fit(design_matrix(train), train['speed'])

# Speed in miles/hr. Duration is measured in seconds, and there are 3600 seconds in an hour.
avg = (test['distance'] * 3600) / (speed_model.predict(design_matrix(test)))

speed_rmse = rmse(test['duration'] - avg)
speed_rmse


# In[ ]:


speed_model.predict(design_matrix(train))


# In[ ]:


speed_model.score(design_matrix(train), train['duration'])


# Finding a different linear regression model for `period`, `region`, and `weekend`, fitting to `speed` on the training set and predicting `speed` on the test set, and finding the RMSE.

# In[ ]:


tree_speed_model = LinearRegression()
choices = ['period', 'region', 'weekend']

def duration_error(predictions, observations):
    """Error between predictions (array) and observations (data frame)"""
    return predictions - observations['duration']

def speed_error(predictions, observations):
    """Duration error between speed predictions and duration observations"""
    return duration_error(observations['distance'] * 3600 / predictions, observations)

def tree_regression_errors(outcome='duration', error_fn=duration_error):
    """Return errors for all examples in test using a tree regression model."""
    errors = []
    for vs in train.groupby(choices).size().index:
        v_train, v_test = train, test
        for v, c in zip(vs, choices):
            v_train = v_train[v_train[c]==v]
            v_test = v_test[v_test[c]==v]
            print(v_train.shape, v_test.shape)
        tree_speed_model.fit(design_matrix(v_train), v_train.loc[:, outcome])
        errors = np.append(errors, error_fn(tree_speed_model.predict(design_matrix(v_test)), v_test))
    return errors

errors = tree_regression_errors()
errors_via_speed = tree_regression_errors('speed', speed_error)
tree_rmse = rmse(np.array(errors))
tree_speed_rmse = rmse(np.array(errors_via_speed))
print('Duration:', tree_rmse, '\nSpeed:', tree_speed_rmse)


# A summary of the results:

# In[ ]:


models = ['constant', 'simple', 'linear', 'period', 'speed', 'tree', 'tree_speed']
pd.DataFrame.from_dict({
    'Model': models,
    'Test RMSE': [eval(m + '_rmse') for m in models]
}).set_index('Model').plot(kind='barh')
plt.xlabel('RMSE')
plt.title('RMSE by Model');


# ### From our summary bar plot, the last regression model is what should be used to predict taxi ride durations.
