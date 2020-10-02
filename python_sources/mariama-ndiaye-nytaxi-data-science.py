#!/usr/bin/env python
# coding: utf-8

# # New York City Taxi Trip Duration

# ## Data Loading and Exploration

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

train = pd.read_csv("../input/train.csv")
testfile = pd.read_csv("../input/test.csv")
train.head(6)


# In[ ]:


train.info()
correlations_data = train.corr()['trip_duration'].sort_values()
correlations_data


# ## Data Cleaning

# We have 1458644 rows for each column so there are no missing values.

# In[ ]:


train.describe()


# We clearly see `trip_duration` takes strange values for `min` and `max`. Let's have a quick visualisation with a boxplot.

# In[ ]:


plt.subplots(figsize=(18,6))
plt.title("Visualisation des outliers")
train.boxplot();


# There are outliers for `trip_duration`. I can't find a proper interpretation and it will probably damage our model, so I choose to get rid of them. We will only keep what I assume to be legit trips, i.e. between 100 and 10000 seconds.

# In[ ]:


#We only keep rows with a trip_duration between 100 and 10000 seconds.
train = train[(train.trip_duration < 10000) & (train.trip_duration > 100)]


# ## Features engineering (selection & transformations)

# First of all, let's select the features we need to make our predicting model.  
# `id` is unique and linked to a specific trip so there's not point in keeping it in our model. `store_and_fwd_flag` is not relevant to make predictions as I assume that vehicle memories work well and that it don't change the trip duration.

# In[ ]:


#Removing 'id' and store_and_fwd_flag' columns
train.drop(['id'], axis=1, inplace=True)
train.drop(['store_and_fwd_flag'], axis=1, inplace=True)
testfile.drop(['store_and_fwd_flag'], axis=1, inplace=True)


# #### Deal with dates

# In[ ]:


#Datetyping the dates so we can work with it
train['pickup_datetime'] = pd.to_datetime(train.pickup_datetime)
testfile['pickup_datetime'] = pd.to_datetime(testfile.pickup_datetime)
train.info()


# Now that we can work with our dates, we are going to create relevant date features for our model. We'll store the week, the weekday and the hour in our dataframe.

# In[ ]:


#Date features creations and deletions
train['week'] = train.pickup_datetime.dt.week
train['weekday'] = train.pickup_datetime.dt.weekday
train['hour'] = train.pickup_datetime.dt.hour
train.drop(['pickup_datetime'], axis=1, inplace=True)
train.drop(['dropoff_datetime'], axis=1, inplace=True)
testfile['week'] = testfile.pickup_datetime.dt.week
testfile['weekday'] = testfile.pickup_datetime.dt.weekday
testfile['hour'] = testfile.pickup_datetime.dt.hour
testfile.drop(['pickup_datetime'], axis=1, inplace=True)


# Now that we have our features, we can take a look at our target.

# In[ ]:


#Visualising the distribution of trip_duration values
plt.subplots(figsize=(18,6))
plt.hist(train['trip_duration'].values, bins=100)
plt.xlabel('trip_duration')
plt.ylabel('number of train records')
plt.show()


# The distribution is right-skewed so we can consider a log-transformation of `trip_duration` data.

# In[ ]:


#Log transformation
plt.subplots(figsize=(18,6))
train['trip_duration'] = np.log(train['trip_duration'].values) #+1 is not needed here as our trip_duration values are all positive and not normalized. But it would be necessary to normalize and add 1 to make a robust code for new data.
plt.hist(train['trip_duration'].values, bins=100)
plt.xlabel('log(trip_duration)')
plt.ylabel('number of train records')
plt.show()


# ## Model Selection and Training

# In[ ]:


y = train["trip_duration"]
train.drop(["trip_duration"], axis=1, inplace=True)
X = train
X.shape, y.shape


# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.shape, y_train.shape, X_test.shape, y_test.shape


# In[ ]:


X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, random_state=42)
X_train.shape, y_train.shape, X_valid.shape, y_valid.shape


# In[ ]:


from sklearn.ensemble import RandomForestRegressor

#The randomforestregressor params are chosen in the following hyperparameters tuning
m1 = RandomForestRegressor(n_estimators=19, min_samples_split=2, min_samples_leaf=4, max_features='auto', max_depth=80, bootstrap=True)
m1.fit(X_train, y_train)
m1.score(X_valid, y_valid)


# In[ ]:


#from sklearn.ensemble import GradientBoostingRegressor

#gradient_boosted = GradientBoostingRegressor()
#gradient_boosted.fit(X_train, y_train)
#gradient_boosted.score(X_valid, y_valid)

#score: around 0.5


# In[ ]:


from sklearn.metrics import mean_squared_error as MSE

print(np.sqrt(MSE(y_valid, m1.predict(X_valid))))
#print(np.sqrt(MSE(y_valid, gradient_boosted.predict(X_valid))))


# RandomForestRegressor seems to fit better than GradientBoostingRegressor. Now here is how I chose the RFR hyperparameters.

# ### Hyperparameters tuning

# We are going to look at the better combination of hyperparameters using RandomizedSearchCV. It took me 138.2min to run this cell so be patient if you're using it.

# In[ ]:


#from sklearn.model_selection import RandomizedSearchCV

#n_estimators = [int(x) for x in np.linspace(start = 5, stop = 20, num = 16)]
#max_features = ['auto', 'sqrt']
#max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
#max_depth.append(None)
#min_samples_split = [2, 5, 10]
#min_samples_leaf = [1, 2, 4]
#bootstrap = [True, False]

#random_grid = {'n_estimators': n_estimators, 'max_features': max_features, 'max_depth': max_depth, 'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf, 'bootstrap': bootstrap}

#random_cv = RandomizedSearchCV(estimator = m1, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
#print(random_cv.best_params_)


# ## Predictions and Submission

# In[ ]:


test_columns = X_train.columns
predictions = m1.predict(testfile[test_columns])


# In[ ]:


my_submission = pd.DataFrame({'id': testfile.id, 'trip_duration': np.exp(predictions)})
my_submission.head()


# In[ ]:


my_submission.to_csv("sub.csv", index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




