#!/usr/bin/env python
# coding: utf-8

# The objective of the notebook is to try and get to the top of the LB (hopefully.!).
# 
# With just two weeks remaining for the competition to end, this will be a perfect start (as the competitions are generally more harder in the final days with all the forum informations, exploratory notebooks, high scoring kernels and so on).
# 
# The general flow is as follows:
# * Understanding the data
# * Validation Strategy
# * Create a baseline model with basic variables
# * Feature Engineering
# * Building various models and parameter tuning
# * Ensembling / stacking.
# 
# **Competition Objective:**
# 
# The objective of the competition is to predict the trip time duration of the Taxis in New Tork City.
# 
# **Understanding the data:**
# 
# Let us import the dataset and have a sneak peak at what kind of data is present inside.
# 

# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection, preprocessing, metrics
import xgboost as xgb
import lightgbm as lgb
from haversine import haversine
color = sns.color_palette()

get_ipython().run_line_magic('matplotlib', 'inline')

pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('display.max_columns', 500)


# Reading the dataset into pandas dataframe and looking at the top few rows.

# In[3]:


train_df = pd.read_csv('../input/nyc-taxi-trip-duration/train.csv', parse_dates=['pickup_datetime'])
test_df = pd.read_csv('../input/nyc-taxi-trip-duration/test.csv', parse_dates=['pickup_datetime'])
print("Train dataframe shape : ",train_df.shape)
print("Test dataframe shape : ", test_df.shape)


# In[4]:


train_df.head()


# In[5]:


test_df.head()


# The columns are self-explanatory and two columns 'dropoff_datetime' and 'trip_duration' are not present in the test set. 
# 
# 'trip_duration' is the column to predict and Root Mean Square Logarithmic Error is our error metric. So let us look at the log distribution of the target variable.

# In[6]:


train_df['log_trip_duration'] = np.log1p(train_df['trip_duration'].values)

plt.figure(figsize=(8,6))
plt.scatter(range(train_df.shape[0]), np.sort(train_df.log_trip_duration.values))
plt.xlabel('index', fontsize=12)
plt.ylabel('log_trip_duration', fontsize=12)
plt.show()


# Looks like there are few outiers in the data. Let us check  the actual count of it.

# In[7]:


(train_df['log_trip_duration'] > 12).sum()


# I think 4 is a smaller value and let us not worry about it now (but probably need to check later if needed).
# 
# Next step is to check whether there are any null values in the data. 

# In[8]:


null_count_df = train_df.isnull().sum(axis=0).reset_index()
null_count_df.columns = ['col_name', 'null_count']
null_count_df


# In[9]:


null_count_df = test_df.isnull().sum(axis=0).reset_index()
null_count_df.columns = ['col_name', 'null_count']
null_count_df


# There are no missing values :)
# 
# **Validation Strategy:**
# 
# Validation strategy is very important because without a proper validation starategy, it will be very hard to evaluate the models against each other. 
# 
# Since dates are given as part of the dataset, it is essential to check whether the train and test datasets are from the same time period or different time period.

# In[10]:


train_df['pickup_date'] = train_df['pickup_datetime'].dt.date
test_df['pickup_date'] = test_df['pickup_datetime'].dt.date

cnt_srs = train_df['pickup_date'].value_counts()
plt.figure(figsize=(12,4))
ax = plt.subplot(111)
ax.bar(cnt_srs.index, cnt_srs.values, alpha=0.8)
ax.xaxis_date()
plt.xticks(rotation='vertical')
plt.show()


# In[11]:


cnt_srs = test_df['pickup_date'].value_counts()
plt.figure(figsize=(12,4))
ax = plt.subplot(111)
ax.bar(cnt_srs.index, cnt_srs.values, alpha=0.8)
ax.xaxis_date()
plt.xticks(rotation='vertical')
plt.show()


# Wow. The distributions are very similar and so we could potentially use the K-fold cross validation for our dataset. Please note that if the train and test datasets are from different time frames, kindly use time based validation.

# **Baseline Model:**
# 
# Now that we have got an idea about the dataset, let us buid a baseline model using xgboost and check the performance. 
# 
# We could create few basic variables from datetime column and convert the store_and_forward_flag to numeric.

# In[12]:


# day of the month #
train_df['pickup_day'] = train_df['pickup_datetime'].dt.day
test_df['pickup_day'] = test_df['pickup_datetime'].dt.day

# month of the year #
train_df['pickup_month'] = train_df['pickup_datetime'].dt.month
test_df['pickup_month'] = test_df['pickup_datetime'].dt.month

# hour of the day #
train_df['pickup_hour'] = train_df['pickup_datetime'].dt.hour
test_df['pickup_hour'] = test_df['pickup_datetime'].dt.hour

# Week of year #
train_df["week_of_year"] = train_df["pickup_datetime"].dt.weekofyear
test_df["week_of_year"] = test_df["pickup_datetime"].dt.weekofyear

# Day of week #
train_df["day_of_week"] = train_df["pickup_datetime"].dt.weekday
test_df["day_of_week"] = test_df["pickup_datetime"].dt.weekday

# Convert to numeric #
map_dict = {'N':0, 'Y':1}
train_df['store_and_fwd_flag'] = train_df['store_and_fwd_flag'].map(map_dict)
test_df['store_and_fwd_flag'] = test_df['store_and_fwd_flag'].map(map_dict)


# In[14]:


# drop off the variables which are not needed #
cols_to_drop = ['id', 'pickup_datetime', 'pickup_date']
train_id = train_df['id'].values
test_id = test_df['id'].values
train_y = train_df.log_trip_duration.values
train_X = train_df.drop(cols_to_drop + ['dropoff_datetime', 'trip_duration', 'log_trip_duration'], axis=1)
test_X = test_df.drop(cols_to_drop, axis=1)


# Let us write a helper function to run the xgboost model and light gbm model.

# In[15]:


def runXGB(train_X, train_y, val_X, val_y, test_X, eta=0.05, max_depth=5, min_child_weight=1, subsample=0.8, colsample=0.7, num_rounds=8000, early_stopping_rounds=50, seed_val=2017):
    params = {}
    params["objective"] = "reg:linear"
    params['eval_metric'] = "rmse"
    params["eta"] = eta
    params["min_child_weight"] = min_child_weight
    params["subsample"] = subsample
    params["colsample_bytree"] = colsample
    params["silent"] = 1
    params["max_depth"] = max_depth
    params["seed"] = seed_val
    params["nthread"] = -1

    plst = list(params.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)
    xgval = xgb.DMatrix(val_X, label = val_y)
    xgtest = xgb.DMatrix(test_X)
    watchlist = [ (xgtrain,'train'), (xgval, 'test') ]
    model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=20)

    pred_val = model.predict(xgval, ntree_limit=model.best_ntree_limit)
    pred_test = model.predict(xgtest, ntree_limit=model.best_ntree_limit)

    return pred_val, pred_test

def runLGB(train_X, train_y, val_X, val_y, test_X, eta=0.05, max_depth=5, min_child_weight=1, subsample=0.8, colsample=0.7, num_rounds=8000, early_stopping_rounds=50, seed_val=2017):
    params = {}
    params["objective"] = "regression"
    params['metric'] = "l2_root"
    params["learning_rate"] = eta
    params["min_child_weight"] = min_child_weight
    params["bagging_fraction"] = subsample
    params["bagging_seed"] = seed_val
    params["feature_fraction"] = colsample
    params["verbosity"] = 0
    params["max_depth"] = max_depth
    params["nthread"] = -1

    lgtrain = lgb.Dataset(train_X, label=train_y)
    lgval = lgb.Dataset(val_X, label = val_y)
    model = lgb.train(params, lgtrain, num_rounds, valid_sets=lgval, early_stopping_rounds=early_stopping_rounds, verbose_eval=20)

    pred_val = model.predict(val_X, num_iteration=model.best_iteration)
    pred_test = model.predict(test_X, num_iteration=model.best_iteration)

    return pred_val, pred_test, model


# Now let us build the baseline model using K-fold cross validation and save the scores in a csv file so as to build ensembles / stacking models later.
# 
# Please increase the number of rounds ('num_rounds') to a high value (1000) and then run the model in local. Using just 10 rounds here 

# In[16]:


# Increase the num_rounds parameter to a higher value (1000) and run the model #
kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)
cv_scores = []
pred_test_full = 0
pred_val_full = np.zeros(train_df.shape[0])
for dev_index, val_index in kf.split(train_X):
    dev_X, val_X = train_X.ix[dev_index], train_X.ix[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    pred_val, pred_test, model = runLGB(dev_X, dev_y, val_X, val_y, test_X, num_rounds=5, max_depth=8, eta=0.3)
    pred_val_full[val_index] = pred_val
    pred_test_full += pred_test
    cv_scores.append(np.sqrt(metrics.mean_squared_error(val_y, pred_val)))
print(cv_scores)
print("Mean CV score : ",np.mean(cv_scores))

pred_test_full = pred_test_full / 5.
pred_test_full = np.expm1(pred_test_full)
pred_val_full = np.expm1(pred_val_full)

# saving train predictions for ensemble #
train_pred_df = pd.DataFrame({'id':train_id})
train_pred_df['trip_duration'] = pred_val_full
train_pred_df.to_csv("train_preds_lgb_baseline.csv", index=False)

# saving test predictions for ensemble #
test_pred_df = pd.DataFrame({'id':test_id})
test_pred_df['trip_duration'] = pred_test_full
test_pred_df.to_csv("test_preds_lgb_baseline.csv", index=False)


# **The mean cv score (when trained offline with 1000 rounds) is 0.4147 and the LB score is 0.41412 (around rank 400 which is not bad for a baseline model)**. 
# 
# This baseline model has helped us understand the following:
# 1. Our overall setup is ready so that now we can make additional changes wherever needed
# 2. The scores seem to be matching between cross validation and leaderboard and so we are probably good in that front
# 
# So some of the next steps are as follows:
# 1. Feature engineering to create more useful variables.
# 2. Ascertain that the cross validation and LB scores are in sync.
# 3. Parameter tuning and building varied models.
# 4. Ensembling / Stacking
# 
# **Feature Engineering:**
# 
# Now that we have our base model and the overall setup ready, let us dive into creating more variables (This is where I generally spend most of my time during competitions). It is a good idea to look at the feature importances of the model which we have built to understand what type of features are generally more predictive. So I got the feature importances from the light gbm model and is as follows:
# 
# | Feature Name | Feature Importance |
# |:----------------|------------------------:|
# | dropoff_latitude | 0.1761 |
# | pickup_latitude | 0.1729 | 
# | pickup_longitude | 0.172 |
# | dropoff_longitude | 0.1581 |
# | pickup_hour | 0.0999 | 
# | pickup_day | 0.0611 |
# | week_of_year | 0.0538 |
# | day_of_week | 0.0499 | 
# | pickup_month | 0.0203 | 
# | passenger_count | 0.0203 | 
# | vendor_id | 0.0132 |
# | store_and_fwd_flag | 0.0021 |
# 
# The important variables order seem to be the lat-lon co-ordinates followed by the time based variables. So some of my ideas to create new variables and the reasons are as follows
# 1. Difference between pickup and dropoff latitude - will give an idea about the distance covered which could be predictive
# 2. Difference between pickup and dropoff longitude - same reason as above
# 3. Haversine distance between pickup and dropoff co-ordinates - to capture the actual distance travelled (commented out so as to use a faster function written by beluga)
# 4. Pickup minute - since pickup hour is an important variable, the minute of pickup might well have been predictive
# 5. Pickup day of year - same reason as above
# 
# So let us create these variables first and re-run it again.

# In[17]:


# difference between pickup and dropoff latitudes #
train_df['lat_diff'] = train_df['pickup_latitude'] - train_df['dropoff_latitude']
test_df['lat_diff'] = test_df['pickup_latitude'] - test_df['dropoff_latitude']

# difference between pickup and dropoff longitudes #
train_df['lon_diff'] = train_df['pickup_longitude'] - train_df['dropoff_longitude']
test_df['lon_diff'] = test_df['pickup_longitude'] - test_df['dropoff_longitude']

## compute the haversine distance ##
#train_df['haversine_distance'] = train_df.apply(lambda row: haversine( (row['pickup_latitude'], row['pickup_longitude']), (row['dropoff_latitude'], row['dropoff_longitude']) ), axis=1)
#test_df['haversine_distance'] = test_df.apply(lambda row: haversine( (row['pickup_latitude'], row['pickup_longitude']), (row['dropoff_latitude'], row['dropoff_longitude']) ), axis=1)

# get the pickup minute of the trip #
train_df['pickup_minute'] = train_df['pickup_datetime'].dt.minute
test_df['pickup_minute'] = test_df['pickup_datetime'].dt.minute

# get the absolute value of time #
train_df['pickup_dayofyear'] = train_df['pickup_datetime'].dt.dayofyear
test_df['pickup_dayofyear'] = test_df['pickup_datetime'].dt.dayofyear


# Now let us re-run the model again with these new variables and check the score.

# In[16]:


# drop off the variables which are not needed #
cols_to_drop = ['id', 'pickup_datetime', 'pickup_date']
train_id = train_df['id'].values
test_id = test_df['id'].values
train_y = train_df.log_trip_duration.values
train_X = train_df.drop(cols_to_drop + ['dropoff_datetime', 'trip_duration', 'log_trip_duration'], axis=1)
test_X = test_df.drop(cols_to_drop, axis=1)

# Increase the num_rounds parameter to a higher value (1000) and run the model #
kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)
cv_scores = []
pred_test_full = 0
pred_val_full = np.zeros(train_df.shape[0])
for dev_index, val_index in kf.split(train_X):
    dev_X, val_X = train_X.ix[dev_index], train_X.ix[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    pred_val, pred_test, model = runLGB(dev_X, dev_y, val_X, val_y, test_X, num_rounds=5, max_depth=8, eta=0.3)
    pred_val_full[val_index] = pred_val
    pred_test_full += pred_test
    cv_scores.append(np.sqrt(metrics.mean_squared_error(val_y, pred_val)))
print(cv_scores)
print("Mean CV score : ",np.mean(cv_scores))

pred_test_full = pred_test_full / 5.
pred_test_full = np.expm1(pred_test_full)
pred_val_full = np.expm1(pred_val_full)

# saving train predictions for ensemble #
train_pred_df = pd.DataFrame({'id':train_id})
train_pred_df['trip_duration'] = pred_val_full
train_pred_df.to_csv("train_preds_lgb.csv", index=False)

# saving test predictions for ensemble #
test_pred_df = pd.DataFrame({'id':test_id})
test_pred_df['trip_duration'] = pred_test_full
test_pred_df.to_csv("test_preds_lgb.csv", index=False)


# **The CV score of this new model is 0.3875 and the LB score is 0.38809. **
# 
# So the scores are pretty much in sync with each other. This is really a great news since in many competitions, they will be far away (in which cases, directional improvement can be looked at).
# 
# Our new feature importances are as follows:
# 
# |Feature name | Feature Importance|
# |:----------------|------------------:|
# | pickup_longitude | 0.1060 |
# | dropoff_latitude | 0.1055| 
# | haversine_distance | 0.1007 | 
# | dropoff_longitude | 0.0990 | 
# | pickup_latitude | 0.0983 |
# | pickup_hour | 0.0890 |
# | lon_diff | 0.0860 | 
# | lat_diff | 0.0815 | 
# | pickup_dayofyear | 0.0560 | 
# | pickup_minute | 0.0459 |
# | pickup_day | 0.0433 | 
# | day_of_week | 0.0358 |
# | week_of_year | 0.0246 |
# | passenger_count | 0.0110 |
# | vendor_id | 0.0080 |
# | pickup_month | 0.0078 |
# | store_and_fwd_flag | 0.0005 |
# 
# As the next step, we can look into the forum posts / kernels  and see if there are any good feature ideas / implementations and try to add them as well into the features list.
# 
# **More Features:**
# 
# This [excellent notebook](https://www.kaggle.com/gaborfodor/from-eda-to-the-top-lb-0-367) by beluga has a lot of wonderful features. I particularly liked the vectorized implementation of the distance  features. We shall also add them into our feature list.
# 

# In[18]:


def haversine_array(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h

def bearing_array(lat1, lng1, lat2, lng2):
    AVG_EARTH_RADIUS = 6371  # in km
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))

train_df['haversine_distance'] = haversine_array(train_df['pickup_latitude'].values, 
                                                     train_df['pickup_longitude'].values, 
                                                     train_df['dropoff_latitude'].values, 
                                                     train_df['dropoff_longitude'].values)
test_df['haversine_distance'] = haversine_array(test_df['pickup_latitude'].values, 
                                                    test_df['pickup_longitude'].values, 
                                                    test_df['dropoff_latitude'].values, 
                                                    test_df['dropoff_longitude'].values)

train_df['direction'] = bearing_array(train_df['pickup_latitude'].values, 
                                          train_df['pickup_longitude'].values, 
                                          train_df['dropoff_latitude'].values, 
                                          train_df['dropoff_longitude'].values)
test_df['direction'] = bearing_array(test_df['pickup_latitude'].values, 
                                         test_df['pickup_longitude'].values, 
                                         test_df['dropoff_latitude'].values, 
                                         test_df['dropoff_longitude'].values)


# Also we have this [wonderful osrm dataset](https://www.kaggle.com/oscarleo/new-york-city-taxi-with-osrm) published by oscarleo which gets the fastest and the second fastest routes for both train and test sets. Thank you oscarleo for this very helpful dataset. 

# In[20]:


train_fr_part1 = pd.read_csv('../input/new-york-city-taxi-with-osrm/fastest_routes_train_part_1.csv', 
                             usecols=['id', 'total_distance', 'total_travel_time', 'number_of_steps'])
train_fr_part2 = pd.read_csv('../input/new-york-city-taxi-with-osrm/fastest_routes_train_part_2.csv', 
                             usecols=['id', 'total_distance', 'total_travel_time', 'number_of_steps'])
test_fr = pd.read_csv('../input/new-york-city-taxi-with-osrm/fastest_routes_test.csv', 
                             usecols=['id', 'total_distance', 'total_travel_time', 'number_of_steps'])
train_fr = pd.concat((train_fr_part1, train_fr_part2))

train_df = train_df.merge(train_fr, how='left', on='id')
test_df = test_df.merge(test_fr, how='left', on='id')

del train_fr_part1, train_fr_part2, train_fr, test_fr
import gc; gc.collect()


# Why just the fastest route alone when the second fastest route is available? I tried adding the second fastest variables but it did not help much in the CV and so did not include them.
# 
# Let us create few more variables by binning the latitudes and longitudes together.

# In[21]:


### some more new variables ###
train_df['pickup_latitude_round3'] = np.round(train_df['pickup_latitude'],3)
test_df['pickup_latitude_round3'] = np.round(test_df['pickup_latitude'],3)

train_df['pickup_longitude_round3'] = np.round(train_df['pickup_longitude'],3)
test_df['pickup_longitude_round3'] = np.round(test_df['pickup_longitude'],3)

train_df['dropoff_latitude_round3'] = np.round(train_df['dropoff_latitude'],3)
test_df['dropoff_latitude_round3'] = np.round(test_df['dropoff_latitude'],3)

train_df['dropoff_longitude_round3'] = np.round(train_df['dropoff_longitude'],3)
test_df['dropoff_longitude_round3'] = np.round(test_df['dropoff_longitude'],3)


# Let us run our models again now to check the scores.

# In[22]:


# drop off the variables which are not needed #
cols_to_drop = ['id', 'pickup_datetime', 'pickup_date']
train_id = train_df['id'].values
test_id = test_df['id'].values
train_y = train_df.log_trip_duration.values
train_X = train_df.drop(cols_to_drop + ['dropoff_datetime', 'trip_duration', 'log_trip_duration'], axis=1)
test_X = test_df.drop(cols_to_drop, axis=1)

# Increase the num_rounds parameter to a higher value (1000) and run the model #
kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)
cv_scores = []
pred_test_full = 0
pred_val_full = np.zeros(train_df.shape[0])
for dev_index, val_index in kf.split(train_X):
    dev_X, val_X = train_X.ix[dev_index], train_X.ix[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    pred_val, pred_test, model = runLGB(dev_X, dev_y, val_X, val_y, test_X, num_rounds=5, max_depth=8, eta=0.3)
    pred_val_full[val_index] = pred_val
    pred_test_full += pred_test
    cv_scores.append(np.sqrt(metrics.mean_squared_error(val_y, pred_val)))
print(cv_scores)
print("Mean CV score : ",np.mean(cv_scores))

pred_test_full = pred_test_full / 5.
pred_test_full = np.expm1(pred_test_full)
pred_val_full = np.expm1(pred_val_full)

# saving train predictions for ensemble #
train_pred_df = pd.DataFrame({'id':train_id})
train_pred_df['trip_duration'] = pred_val_full
train_pred_df.to_csv("train_preds_lgb.csv", index=False)

# saving test predictions for ensemble #
test_pred_df = pd.DataFrame({'id':test_id})
test_pred_df['trip_duration'] = pred_test_full
test_pred_df.to_csv("test_preds_lgb.csv", index=False)


# **The CV score of this version is 0.3784 and the LB score is 0.3799. **
# 
# Though the deviation between CV and LB deviated a little compared to previous submissions, I think it is okay since the deviation is not very high.
# 
# Now that we are on the right path, let us try to create few more variables in the next version to further improve the score.

# **More to come. Stay tuned.!**
