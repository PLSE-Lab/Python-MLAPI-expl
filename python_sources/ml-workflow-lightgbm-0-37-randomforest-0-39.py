#!/usr/bin/env python
# coding: utf-8

# # New York City Taxi Trip Duration

# The competition dataset is based on the 2016 NYC Yellow Cab trip record data made available in Big Query on Google Cloud Platform. The data was originally published by the NYC Taxi and Limousine Commission (TLC). The data was sampled and cleaned for the purposes of this playground competition. Based on individual trip attributes, participants should **predict the duration of each trip in the test set.**
# 
# ### File descriptions
# 
# **train.csv** - the training set (contains 1458644 trip records)  
# **test.csv** - the testing set (contains 625134 trip records)  
# **sample_submission.csv** - a sample submission file in the correct format
# 
# ### Table of Contents:
# **[I. Data loading and overview](#one)**
# - [a. Loading the data](#one-a)
# - [b. Overview](#one-b)
# 
# **[II. Data cleaning](#two)**
# - [a. Duplicated and missing values](#two-a)
# - [b. Deal with outliers](#two-b)
# 
# **[III. Features engineering](#three)**
# - [a. Target](#three-a)
# - [b. Deal with categorical features](#three-b)
# - [c. Deal with dates](#three-c)
# - [d. Distance and speed creations](#three-d)
# - [e. Correlations and dimensionality reductions](#three-e)
# 
# **[IV. Model selection](#four)**
# - [a. Split](#four-a)
# - [b. Metrics](#four-b)
# - [c. Models](#four-c)
# - [d. Cross-validation](#four-d)
# 
# **[V. Hyperparameters tuning](#five)**
# 
# **[VI. Training and predictions](#six)**

# ## I. Data loading and overview <a id="one"></a>

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# ### a. Loading the data <a id="one-a"></a>

# In[ ]:


df = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[ ]:


df.head(5)


# In[ ]:


test.head(5)


# ### b. Overview <a id="one-b"></a>

# Colonne | Description
# ------- | -------
# **id** | a unique identifier for each trip  
# **vendor_id** | a code indicating the provider associated with the trip record  
# **pickup_datetime** | date and time when the meter was engaged  
# **dropoff_datetime** | date and time when the meter was disengaged  
# **passenger_count** | the number of passengers in the vehicle (driver entered value)  
# **pickup_longitude** | the longitude where the meter was engaged  
# **pickup_latitude** | the latitude where the meter was engaged  
# **dropoff_longitude** | the longitude where the meter was disengaged  
# **dropoff_latitude** | the latitude where the meter was disengaged  
# **store_and_fwd_flag** | This flag indicates whether the trip record was held in vehicle memory before sending to the vendor because the vehicle did not have a connection to the server (Y=store and forward; N=not a store and forward trip)  
# **trip_duration** | duration of the trip in seconds  
# 
# *Disclaimer: The decision was made to not remove dropoff coordinates from the dataset order to provide an expanded set of variables to use in Kernels.*

# In[ ]:


df.info()
test.info()


# ## II. Data cleaning <a id="two"></a>

# ### a. Duplicated and missing values <a id="two-a"></a>

# In[ ]:


#Count the number of duplicated rows
df.duplicated().sum()


# In[ ]:


#Count the number of NaN values for each column
df.isna().sum()


# There are no duplicated or missing values.

# ### b. Deal with outliers <a id="two-b"></a>

# In[ ]:


df.describe()


# We clearly see `trip_duration` takes strange values for `min` and `max`. Let's have a quick visualization with a boxplot.

# In[ ]:


#Visualize univariate outliers
plt.subplots(figsize=(18,6))
plt.title("Outliers visualization")
df.boxplot();


# There are outliers for `trip_duration`. I can't find a proper interpretation and it will probably damage our model, so I choose to get rid of them.

# In[ ]:


#Only keep trips that lasted less than 5900 seconds
df = df[(df.trip_duration < 5900)]


# In[ ]:


#Only keep trips with passengers
df = df[(df.passenger_count > 0)]


# In[ ]:


#Plot pickup positions to visualize outliers
pickup_longitude = list(df.pickup_longitude)
pickup_latitude = list(df.pickup_latitude)
plt.subplots(figsize=(18,6))
plt.plot(pickup_longitude, pickup_latitude, '.', alpha = 1, markersize = 10)
plt.xlabel('pickup_longitude')
plt.ylabel('pickup_latitude')
plt.show()


# In[ ]:


#Plot dropoff positions to visualize outliers
dropoff_longitude = list(df.dropoff_longitude)
dropoff_latitude = list(df.dropoff_latitude)
plt.subplots(figsize=(18,6))
plt.plot(dropoff_longitude, dropoff_latitude, '.', alpha = 1, markersize = 10)
plt.xlabel('dropoff_longitude')
plt.ylabel('dropoff_latitude')
plt.show()


# In[ ]:


#Remove position outliers
df = df[(df.pickup_longitude > -100)]
df = df[(df.pickup_latitude < 50)]
#df = df[(df.dropoff_longitude < -70) & (df.dropoff_longitude > -80)]
#df = df[(df.dropoff_latitude < 50)]


# ## III. Features engineering <a id="three"></a>

# ### a. Target <a id="three-a"></a>

# In[ ]:


#Visualize the distribution of trip_duration values
plt.subplots(figsize=(18,6))
plt.hist(df['trip_duration'].values, bins=100)
plt.xlabel('trip_duration')
plt.ylabel('number of train records')
plt.show()


# The distribution is **right-skewed** so we can consider a log-transformation of `trip_duration` column.

# In[ ]:


#Log-transformation
plt.subplots(figsize=(18,6))
df['trip_duration'] = np.log(df['trip_duration'].values)
plt.hist(df['trip_duration'].values, bins=100)
plt.xlabel('log(trip_duration)')
plt.ylabel('number of train records')
plt.show()


# ### b. Deal with categorical features <a id="three-b"></a>

# In[ ]:


#One-hot encoding binary categorical features
df = pd.concat([df, pd.get_dummies(df['store_and_fwd_flag'])], axis=1)
test = pd.concat([test, pd.get_dummies(test['store_and_fwd_flag'])], axis=1)

df.drop(['store_and_fwd_flag'], axis=1, inplace=True)

df = pd.concat([df, pd.get_dummies(df['vendor_id'])], axis=1)
test = pd.concat([test, pd.get_dummies(test['vendor_id'])], axis=1)

df.drop(['vendor_id'], axis=1, inplace=True)


# ### c. Deal with dates <a id="three-c"></a>

# In[ ]:


#Datetyping the dates
df['pickup_datetime'] = pd.to_datetime(df.pickup_datetime)
test['pickup_datetime'] = pd.to_datetime(test.pickup_datetime)

df.drop(['dropoff_datetime'], axis=1, inplace=True) #as we don't have this feature in the testset

#Date features creations and deletions
df['month'] = df.pickup_datetime.dt.month
df['week'] = df.pickup_datetime.dt.week
df['weekday'] = df.pickup_datetime.dt.weekday
df['hour'] = df.pickup_datetime.dt.hour
df['minute'] = df.pickup_datetime.dt.minute
df['minute_oftheday'] = df['hour'] * 60 + df['minute']
df.drop(['minute'], axis=1, inplace=True)

test['month'] = test.pickup_datetime.dt.month
test['week'] = test.pickup_datetime.dt.week
test['weekday'] = test.pickup_datetime.dt.weekday
test['hour'] = test.pickup_datetime.dt.hour
test['minute'] = test.pickup_datetime.dt.minute
test['minute_oftheday'] = test['hour'] * 60 + test['minute']
test.drop(['minute'], axis=1, inplace=True)

df.drop(['pickup_datetime'], axis=1, inplace=True)

df.info()


# ### d. Distance and speed creations <a id="three-d"></a>

# In[ ]:


#Function aiming at calculating distances from coordinates
def ft_haversine_distance(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371 #km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h

#Add distance feature
df['distance'] = ft_haversine_distance(df['pickup_latitude'].values,
                                                 df['pickup_longitude'].values, 
                                                 df['dropoff_latitude'].values,
                                                 df['dropoff_longitude'].values)
test['distance'] = ft_haversine_distance(test['pickup_latitude'].values, 
                                                test['pickup_longitude'].values, 
                                                test['dropoff_latitude'].values, 
                                                test['dropoff_longitude'].values)


# In[ ]:


#Function aiming at calculating the direction
def ft_degree(lat1, lng1, lat2, lng2):
    AVG_EARTH_RADIUS = 6371 #km
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))

#Add direction feature
df['direction'] = ft_degree(df['pickup_latitude'].values,
                                df['pickup_longitude'].values,
                                df['dropoff_latitude'].values,
                                df['dropoff_longitude'].values)
test['direction'] = ft_degree(test['pickup_latitude'].values,
                                  test['pickup_longitude'].values, 
                                  test['dropoff_latitude'].values,
                                  test['dropoff_longitude'].values)


# In[ ]:


#Visualize distance outliers
df.boxplot(column='distance', return_type='axes');


# In[ ]:


#Remove distance outliers
df = df[(df.distance < 200)]


# In[ ]:


#Create speed feature
df['speed'] = df.distance / df.trip_duration


# In[ ]:


#Visualize speed feature
df.boxplot(column='speed', return_type='axes');


# In[ ]:


#Remove speed outliers
df = df[(df.speed < 30)]
df.drop(['speed'], axis=1, inplace=True)


# ### e. Correlations and dimensionality reductions <a id="three-e"></a>

# In[ ]:


#Correlations between variables
fig, ax = plt.subplots(figsize=(14,5))  
sns.heatmap(data=df.corr(), annot=True, cmap = plt.cm.RdYlBu_r, linewidths=.1, ax=ax).set_title('Correlations between variables');


# ## IV. Model selection <a id="four"></a>

# ### a. Split <a id="four-a"></a>

# In[ ]:


#Split the labeled data frame into two sets: features and target
y = df["trip_duration"]
df.drop(["trip_duration"], axis=1, inplace=True)
df.drop(['id'], axis=1, inplace=True)
X = df

X.shape, y.shape


# In[ ]:


#Split the labeled data frame into two sets to train then test the models
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.shape, y_train.shape, X_test.shape, y_test.shape


# ### b. Metrics <a id="four-b"></a>

# For this specific problematic, we'll measure the error using the RMSE (Root Mean Squared Error).

# In[ ]:


from sklearn.metrics import mean_squared_error as MSE


# ### c. Models <a id="four-c"></a>

# In[ ]:


#%%time
#Try GradientBoosting
#from sklearn.ensemble import GradientBoostingRegressor

#gb = GradientBoostingRegressor()
#gb.fit(X_train, y_train)
#print(gb.score(X_train, y_train), gb.score(X_test, y_test))
#print(np.sqrt(MSE(y_test, gb.predict(X_test))))
    
#Output
    #0.7454771059776502 0.7443578507676307
    #0.39291173774102295
    #CPU times: user 3min 48s, sys: 328 ms, total: 3min 48s
    #Wall time: 3min 48s


# In[ ]:


#%%time
#Try RandomForest
#from sklearn.ensemble import RandomForestRegressor

#rf = RandomForestRegressor()
#rf.fit(X_train, y_train)
#print(rf.score(X_train, y_train), rf.score(X_test, y_test))
#print(np.sqrt(MSE(y_test, rf.predict(X_test))))

#Output:
    #0.9601197799928392 0.7790255381297454
    #0.36530012047088345
    #CPU times: user 3min, sys: 792 ms, total: 3min 1s
    #Wall time: 3min 1s


# In[ ]:


#%%time
#Try LightGBM ----------------------------
import lightgbm as lgb

#lgb_params = {
#    'metric': 'rmse',
#    'is_training_metric': True}

#lgb_train = lgb.Dataset(X_train, y_train)
#lgb_test = lgb.Dataset(X_test, y_test)
#lgb_model = lgb.train(lgb_params, lgb_train, num_boost_round=100, valid_sets=[lgb_train, lgb_test], early_stopping_rounds=5)

#Output
    #[100]	valid_0's rmse: 0.362209	valid_1's rmse: 0.3629
    #CPU times: user 40.9 s, sys: 332 ms, total: 41.2 s
    #Wall time: 21.2 s

#Try LightGBM with sklearn API ------------
#from lightgbm import LGBMRegressor

#lgbm = lgb.LGBMRegressor()
#lgbm.fit(X, y)
#print(lgbm.score(X_train, y_train), lgbm.score(X_test, y_test))
#print(np.sqrt(MSE(y_test, lgbm.predict(X_test))))

#Output:
    #0.7812886118508641 0.7827256176145024
    #0.3623481127815768
    #CPU times: user 42 s, sys: 1.08 s, total: 43 s
    #Wall time: 22.5 s


# **LightGBM** is blazingly fast compared to RandomForest and classic GradientBoosting, while fitting better. It is our clear winner.

# ### d. Cross-validation <a id="four-d"></a>

# In[ ]:


#Cross-validation on LightGBM model --------------------------
#lgb_df = lgb.Dataset(X, y)
#lgb.cv(lgb_params, lgb_df, stratified=False) #False is needed as it only works with classification

#Cross-validation on LightGBM model (sklearn API) ------------
#from sklearn.model_selection import cross_val_score

#cv_score = cross_val_score(lgbm, X, y, cv=5)
#print(cv_score)
#print(np.mean(cv_score))

#Output:
    #[0.77872018 0.7801329  0.77988107 0.78049745 0.77904688]
    #0.7796556968369478


# Our LightGBM model is stable.

# ## V. Hyperparameters tuning <a id="five"></a>

# In[ ]:


#Hyperparameters tuning using RandomizedSearchCV
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


# To fine-tune the hyperparameters in LGB, you can take a look at its documentation which provides some great advice: https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html.

# In[ ]:


#Test the following parameters
lgb_params = {
    #'metric' : 'rmse',
    'learning_rate': 0.1,
    'max_depth': 25,
    'num_leaves': 1000, 
    'objective': 'regression',
    'feature_fraction': 0.9,
    'bagging_fraction': 0.5,
    'max_bin': 1000 }

#lgb_train = lgb.Dataset(X_train, y_train)
#lgb_test = lgb.Dataset(X_test, y_test)
#lgb_model = lgb.train(lgb_params, lgb_train, num_boost_round=1500, valid_sets=[lgb_train, lgb_test], early_stopping_rounds=5)


# ## VI. Training and predictions <a id="six"></a>

# In[ ]:


#%%time
#Training on all labeled data using the best parameters in hyperparameters tuning
#rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=10, min_samples_split=15, max_features='auto', max_depth=90, bootstrap=True)
#rf.fit(X, y)


# In[ ]:


#%%time
#Training on all labeled data using the best parameters (sklearn API version)
#from lightgbm import LGBMRegressor

#lgbm = lgb.LGBMRegressor(n_estimators=500, num_leaves=1000, max_depth=25, objective='regression')
#lgbm.fit(X, y)


# In[ ]:


get_ipython().run_cell_magic('time', '', '#Training on all labeled data using the best parameters\nlgb_df = lgb.Dataset(X, y)\nlgb_model = lgb.train(lgb_params, lgb_df, num_boost_round=1500)')


# In[ ]:


#Make predictions on test data frame
test_columns = X.columns
predictions = lgb_model.predict(test[test_columns])


# In[ ]:


#Create a data frame designed a submission on Kaggle
submission = pd.DataFrame({'id': test.id, 'trip_duration': np.exp(predictions)})
submission.head()


# In[ ]:


#Create a csv out of the submission data frame
submission.to_csv("sub.csv", index=False)

