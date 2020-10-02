#!/usr/bin/env python
# coding: utf-8

# # How much will it cost me?
# 
# ## Note
# 
# This is one of my first machine learning related projects and the only objective is to learn. I've tried to do my best at the time, but it should be taken with a pinch of salt. Thus, critics are more than welcome!
# 
# # Introduction
# The final cost of a taxi trip is usually a surprise, depending on numerous factors that can not be foreseen in advance.
# 
# Although the most important are time and distance of a route, there are many others that affect in a more indirect way, such as the traffic of a determined area, the weather, the time of the day...
# 
# In this project I have the objective of predicting the final cost of a trip considering only information you can have beforehand. Since this is a learning project I'm only going to use the data of a single month: May 2016

# # Initial data cleaning
# 
# Before exploring the data let's remove the variables that won't be used and impossible or partial data.

# In[ ]:


import pandas as pd
import numpy as np
import datetime as dt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Parses the datetime string into datetime objects
def dateparse(x):
    try:
        return dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    except:
        return pd.NaT

# Load cleaned data
df = pd.read_csv("../input/chicago_taxi_trips_2016_05.csv", parse_dates=['trip_start_timestamp', 'trip_end_timestamp'],  date_parser=dateparse)
print(df.shape)


# In[ ]:


df.head()


# ## Feature selection
# 
# Since we are restricting the problem data to that which can be obtained before taking a taxi, we are gonna drop all but those variables.
# 
# We have two factors derived from the location: census and community area. Because for privacy reasons some census tracks are missing, we ar[](http://)e not going to use that factor.
# 
# As for the target variable, we are only considering the 'fare' without any extra (ie, tips).

# In[ ]:


# Fields not needed to our problem
to_drop = ["taxi_id",
           "pickup_census_tract",
           "dropoff_census_tract",
           "tips",
           "trip_seconds",
           "trip_miles",
           "extras",
           "trip_total",
           "company",
           "tolls",
           "payment_type",
           "trip_end_timestamp"]

# Drop selected fields in place
df.drop(to_drop, inplace=True, axis=1)


# ## Missing values
# Some entries have features missing values:

# In[ ]:


# For each feature I'm going to use for training, let's calculate and print
#the number and porcentage of missing values
features = ["trip_start_timestamp", "pickup_community_area", "dropoff_community_area", "fare", "pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"]
for f in features:
    na = df[f].isnull().sum()
    print(f, "->", "Missing values:", na, "Percentage:", na/len(df)*100)


# For 'fare', 'tolls' and the timestamps, the number of missing is low enough to delete them considering it defective data.
# 
# For the other variables we'll also need to remove them, as they are necessary for the algorithms and cannot be imputed.

# In[ ]:


df.dropna(inplace=True)
df.shape


# ## Split timestamps into usable components
# 
# Because year and month remain constant, we are only interested on the actual weekday.
# 
# Observing the time we see that seconds are always 0 and minutes always multiple of 15. I consider useful the creation of a variable 'time' with unit 15 minutes.

# In[ ]:


# Transform the start datatime object into discrete weekday and time features
df['weekday'] = df['trip_start_timestamp'].map(lambda x: x.weekday())
df['time'] = df['trip_start_timestamp'].map(lambda x: x.hour*4 + round(x.minute/15))
df.drop('trip_start_timestamp', inplace=True, axis=1)


# ## Generate additional geospatial data
# 
# It's not possible to calculate the final length of the trip beforehand, but we can calculate the lineal distance between our start and end position.
# 
# All latitude and longitude are encoded with a lookup table located on another file.

# In[ ]:


import geopy.distance

# Load lookup table
lt = pd.read_json("../input/column_remapping.json")

# Change indices with the real value
df['pickup_latitude'] = df['pickup_latitude'].map(lambda x: lt.pickup_latitude[x])
df['pickup_longitude'] = df['pickup_longitude'].map(lambda x: lt.pickup_longitude[x])
df['dropoff_latitude'] = df['dropoff_latitude'].map(lambda x: lt.dropoff_latitude[x])
df['dropoff_longitude'] = df['dropoff_longitude'].map(lambda x: lt.dropoff_longitude[x])


# In[ ]:


# Calculate lineal distance using coordinates
def calculate_distance(src):
    coords_1 = (src["pickup_latitude"], src["pickup_longitude"])
    coords_2 = (src["dropoff_latitude"],src["dropoff_longitude"])
    return geopy.distance.distance(coords_1, coords_2).m

# Generate lineal distance field
df['distance'] = df.apply(calculate_distance, axis='columns')


# # Exploratory analysis and anomaly detection
# Because it is a small set of variable lets analyze one by one.

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # Distance

# In[ ]:


sns.distplot(df["distance"]);


# We clearly see a big number of observation having a distance of 0m. This is likely to be bad data, so I'll remove those rows. Also, I'll consider as outliers any observation with a distance greater than 35000m.

# In[ ]:


# Remove detected outliers
df = df[df.distance != 0]
df = df[df.distance <= 35000]
sns.distplot(df["distance"]);
df["distance"].describe()


# That look much better!
# 
# ## Time

# In[ ]:


# Since I've scaled the time feture to blocks of 15 minutes if divided by 4 obtain hours
sns.distplot(df["time"]/4);
(df["time"]/4).describe()


# We can clearly see the daily trends, but not any outlier.
# 
# ## Weekday

# In[ ]:


# Maybe some days have more trips than others
sns.countplot(df["weekday"]);
df["weekday"].describe()


# Surprisingly we don't see any trend here. Maybe the work-related trips of the workdays is similar to the leisure one on the weekends.

# In[ ]:


# Number of trips for time for each weekday
sns.violinplot("weekday", "time", data=df)


# ## Dropoff coordinates
# 

# In[ ]:


# Plot dropoff latitude and longitude as map coordinates
sns.jointplot(y="dropoff_latitude", x="dropoff_longitude", data=df);


# We can see two cluster-like constructions on the vertical axes. Also, there are some potential outliers at low longitude values.

# In[ ]:


# Remove detected outliers and plot again
df = df[df.dropoff_longitude >= -87.85]
sns.jointplot(y="dropoff_latitude", x="dropoff_longitude", data=df);


# ## Pickup coordinates

# In[ ]:


# Plot pickup latitude and longitude as map coordinates
sns.jointplot(y="pickup_latitude", x="pickup_longitude", data=df);


# This look exactly the same plot as the dropoff case.

# In[ ]:


# Remove detected outliers and plot again
df = df[df.pickup_longitude >= -87.85]
sns.jointplot(y="pickup_latitude", x="pickup_longitude", data=df)


# ## Fare

# In[ ]:


# Generate fare histogram
sns.distplot(df["fare"]);
df["fare"].describe()


# Here we have some heavy outliers. If we just remove the 0.4% most expensive trips:

# In[ ]:


# Remove heavy outliers and plot again
df = df[df["fare"]<=55]
sns.distplot(df["fare"]);


# Clearly most of the trips are short.
# 
# ## Pickup community area

# In[ ]:


# Plot histogram for pickup community area
sns.distplot(df["pickup_community_area"])
df["pickup_community_area"].describe()


# ## Dropoff community area
# 

# In[ ]:


# Plot dropoff community area to detect potential outliers
sns.distplot(df["dropoff_community_area"])
df["dropoff_community_area"].describe()


# 
# 
# As expected source and destination are very similar. There are two groups of areas much more present than the rest.

#  ## Test / train split generation
# 

# In[ ]:


from sklearn.model_selection import train_test_split

X = df.drop('fare', axis=1)
y = df["fare"]
y = np.asarray(y, dtype=np.float64)

# Generate sets: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1337)


# # Model building

# ## Score function
# 
# A score function is needed to compare different models. Because in this case I'm going to use a common error function (MSE) which is included in sklearn metrics and most of the models, I don't need to define it explicitly. A helper function is implemented to calculate the scores given a model.
# 
# As a baseline for the model performance a dummy model which return the mean is created.

# In[ ]:


from sklearn.metrics import mean_squared_error
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import GridSearchCV
import seaborn as sns

# Given a model calculate the train and test scores
def get_scores(est):
    y_train_predict = est.predict(X_train)
    train_predict = mean_squared_error(y_train, y_train_predict)
    y_test_predict = est.predict(X_test)
    test_predict = mean_squared_error(y_test, y_test_predict)
    print("Train mse:", train_predict, "; Test mse:", test_predict)
    return (train_predict, test_predict)

# Given a model plot the residual plot for the test dataset
def plot_residuals(est):
    sns.residplot(est.predict(X_test), y_test)

# Dummy model generation and score
dummyModel = DummyRegressor(strategy="mean")
dummyModel.fit(X_train, y_train)
get_scores(dummyModel)


# The dummy model achieved a train score of 140.
# 
# ## Support Vector Machine
# 
# ```py
# from sklearn import svm
# est_svm = svm.SVR(cache_size=13000)
# # Train and score model
# est_svm = est_svm.fit(X_train, y_train)
# get_scores(est_svm)
# plot_residuals(est_svm)
# ```
# 
# Too slow for this dataset, stopped after 20 minutes with plenty of available memory.

# ##  Decision Tree

# In[ ]:


from sklearn import tree

# Parameters to tune
parameters = {'min_samples_split':[64, 128, 256],
              'min_samples_leaf':[2, 4, 16]}

# Declare objects
est_dt_r = tree.DecisionTreeRegressor()
est_dt = GridSearchCV(est_dt_r, parameters, n_jobs=-1, cv=4, verbose=1)
# Train and score model
est_dt = est_dt.fit(X_train, y_train)
get_scores(est_dt)
plot_residuals(est_dt)


# In[ ]:


# Print best parameters found by the grid search
est_dt.best_params_


# For a first model and a basic grid search the results are quite good.
# 
# Looking at the residuals plot we see a clear lineal trend that will appear on almost all the models. This can mean that we missed some important explanatory variable, which is normal, since we have restricted the set to a limited subset of variables.
# 
# ##  Stochastic Gradient Descent

# In[ ]:


from sklearn.linear_model import SGDRegressor

# Parameters to tune
parameters = {'alpha':[0.001, 0.0001, 0.00001]}

# Declare objects
est_nn_r = SGDRegressor(max_iter=10000, tol=1e-3)
est_nn = GridSearchCV(est_nn_r, parameters, n_jobs=-1, cv=4, verbose=1)
# Train and score model
est_nn = est_nn.fit(X_train, y_train)
get_scores(est_nn)
plot_residuals(est_nn)


#  This is a very bad model, order of magnitude worse than the dummy one.
#  
# ## Ensemble methods
# 
# Ensamble methods use different models to obtain a better one. Usually active better results and less overfitting, so I'm expecting to active better results than to the previous ones.
# 
# ### Random forest

# In[ ]:


from sklearn.ensemble import RandomForestRegressor

# Parameters to tune
parameters = {'n_estimators':[10, 50, 100, 150],
              'min_samples_split':[64, 128, 256],
              'min_samples_leaf': [2, 4, 6]}

# Declare objects
est_rf_r = RandomForestRegressor(random_state=1337)
est_rf = GridSearchCV(est_rf_r, parameters, cv=4, verbose=1, n_jobs=-1)
# Train and score model
est_rf = est_rf.fit(X_train, y_train)
get_scores(est_rf)
plot_residuals(est_rf)


# In[ ]:


# Print best parameters found by the grid search
est_rf.best_params_


#  Random forest is just a combination of smaller decisions trees than then vote. I was expecting to improve respect the single decision tree model, but the results are almost identical.
#  
#  ### Ada boost

# In[ ]:


from sklearn.ensemble import AdaBoostRegressor

# Parameters to tune
parameters = {'n_estimators':[25, 50, 100],
              'loss':["square", "linear"],
              'learning_rate': [0.75, 1, 1.25]}

# Declare objects
est_ada_r = AdaBoostRegressor(random_state=1337)
est_ada = GridSearchCV(est_ada_r, parameters, cv=4, verbose=1, n_jobs=-1)
# Train and score model
est_ada = est_ada.fit(X_train, y_train)
get_scores(est_ada)
plot_residuals(est_ada)


# In[ ]:


# Print best parameters found by the grid search
est_ada.best_params_


# This is the first time the residual analysis have a significative change. Despite this model is not bad, it's far from the best at the moment.
# 
# ### Gradient Tree Boosting

# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor

# Parameters to tune
parameters = {'n_estimators':[50, 100, 150],
              'min_samples_split':[64, 128, 256],
              'min_samples_leaf': [2, 4, 6]}

# Declare objects
est_gtb_r = GradientBoostingRegressor(loss='ls')
est_gtb = GridSearchCV(est_gtb_r, parameters, cv=4, verbose=1, n_jobs=-1)
# Train and score model
est_gtb = est_gtb.fit(X_train, y_train)
get_scores(est_gtb)
plot_residuals(est_gtb)
# Print best parameters found by the grid search
print(est_gtb.best_params_)


#  That's quite a good model, but doubles the error compared with the decision tree. 
#  
#  ## Neural network model
#  
# As a difference with the previous models, the data have been scaled due to the sensitivity of the algorithm to this matter.

# In[ ]:


from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# Fit and transform the scaler with the train data
scaler = StandardScaler()
scaler.fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

# Define and train the model
est_net = MLPRegressor(learning_rate="adaptive", max_iter=400, hidden_layer_sizes=(150, ))
est_net = est_net.fit(X_train_s, y_train)


# In[ ]:


# Scale the test data and generate benchmark
y_train_predict = est_net.predict(X_train_s)
train_predict = mean_squared_error(y_train, y_train_predict)
y_test_predict = est_net.predict(X_test_s)
test_predict = mean_squared_error(y_test, y_test_predict)
print("Train mse:", train_predict, "; Validation mse:", test_predict)

sns.residplot(est_net.predict(X_test_s), y_test);


#  Although here only appears a single model, I have tried changing some parameters to try to improve the results, with not much difference.This fit into the good models, but still far from the decision trees.
#  
#  
# # Final model selecction
#  
# Random forest is selected as the final model. Let's make more plots to ensure analize the quality of the solution.

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
est = RandomForestRegressor(random_state=1337, min_samples_split=60, n_estimators=300, n_jobs=-1)              
est = est.fit(X_train, y_train)
get_scores(est)
plot_residuals(est)


# In[ ]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
y_test_predict = est.predict(X_test)

mae = mean_absolute_error(y_test, y_test_predict)
mse = mean_squared_error(y_test, y_test_predict)
r_2 = r2_score(y_test, y_test_predict)
print("MSE -> ", mse)
print("MAE -> ", mae)
print("R^2 -> ", r_2)


# # Evaluation
# 
# From the last residual plot we clearly see there is a trend, so the final model have miss some
# important relation. This is normal, since the features set I limit this project is quite small. Despite
# that we can see at our benchmark metric as well as some additional ones.
# 
# The determination coefficient tell us how well a model fits some data, with a maximum value of 1,
# 0.93 is a very good result.
# 
# With the mean error value, we can say that in average our model with differs with the real cost of
# about $1.5, which for the use case is low enough.
# 
# Finally, our score MSE compared with the MAE indicates that may be a subset of point which our
# model does not fit very well (we can also see that in the residual plot). Nevertheless, 9.3 is good for
# our problem.

# # Future work
# 
# A typical approach to improve a model is to use more data. In this case I only use the data of May,
# including other months could improve the overall score.
# 
# In our case however, we've identified on the residual plot that an important relation is missing.
# Thus, introducing new variables like the weather, holidays or especial events could minimize this
# problem. Considering this I think this model can improve.
# 
# Finally, we could try different algorithms, like a complex neural network, although this may implied
# other problems and is a bit overkill.
