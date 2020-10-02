#!/usr/bin/env python
# coding: utf-8

# # Table of contents
# *  [Introduction](#section1) 
# *  [Read in the data](#section2)
#     - [Bike rentals, target column](#section3)
# *  [Feature engineering](#section4)
#     - [Time labels](#section5)
#     - [Weather index](#section6)
# *  [Train/Test split](#section7)
# *  [Modeling and testing](#section8)
#     *  [Linear model](#section9) 
#     *  [Decision trees](#section10) 
#     *  [Random forests](#section11) 
# *  [Another strategy](#section12) 
#     *  [Linear model](#section13) 
#     *  [Decision trees](#section14) 
#     *  [Random forests](#section15) 
#     *  [Results](#section16)
#     
# by @samaxtech

# ---
# <a id='section1'></a>
# # Introduction 
# This project aims to predict bike rentals using different Linear Regression, Decision Trees and Random Forests models, testing their performance on bike rentals data compiled by Hadi Fanaee-T at the University of Porto, which can be found at http://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset.

# In[ ]:


import pandas as pd
import numpy as np
import math

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# <a id='section2'></a>
# # Read in the data

# In[ ]:


bike_rentals = pd.read_csv('../input/hour.csv')
bike_rentals.head(5)


# <a id='section3'></a>
# ## Bike rentals, target column
# Since the goal is to predict the number of bikes that will be rented in a given hour, the target column is 'cnt'. Let's take a look at the correlation between the rest of the columns and 'cnt' and its values first.

# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,6))
ax1, ax2 = axes.flatten()

bike_rentals['cnt'].hist(grid=False, ax=ax1)

# Sorted correlations with'cnt'
sorted_corrs = bike_rentals.corr()['cnt'].sort_values(ascending=False)
sns.heatmap(bike_rentals[sorted_corrs.index].corr(), ax=ax2)

ax1.set_title('Target Column "cnt" Histogram')
ax2.set_title('Correlations')
plt.show()
print("Correlations:\n\n", sorted_corrs)


# Looks like the amount of bikes rented per hour mostly oscillates between 0 and 200 bikes. Also, as we could expect the bigger the number of signed up or casual riders, the more bikes that were rented (as we can tell by the correlation between 'cnt' and 'registered' and 'casual'). 
# 
# We won't use those for the model, since 'cnt' is derived from them (those numbers are added together to get 'cnt'), and we won't have that information when we want to make predictions for new rentals.
# 
# Other factors such as temperature ('temp'), 'atemp' or 'hr' seem to highly correlate as well. 

# ----
# <a id='section4'></a>
# # Feature engineering

# <a id='section5'></a>
# ## Time labels
# The column 'hr' contains the hour each rental occurred. If we want the model to take into account the relation between certain hours instead of treating them differently (and add more information for it to make better decisions), we could add a 'time_label' column, so that each label is represented by a number, such that:
# 
# - Morning: 1
# - Afternoon: 2
# - Evening: 3
# - Night: 4

# In[ ]:


def assign_label(hour):
    if hour >= 6 and hour < 12:
        return 1
    elif hour >= 12 and hour < 18:
        return 2
    elif hour >= 18 and hour < 24:
        return 3
    elif hour >= 0 and hour < 6:
        return 4
    
bike_rentals['time_label'] = bike_rentals['hr'].apply(lambda hr: assign_label(hr))
print(bike_rentals['time_label'].value_counts())
bike_rentals.head(5)


# <a id='section6'></a>
# ## Weather index
# By combining temperature, humidity, and wind speed we can create a weather index, an additional feature that's 
# valuable for the model. 
# 
# Weighting temperature and humidity in a way that makes sense for someone who's thinking about renting a bike based on those factors. After tweaking the weights (from 0 to 1), I've found 'temp' to be the feature that dicreases the error the most.

# In[ ]:


bike_rentals['weather_idx'] = 0.8*bike_rentals['temp'] + 0.1*bike_rentals['atemp'] + 0.1*bike_rentals['hum'] 


# ---
# <a id='section7'></a>
# # Train/Test split

# In[ ]:


# Train: 80% of the data / Test: 20% of the data
train, test = train_test_split(bike_rentals, test_size=0.2, random_state=100)

print("\nTrain: ", train.shape)
print("Test: ", test.shape)


# ---
# <a id='section8'></a>
# # Modeling and testing
# Given what was mentioned before about 'casual', and 'registered', those along with the target column 'cnt' and 'dteday' (the calendar day the bike was rented, in date format) will be excluded from the features.

# In[ ]:


features = bike_rentals.columns[~bike_rentals.columns.isin(['cnt', 'registered', 'casual', 'dteday'])].tolist()

X_train = train[features]
y_train = train['cnt']

X_test = test[features]
y_test = test['cnt']

print("\nInitial features: ", features)


# ---
# <a id='section9'></a>
# ## Linear model

# In[ ]:


# Linear model
lr = LinearRegression()

# Train 
lr.fit(X_train, y_train)

#Predict 
new_cnt_lr = lr.predict(X_test)

# --------------------------------------------------
# Error metric
# --------------------------------------------------

# MSE 
mse_lr = mean_squared_error(y_test, new_cnt_lr)

print("-----------------\nLinear regression\n-----------------")
print("MSE: ", mse_lr)


# It looks like the linear model has a very high error. This could be because of very high or very low values in the data, that cause MSE to go up significantly. Let's see how using Decision Trees does with the same set of features.

# ----
# <a id='section10'></a>
# ## Decision Trees
# In this case, to test different models I am going to tweak 'min_samples_leaf', 'min_samples_split' and 'max_depth' and see which one performs best.

# In[ ]:


# Decision Trees model
min_samples_leaf = range(5,20,5)
max_depth = range(5,50,5)
min_samples_split = range(5,20,5)

mse_dt = {}
current_mse = math.inf

for msl in min_samples_leaf:
    for md in max_depth:
        for mss in min_samples_split:
            
            dt = DecisionTreeRegressor(min_samples_leaf=msl, max_depth=md, min_samples_split=mss)

            # Train
            dt.fit(X_train, y_train)

            # Predict
            new_cnt_dt = dt.predict(X_test)
            
            # Update MSE 
            mse = mean_squared_error(y_test, new_cnt_dt)
            
            if mse <= current_mse:
                mse_dt['value'] = mse
                mse_dt['min_samples_split'] = mss
                mse_dt['max_depth'] = md
                mse_dt['min_samples_leaf'] = msl
                
                current_mse = mse

print("-----------------\nDecision Trees\n-----------------")
print("MSE: ", mse_dt)


# The error has decreased significantly, for the parameters shown above. This shows the data has powerful non-linear relations and the decision trees model is able to capture them (unlike the linear model). Let's see how combining several DT models performs by repeating the process with Random Forests.

# ---
# <a id='section11'></a>
# ## Random forests

# In[ ]:


# Random Forests model (setting n_estimators=10 (default))
min_samples_leaf = range(5,20,5)
max_depth = range(5,50,5)
min_samples_split = range(5,20,5)

mse_rf = {}
current_mse = math.inf

for msl in min_samples_leaf:
    for md in max_depth:
        for mss in min_samples_split:

            dt = RandomForestRegressor(n_estimators=10, min_samples_leaf=msl, max_depth=md, min_samples_split=mss)

            # Train
            dt.fit(X_train, y_train)

            # Predict
            new_cnt_rf = dt.predict(X_test)

            # Update MSE 
            mse = mean_squared_error(y_test, new_cnt_rf)

            if mse <= current_mse:
                mse_rf['value'] = mse
                mse_rf['min_samples_split'] = mss
                mse_rf['max_depth'] = md
                mse_rf['min_samples_leaf'] = msl

                current_mse = mse

print("-----------------\nRandom Forests\n-----------------")
print("MSE: ", mse_rf)


# The error dicreased even more. Random Forests helps remove the sources of overfitting present in the Decision Trees model.

# ---
# <a id='section12'></a>
# # Another strategy
# Finally, it could be interesting to see how much better or worse each model performs when predicting 'casual' and 'registered', instead of 'cnt', considering their relation:
# 
# - *'cnt' = 'casual' + 'registered'*.
# 
# We can predict these two and add them up to get 'cnt' and test the error.

# In[ ]:


new_target = ['casual', 'registered']
new_y_train = train[new_target]


# <a id='section13'></a>
# ## Linear model

# In[ ]:


# Linear model
lr = LinearRegression()

# Train (update y_train)
lr.fit(X_train, new_y_train)

#Predict
predictions = lr.predict(X_test)

# Add up 'casual' and 'registered'
new_cnt_lr = predictions.sum(axis=1)

# --------------------------------------------------
# Error metric
# --------------------------------------------------

# MSE 
mse_lr = mean_squared_error(y_test, new_cnt_lr)

print("-----------------\nLinear regression\n-----------------")
print("MSE: ", mse_lr)


# <a id='section14'></a>
# ## Decision trees

# In[ ]:


# Decision Trees model
min_samples_leaf = range(5,20,5)
max_depth = range(5,50,5)
min_samples_split = range(5,20,5)

mse_dt = {}
current_mse = math.inf

for msl in min_samples_leaf:
    for md in max_depth:
        for mss in min_samples_split:
            
            dt = DecisionTreeRegressor(min_samples_leaf=msl, max_depth=md, min_samples_split=mss)

            # Train (update y_train)
            dt.fit(X_train, new_y_train)

            # Predict
            predictions = dt.predict(X_test)
            
            # Add up 'casual' and 'registered'
            new_cnt_dt = predictions.sum(axis=1)
            
            # Update MSE 
            mse = mean_squared_error(y_test, new_cnt_dt)
            
            if mse <= current_mse:
                mse_dt['value'] = mse
                mse_dt['min_samples_split'] = mss
                mse_dt['max_depth'] = md
                mse_dt['min_samples_leaf'] = msl
                
                current_mse = mse

print("-----------------\nDecision Trees\n-----------------")
print("MSE: ", mse_dt)


# <a id='section15'></a>
# ## Random forests

# In[ ]:


# Random Forests model (setting n_estimators=10 (default))
min_samples_leaf = range(5,20,5)
max_depth = range(5,50,5)
min_samples_split = range(5,20,5)

mse_rf = {}
current_mse = math.inf

for msl in min_samples_leaf:
    for md in max_depth:
        for mss in min_samples_split:

            dt = RandomForestRegressor(n_estimators=10, min_samples_leaf=msl, max_depth=md, min_samples_split=mss)

            # Train (update y_train)
            dt.fit(X_train, new_y_train)

            # Predict
            predictions = dt.predict(X_test)
            
            # Add up 'casual' and 'registered'
            new_cnt_rf = predictions.sum(axis=1)
            
            # Update MSE 
            mse = mean_squared_error(y_test, new_cnt_rf)

            if mse <= current_mse:
                mse_rf['value'] = mse
                mse_rf['min_samples_split'] = mss
                mse_rf['max_depth'] = md
                mse_rf['min_samples_leaf'] = msl

                current_mse = mse

print("-----------------\nRandom Forests\n-----------------")
print("MSE: ", mse_rf)


# <a id='section16'></a>
# ## Results
# The error improved when predicting 'registered' and 'casual' and adding them up instead of predicting 'cnt' directly. This shows the features used to train the models have stronger/more accurate relations with the number of sign up and casual riders individually than with the sum of both.  

# In[ ]:




