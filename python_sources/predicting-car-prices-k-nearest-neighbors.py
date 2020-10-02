#!/usr/bin/env python
# coding: utf-8

# # Table of contents
# *  [Introduction](#section1) 
# *  [Read in the data](#section2)
# *  [Cleaning](#section3)
#     - [Numeric vs. non-numeric columns](#section4)
#     - [Missing values](#section5)
#     - [Normalization](#section6)
# *  [Univariate model](#section7)
#     - [Definition](#section8)
#     - [Testing](#section9)
#     - [Results](#section10)
# *  [Multivariate model](#section11)
#     - [Definition](#section12)
#     - [Top 5 features](#section13)
#     - [Testing](#section14)
#         - [Feature selection](#section15)
#         - [Hyperparameter tuning](#section16)
#     - [Results](#section17)
# *  [Cross validation](#section18) 
#     - [Feature selection](#section19)
#     - [Hyperparameter tuning](#section20)
#     - [Results](#section21)
#     
#     by @samaxtech

# ---
# <a id='section1'></a>
# # Introduction
# This project aims to predict car prices using different K-Nearest Neighbors models. The data is sourced from https://archive.ics.uci.edu/ml/datasets/automobile.

# In[ ]:


import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, KFold

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# <a id='section2'></a>
# # Read in the data

# In[ ]:


cols = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 'drive-wheels', 'engine-location', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type', 'num-of-cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']
cars = pd.read_csv('../input/imports-85.data.txt', names=cols)
print(cars.shape)
cars.head()


# In[ ]:


cars.describe()


# <a id='section3'></a>
# # Cleaning

# ----
#  <a id='section4'></a>
# ## Numeric vs. non-numeric columns
# Before selecting the features to use for the model, let's see which ones are numeric. 
# 
# In this case, referring to the Attribute Information of the dataset, found at https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.names, and only selecting numeric columns with continuous values results in the most effective way to achieve this.

# In[ ]:


continuous_numeric = ['normalized-losses', 'wheel-base', 'length', 'width', 
                      'height', 'curb-weight', 'bore', 'stroke', 'compression-ratio', 
                      'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']

numeric_cars = cars[continuous_numeric].copy()
numeric_cars.head()


# <a id='section5'></a>
# ## Missing values

# In[ ]:


numeric_cars.isnull().sum()


# While there's not any NULL values in the cars dataframe, the 'normalized-losses' column contains 41 missing values, symbolized by a question mark '?', as seen below.

# In[ ]:


numeric_cars['normalized-losses'].value_counts()


# Let's replace any question mark in the data with the numpy.nan missing value.

# In[ ]:


numeric_cars.replace('?', np.nan, inplace=True)
print("\nMissing values before: \n\n", numeric_cars.isnull().sum(), "\n\n")


# Any column that now has NaN values on it, before containg question marks, which made pandas cast it to the object data types, as seen below.

# In[ ]:


numeric_cars.dtypes


# Let's convert those to numeric types, since they all contain numeric data values.

# In[ ]:


to_numeric_cols = ['normalized-losses', 'bore', 'stroke', 'horsepower', 'peak-rpm', 'price']
numeric_cars[to_numeric_cols] = numeric_cars[to_numeric_cols].astype(float)
numeric_cars.dtypes


# The dataset has 205 rows, and we've seen how there're up to 41 NaN values. This means handling those by removing any row where there's a NaN value would result in losing close to 25% of the data, which is not a good solution.
# 
# Let's only apply that to any row that has more than one missing value, and handle the rest by replacing any NaN value with the average of that column.

# In[ ]:


numeric_cars.dropna(axis=0, thresh=2, inplace=True)
numeric_cars = numeric_cars.fillna(numeric_cars.mean())
print("\nMissing values after: \n\n", numeric_cars.isnull().sum(), "\n")


# <a id='section6'></a>
# ## Normalization
# Normalizing the numeric values using min-max normalization will make all values range from 0 to 1. This will prevent outliers when measuring squared erros.
# 
# Let's apply that to all columns except for the target column, 'price'.

# In[ ]:


normalized_cars = (numeric_cars - numeric_cars.min())/(numeric_cars.max() - numeric_cars.min())
#normalized_cars = np.abs((numeric_cars - numeric_cars.mean())/numeric_cars.std())
normalized_cars['price'] = numeric_cars['price']
print(normalized_cars.shape)
normalized_cars.head()


# <a id='section7'></a>
# # Univariate model

# The univariate model will use test/train validation, taking a single column as the selected feature, split the dataset into a training and test set, train and make predictions, returning the RMSE for the model.
# 
# It takes the training column name, target column name, the dataframe object, and a parameter for the _k_ value.
# 
# In this case, we'll consider splitting the dataset so that 50% of the rows represent the training set and the remaining 50% represent the test set.
# 
# Since we want to predict car prices, we will use the 'price' column as the target column for the model.

# <a id='section8'></a>
# ## Definition

# In[ ]:


# Univariate model
def knn_train_test_uni(feature, target_column, df, k):
    
    # Randomize order of rows in data frame.
    np.random.seed(1)
    shuffled_index = np.random.permutation(df.index)
    rand_df = df.reindex(shuffled_index)

    # Split the dataset
    train_set = rand_df.iloc[0:int(len(rand_df)/2)]
    test_set = rand_df.iloc[int(len(rand_df)/2):]
    
    # Train
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(train_set[[feature]], train_set[target_column])
    
    # Predict
    predictions = knn.predict(test_set[[feature]])
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(test_set[target_column], predictions))
    
    return rmse


# <a id='section9'></a>
# ## Testing
# Let's test different models by changing:
# 
# - The column used as feature: every numeric column (as previously stored in 'continuous_numeric').
# - The _k_ value: 1, 3, 5, 7 and 9.
# 
# For every numeric column, we'll try all k values, store and plot the RMSE results.

# In[ ]:


k_values = [1, 3, 5, 7, 9]
rmse_uni = {}
current_rmse = []
target_column = 'price'

for feature in continuous_numeric[0:-1]:
    for k in k_values:
        current_rmse.append(knn_train_test_uni(feature, target_column, normalized_cars, k))
        
    rmse_uni[feature] = current_rmse
    current_rmse = []

rmse_uni


# <a id='section10'></a>
# ## Results

# In[ ]:


fig, ax = plt.subplots(1)

for key, values in rmse_uni.items():
    ax.plot(k_values, values, label=key)
    ax.set_xlabel('k value')
    ax.set_ylabel('RMSE')
    ax.set_title('RMSE for Each Training Column\nvs. k value')
    ax.tick_params(top="off", left="off", right="off", bottom='off')
    ax.legend(bbox_to_anchor=(1.5, 1), prop={'size': 11})
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)


# We can see how the RMSE values range from 4,000 up to 11,000 dollars. 
# 
# Later on in this project, I will select the top 5 features based on the average of all RMSE values for each _k_ value, to be used to test different multivariate models, defined in the next section.

# <a id='section11'></a>
# # Multivariate model
# The multivariate model will perform the exact same way as the univariate, but will take a list of column names to be used as features.

# <a id='section12'></a>
# ## Definition

# In[ ]:


# Multivariate model
def knn_train_test(features, target_column, df, k):
    
    # Randomize order of rows in data frame.
    np.random.seed(1)
    shuffled_index = np.random.permutation(df.index)
    rand_df = df.reindex(shuffled_index)

    # Split the dataset
    train_set = rand_df.iloc[0:int(len(rand_df)/2)]
    test_set = rand_df.iloc[int(len(rand_df)/2):]
    
    # Train
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(train_set[features], train_set[target_column])
    
    # Predict
    predictions = knn.predict(test_set[features])
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(test_set[target_column], predictions))
    
    return rmse


# <a id='section13'></a>
# ## Top 5 features
# In order to effectively test different set of features for the multivariate model, using sets of the top 5 from the univariate model could be a good approach.
# 
# We can achieve this by computing the average of all RMSE values got for each _k_ value, and assign that to every column that's been used.

# In[ ]:


avg_rmse = {}

for key, values in rmse_uni.items():
    avg_rmse[key] = np.mean(values)

avg_rmse = pd.Series(avg_rmse)
avg_rmse.sort_values()


# The top 5 features using this method are 'highway-mpg', 'curb-weight', 'horsepower', 'width' and 'city-mpg'.

# <a id='section14'></a>
# ## Testing
# <a id='section15'></a>
# ### Feature selection
# To test different multivariate models, I will select the best 2, 3, 4, 5 and lastly 6 best features of the previous univariate 'RMSE ranking', and see which set of features performs best for the default _k_ value.

# In[ ]:


features = {
        'best_2': ['highway-mpg', 'curb-weight'],
        'best_3': ['highway-mpg', 'curb-weight', 'horsepower'],
        'best_4': ['highway-mpg', 'curb-weight', 'horsepower', 'width'],
        'best_5': ['highway-mpg', 'curb-weight', 'horsepower', 'width', 'city-mpg'],
        'best_6': ['highway-mpg', 'curb-weight', 'horsepower', 'width', 'city-mpg', 'length']
    } 

rmse_multi = {}
target_column = 'price'
k = 5

for key, value in features.items():
    rmse_multi[key] = knn_train_test(value, target_column, normalized_cars, k)
    
pd.Series(rmse_multi).sort_values()


# <a id='section16'></a>
# ### Hyperparameter tuning
# From the top 3 models in the last section (those using 'best_6', 'best_2', and 'best_3' as features), let's see how they perform when tuning the _k_ value from 1 to 25.

# In[ ]:


top_models = {
        'best_2': ['highway-mpg', 'curb-weight'],
        'best_3': ['highway-mpg', 'curb-weight', 'horsepower'],
        'best_6': ['highway-mpg', 'curb-weight', 'horsepower', 'width', 'city-mpg', 'length']
    } 

k_values = list(range(1, 26))
rmse_multi_k = {}
rmse_current = []

for key, value in top_models.items():
    for k in k_values:
        rmse_current.append(knn_train_test(value, target_column, normalized_cars, k))
        
    rmse_multi_k[key] = rmse_current
    rmse_current = []
    
print(rmse_multi_k)


# <a id='section17'></a>
# ## Results

# In[ ]:


# Returns a dict with the min value of every key's list and its index the list
def min_key_value(dictionary):
    min_values = {}
    for k, v in dictionary.items():
        min_values[k] = [min(v), v.index(min(v))]
        
    return min_values
        
best_k = min_key_value(rmse_multi_k)
print(best_k)

# Plot results
fig, ax = plt.subplots(1)

for key, values in rmse_multi_k.items():
    ax.plot(k_values, values, label=key)
    ax.set_xlabel('k value')
    ax.set_ylabel('RMSE')
    ax.set_title('RMSE for Top 3 Models vs. k value\n Test/Train Validation')
    ax.tick_params(top="off", left="off", right="off", bottom='off')
    ax.legend(bbox_to_anchor=(1.5, 1), prop={'size': 11})
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)


# The best results for each model are:
# 
# - 'best_6': RMSE of **3303.16** dollars, for **k=1**
# - 'best_3': RMSE of **3382.88** dollars, for **k=2**
# - 'best_2': RMSE of **3802.87** dollars, for **k=6**
# 
# All of them tend to perform worse as k increases from a certain point (between k=5 and k=10 approximately). 
# 
# It also seems like the more features, the lower _k_ value that performs best. This may be because more features make entries (cars in this case) more unique, which means if a new car has a lot of attributes or features that make it distinct, it will be harder for it to be similar to a big number of cars in our training set (i.e. selecting a big _k_ when using many features may result in worse predictions).
# 
# As an example, that may be why the case of _k=1_ for the 'best_6' model (6 attributes/features) performs best, since even though every new car only chooses one similar/close in distance car (neighbor) from the training set, that one car is the most similar cosindering a big number of attributes, which results in a a good prediction.

# <a id='section18'></a>
# # Cross validation
# Lastly, for the multivariate model let's modify the knn_train_test() function to use k-fold cross validation instead of test/train validation and see how it performs for 10 folds.

# In[ ]:


def knn_cross_validation(features, target_column, df, k): 
    knn = KNeighborsRegressor(n_neighbors=k)
    kf = KFold(n_splits=10, shuffle=True, random_state=1)
    mses = cross_val_score(knn, df[features], df[target_column], scoring='neg_mean_squared_error', cv=kf)
    avg_rmse = np.mean(np.sqrt(np.absolute(mses)))
    
    return avg_rmse


# <a id='section19'></a>
# ## Feature selection
# Proceeding the exact same way as before, using the same selected features, the RMSE values returned are:

# In[ ]:


features = {
        'best_2': ['highway-mpg', 'curb-weight'],
        'best_3': ['highway-mpg', 'curb-weight', 'horsepower'],
        'best_4': ['highway-mpg', 'curb-weight', 'horsepower', 'width'],
        'best_5': ['highway-mpg', 'curb-weight', 'horsepower', 'width', 'city-mpg'],
        'best_6': ['highway-mpg', 'curb-weight', 'horsepower', 'width', 'city-mpg', 'length']
    } 

rmse_multi = {}
target_column = 'price'
k = 5

for key, value in features.items():
    rmse_multi[key] = knn_cross_validation(value, target_column, normalized_cars, k)
    
pd.Series(rmse_multi).sort_values()


# This time, the 3 best models in terms of RMSE were 'best_4', 'best_3', and 'best_5'. 
# 
# Performing hyperparameter tuning will tell us what the optimal _k_ value is for each of them, just as we did before.

# <a id='section20'></a>
# ## Hyperparameter tuning

# In[ ]:


top_models = {
        'best_3': ['highway-mpg', 'curb-weight', 'horsepower'],
        'best_4': ['highway-mpg', 'curb-weight', 'horsepower', 'width'],
        'best_5': ['highway-mpg', 'curb-weight', 'horsepower', 'width', 'city-mpg']
    } 

k_values = list(range(1, 26))
rmse_multi_k = {}
rmse_current = []

for key, value in top_models.items():
    for k in k_values:
        rmse_current.append(knn_cross_validation(value, target_column, normalized_cars, k))
        
    rmse_multi_k[key] = rmse_current
    rmse_current = []
    
print(rmse_multi_k)


# <a id='section21'></a>
# ## Results

# In[ ]:


best_k = min_key_value(rmse_multi_k)
print(best_k)

# Plot results
fig, ax = plt.subplots(1)

for key, values in rmse_multi_k.items():
    ax.plot(k_values, values, label=key)
    ax.set_xlabel('k value')
    ax.set_ylabel('RMSE')
    ax.set_title('RMSE for Top 3 Models vs. k value\n 10-Fold Cross Validation')
    ax.tick_params(top="off", left="off", right="off", bottom='off')
    ax.legend(bbox_to_anchor=(1.5, 1), prop={'size': 11})
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)


# Using k-fold cross validation, the trend is similar to what we got before using test/train validation. However, this time we got better results:
# 
# - 'best_3': RMSE of **2824.09** dollars, for **k=2**
# - 'best_4': RMSE of **3035.18** dollars, for **k=3**
# - 'best_5': RMSE of **3159.23** dollars, for **k=3**

# In[ ]:




