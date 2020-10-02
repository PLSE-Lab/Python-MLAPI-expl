#!/usr/bin/env python
# coding: utf-8

# ## Exercise notebook for the third session (30 min)
# 
# This is the exercise notebook for the third session of the [Machine Learning workshop series at Harvey Mudd College](http://www.aashitak.com/ML-Workshops/). Please feel free to ask for help from the instructor and/or TAs.

# First we import python modules:

# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.preprocessing import PolynomialFeatures

import warnings
warnings.simplefilter('ignore')


# Now we will tackle the [Bike Sharing Demand](https://www.kaggle.com/c/bike-sharing-demand/overview) dataset

# In[2]:


path = '../input/'
rides = pd.read_csv(path + 'train.csv')
rides.head()


# **Data Fields**
# 
# * datetime - hourly date + timestamp    
# * season -  1 = spring, 2 = summer, 3 = fall, 4 = winter   
# * holiday - whether the day is considered a holiday  
# * workingday - whether the day is neither a weekend nor holiday  
# * weather -   
#     1: Clear, Few clouds, Partly cloudy, Partly cloudy   
#     2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist   
#     3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds   
#     4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog   
# * temp - temperature in Celsius  
# * atemp - "feels like" temperature in Celsius  
# * humidity - relative humidity  
# * windspeed - wind speed  
# * casual - number of non-registered user rentals initiated  
# * registered - number of registered user rentals initiated  
# * count - number of total rentals  

# Let us look at the *datetime* values.

# In[ ]:


rides['datetime'].values[:5]


# Now we perform some feature engineering and data pre-processing similar to what we practised in the previous two sessions. 

# In[ ]:


from datetime import datetime

# We extract 'month', 'hour', 'weekday' from the 'datetime' column
def extract_from_datetime(rides):
    rides["date"] = rides["datetime"].apply(lambda x : x.split()[0])
    rides["hour"] = rides["datetime"].apply(lambda x : x.split()[1].split(":")[0])
    rides["weekday"] = rides["date"].apply(lambda dateString : 
                            datetime.strptime(dateString,"%Y-%m-%d").weekday())
    rides["month"] = rides["date"].apply(lambda dateString : 
                            datetime.strptime(dateString,"%Y-%m-%d").month)
    return rides

# We one-hot encode the categorical features
def one_hot_encoding(rides):
    dummy_fields = ['season', 'weather', 'month', 'hour', 'weekday']
    for each in dummy_fields:
        dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
        rides = pd.concat([rides, dummies], axis=1)
    return rides

# We drop the columns that are redundant now
def drop_features(rides):
    features_to_drop = ['datetime', 'date', 
                        'month', 'hour', 'weekday', 
                        'season', 'weather']

    rides = rides.drop(features_to_drop, axis=1)
    return rides

# Now we aggregate all the above defined functions inside a function
def feature_engineering(rides):
    rides = extract_from_datetime(rides)
    rides = one_hot_encoding(rides)
    rides = drop_features(rides)
    return rides

# Now we apply all the above defined functions to the rides dataframe
rides = feature_engineering(rides)


# The reason we defined all the steps as functions and bundled them into another function `feature_engineering` is so as to reuse the code for processing the data from `test.csv` file for which we will make predictions at the end.

# In[ ]:


rides.head()


# In[ ]:


rides.columns


# In[ ]:


rides.shape


# For all algorithms using gradient descent for minimizing the cost function, normalizing features helps speed up the learning process. This is because otherwise the features with values higher in magnitudes will dominate the updates. See [here](https://www.coursera.org/lecture/machine-learning/gradient-descent-in-practice-i-feature-scaling-xx3Da) and [here](https://gist.github.com/oskarth/3469833). We substract the qunatitative features by their mean and divide by their standard deviation to redistribute them to have mean 0 and standard deviation 1. 
# $$ x' = \frac{x - \mu}{\sigma} $$
# ![](https://www.jeremyjordan.me/content/images/2018/01/Screen-Shot-2018-01-23-at-2.27.20-PM.png)

# In[ ]:


quantitative_features = ['temp', 'atemp', 'humidity', 'windspeed']

# Store scalings in a dictionary so we can convert back later
scaled_features = {}
for each in quantitative_features:
    mean, std = rides[each].mean(), rides[each].std()
    scaled_features[each] = [mean, std]
    rides.loc[:, each] = (rides[each] - mean)/std


# In[ ]:


# Next we extract the target variables from the dataframe
target = rides[['casual', 'registered', 'count']]
target = np.log1p(target)
rides = rides.drop(['casual', 'registered', 'count'], axis=1)


# First we split the data into training and validation set.

# In[ ]:


X_train, X_valid, y_train, y_valid = train_test_split(rides, target,
                                        random_state = 0)


# Train a linear regression model using the training set and calculate the $R^2$ score for both training and validation set.

# In[ ]:





# You should get the following $R^2$ values for the training and validation set.  
# `R-squared score (training): 0.641
# R-squared score (validation): 0.625`

# Let us try polynomial regression with degree 2. First we get polynomial features.

# In[ ]:


poly2 = PolynomialFeatures(degree=2)
X_poly2 = poly2.fit_transform(rides)
X_train_poly2, X_valid_poly2, y_train_poly2, y_valid_poly2 = train_test_split(X_poly2, 
                                                    target['count'], random_state = 0)


# Train a polynomial regression model using [`LinearRegression()`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) on polynomial features and call it `polyreg2`. 

# In[ ]:





# Train a polynomial regression coupled with Ridge and call it `polyreg2_ridge`. Tune the regularization parameter alpha.

# In[ ]:





# Train a polynomial regression coupled with Lasso and call it `polyreg2_lasso`. Tune the regularization parameter alpha.

# In[ ]:





# Now let us try polynomial regression with degree 3.

# In[ ]:


poly3 = PolynomialFeatures(degree=3)
X_poly3 = poly3.fit_transform(rides)
X_train_poly3, X_valid_poly3, y_train_poly3, y_valid_poly3 = train_test_split(X_poly3, 
                                                    target['count'], random_state = 0)


# In[ ]:


polyreg3 = LinearRegression().fit(X_train_poly3, y_train_poly3)

polyreg3_train_score = polyreg3.score(X_train_poly3, y_train_poly3)
polyreg3_valid_score = polyreg3.score(X_valid_poly3, y_valid_poly3)

print('R-squared score (training): {:.3f}'
     .format(polyreg3_train_score))
print('R-squared score (validation): {:.3f}'
     .format(polyreg3_valid_score))


# This suggests the model has overfitted to the training set excessively. Nonetheless, the very high $R^2$ looks promising, so we use regularization on the the polynomial regression with degree 3 features. 

# Train the polynomial regression for degree 3 coupled with Ridge and call it `polyreg3_ridge`. Tune the regularization parameter alpha starting with a not so high value, say 10.

# In[ ]:





# Train the polynomial regression for degree 3 coupled with Lasso and call it `polyreg3_lasso`. Try a few differnt values for the regularization parameter alpha.

# In[ ]:





# Using the following function for the root mean-squared error (RMSE), compare the different regression models, preferably by plotting a graph. Similarly, plot a graph to compare the $R^2$ scores as well.

# In[ ]:


def get_rmse(reg):
    y_pred_train = reg.predict(X_train_poly)
    train_rmse = np.sqrt(mean_squared_error(y_train_poly, y_pred_train))
    y_pred_valid = reg.predict(X_valid_poly)
    valid_rmse = np.sqrt(mean_squared_error(y_valid_poly, y_pred_valid))
    return train_rmse, valid_rmse


# Next steps:
# * Read the test.csv file into a dataframe
# * Feature engineer the dataframe in exactly the same way as above by using function `feature_engineering`.
# * Scale the quantitative variables the same way as above
# * Train a model
# * Predict
# * Convert the predictions using exponential (since our model is built using log for the target variable)
# * Create a dataframe for the results with the right format
# * Save it into csv file and submit

# ### Acknowledgement:
# 
# The credits for the images used in the above are as follows.
# - Image 1: https://commons.wikimedia.org/wiki/File:Gaussian_kernel_regression.png
# 
# For the feature engineering of the data, inspiration is taken from the following two publically shared sources:
# * Udacity Github Repository: https://github.com/udacity/deep-learning-v2-pytorch/tree/master/project-bikesharing
# * Kaggle kernel by Vivek Srinivasan: https://www.kaggle.com/viveksrinivasan/eda-ensemble-model-top-10-percentile
# 

# In[ ]:




