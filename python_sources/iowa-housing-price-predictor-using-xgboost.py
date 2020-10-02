#!/usr/bin/env python
# coding: utf-8

# # Using XGBoost
# XGBoost is an implementation of the **Gradient Boosted Decision Trees algorithm**

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
import warnings

warnings.simplefilter(action='ignore', category=DeprecationWarning) 
#so pandas doesn't spit out a warning everytime

# Loading in Iowa housing data
data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
data.dropna(axis=0, subset=['SalePrice'], inplace=True) #drops data with missing SalePrice value
y = data.SalePrice
X = data.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object'])
# this is the path to the Iowa data we will use
train_X, test_X, train_y, test_y = train_test_split(X.as_matrix(), y.as_matrix(), test_size=0.25)

# now we create our imputer
my_imputer = Imputer()
train_X = my_imputer.fit_transform(train_X)
test_X = my_imputer.fit_transform(test_X)

Now we are going to fit a model just like we would in sckikit-learn.
# In[ ]:


from xgboost import XGBRegressor

my_model = XGBRegressor()
# we can add silent=True to avoid printing out updates with each cycle
my_model.fit(train_X, train_y, verbose=False)


# Now we will evaluate our model by making a prediction and see what the MAE is, just like in scikit.

# In[ ]:


# make predictions
predictions = my_model.predict(test_X)

from sklearn.metrics import mean_absolute_error
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))


# # Model Tuning
# 
# XGBoost has a few parameters that can dramatically affect your model's accuracy and training speed. Now that we have our baseline, we can tune the parameters to make our model better.

# In[ ]:


my_model = XGBRegressor(n_estimator = 1000)
my_model.fit(train_X, train_y, early_stopping_rounds = 5, eval_set = [(test_X, test_y)], verbose = False)


# Now we'll add a small learning rate to hopefully increase the accuracy of our model, although this will also increase the time it takes to train our model which may be problematic when we encounter larger datasets.

# In[ ]:


my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
my_model.fit(train_X, train_y, early_stopping_rounds=100, 
             eval_set=[(test_X, test_y)], verbose=False)

# make our final predictions
predictions = my_model.predict(test_X)
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))


# 

# 

# 
