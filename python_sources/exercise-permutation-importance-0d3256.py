#!/usr/bin/env python
# coding: utf-8

# # Exercises
# 
# ## Intro
# 
# You will think about and calculate permutation importance with a sample of data from the [Taxi Fare Prediction](https://www.kaggle.com/c/new-york-city-taxi-fare-prediction) competition.
# 
# We won't focus on data exploration or model building for now. You can just run the cell below to 
# - Load the data
# - Divide the data into training and validation
# - Build a model that predicts taxi fares
# - Print a few rows for you to review

# In[ ]:


# Loading data, dividing, modeling and EDA below
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv('../input/new-york-city-taxi-fare-prediction/train.csv', nrows=50000)

# Remove data with extreme outlier coordinates or negative fares
data = data.query('pickup_latitude > 40.7 and pickup_latitude < 40.8 and ' +
                  'dropoff_latitude > 40.7 and dropoff_latitude < 40.8 and ' +
                  'pickup_longitude > -74 and pickup_longitude < -73.9 and ' +
                  'dropoff_longitude > -74 and dropoff_longitude < -73.9 and ' +
                  'fare_amount > 0'
                  )

y = data.fare_amount

base_features = ['pickup_longitude',
                 'pickup_latitude',
                 'dropoff_longitude',
                 'dropoff_latitude',
                 'passenger_count']

X = data[base_features]


train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
first_model = RandomForestRegressor(n_estimators=30, random_state=1).fit(train_X, train_y)

# Environment Set-Up for feedback system.
import sys
sys.path.append('../input/ml-insights-tools')
from learntools.core import binder
binder.bind(globals())
from ex2 import *
print("Setup Complete")

# show data
print("Data sample:")
data.head()


# The following two cells may also be useful to understand the values in the training data:

# In[ ]:


train_X.describe()


# In[ ]:


train_y.describe()


# ## Question 1
# 
# The first model uses the following features
# - pickup_longitude
# - pickup_latitude
# - dropoff_longitude
# - dropoff_latitude
# - passenger_count
# 
# Before running any code... which variables seem potentially useful for predicting taxi fares? Do you think permutation importance will necessarily identify these features as important?
# 
# Once you've thought about it, run `q_1.solution()` below to see how you might think about this before running the code.

# In[ ]:


q_1.solution()


# ## Question 2
# 
# Create a `PermutationImportance` object called `perm` to show the importances from `first_model`.  Fit it with the appropriate data and show the weights.
# 
# For your convenience, the code from the tutorial has been copied into a comment in this code cell.

# In[ ]:


import eli5
from eli5.sklearn import PermutationImportance

# Make a small change to the code below to use in this problem. 
perm = PermutationImportance(first_model, random_state=1).fit(val_X, val_y)

q_2.check()

# uncomment the following line to visualize your results
eli5.show_weights(perm, feature_names = val_X.columns.tolist())


# Uncomment the lines below for a hint or to see the solution.

# In[ ]:


# q_2.hint()
# q_2.solution()


# ## Question 3
# Before seeing these results, we might have expected each of the 4 directional features to be equally important.
# 
# But, on average, the latitude features matter more than the longititude features. Can you come up with any hypotheses for this?
# 
# After you've thought about it, check here for some possible explanations:

# In[ ]:


q_3.solution()


# ## Question 4
# 
# Without detailed knowledge of New York City, it's difficult to rule out most hypotheses about why latitude features matter more than longitude.
# 
# A good next step is to disentangle the effect of being in certain parts of the city from the effect of total distance traveled.  
# 
# The code below creates new features for longitudinal and latitudinal distance. It then builds a model that adds these new features to those you already had.
# 
# Fill in two lines of code to calculate and show the importance weights with this new set of features. As usual, you can uncomment lines below to check your code, see a hint or get the solution.

# In[ ]:


# create new features
data['abs_lon_change'] = abs(data.dropoff_longitude - data.pickup_longitude)
data['abs_lat_change'] = abs(data.dropoff_latitude - data.pickup_latitude)

features_2  = ['pickup_longitude',
               'pickup_latitude',
               'dropoff_longitude',
               'dropoff_latitude',
               'abs_lat_change',
               'abs_lon_change']

X = data[features_2]
new_train_X, new_val_X, new_train_y, new_val_y = train_test_split(X, y, random_state=1)
second_model = RandomForestRegressor(n_estimators=30, random_state=1).fit(new_train_X, new_train_y)

# Create a PermutationImportance object on second_model and fit it to new_val_X and new_val_y
# Use a random_state of 1 for reproducible results that match the expected solution.
perm2 = PermutationImportance(second_model, random_state=1).fit(new_val_X, new_val_y)

# show the weights for the permutation importance you just calculated
eli5.show_weights(perm2, feature_names=new_val_X.columns.tolist())

#q_4.check()


# How would you interpret these importance scores? Distance traveled seems far more important than any location effects. 
# 
# But the location still affects model predictions, and dropoff location now matters slightly more than pickup location. Do you have any hypotheses for why this might be? The techniques used later in the course will help us dive into this more.

# In[ ]:


# q_4.solution()


# ## Question 5
# 
# A colleague observes that the values for `abs_lon_change` and `abs_lat_change` are pretty small (all values are between -0.1 and 0.1), whereas other variables have larger values.  Do you think this could explain why those coordinates had larger permutation importance values in this case?  
# 
# Consider an alternative where you created and used a feature that was 100X as large for these features, and used that larger feature for training and importance calculations. Would this change the outputted permutaiton importance values?
# 
# Why or why not?
# 
# After you have thought about your answer, either try this experiment or look up the answer in the cell below

# In[ ]:


q_5.solution()


# ## Question 6
# 
# You've seen that the feature importance for latitudinal distance is greater than the importance of longitudinal distance. From this, can we conclude whether travelling a fixed latitudinal distance tends to be more expensive than traveling the same longitudinal distance?
# 
# Why or why not? Check your answer below.

# In[ ]:


q_6.solution()


# ## Congrats
# 
# Permutation importance is useful useful for debugging, understanding your model, and communicating a high-level overview from your model.  Next up will be partial dependence plots, which will help you see how individual features affect predictions.
