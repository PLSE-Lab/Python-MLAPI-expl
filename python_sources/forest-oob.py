#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import pprint
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Here comes nothing
# DISCLAIMER: this kernel is my first one, ever! Comments and suggestions are highly appreciated =)
# The algorithm I ended up using is **Random Forest**, because that's the one I learned so far from here https://www.kaggle.com/learn/machine-learning  :p
# 
# 
# 

# In[ ]:


train = pd.read_csv('../input/challenge-data/train.csv', encoding='ISO-8859-1')
test = pd.read_csv('../input/challenge-data/test.csv', encoding='ISO-8859-1')
submission = pd.read_csv('../input/challenge-data/sample_submission.csv', encoding='ISO-8859-1')


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


# Check for negative prices
train.groupby(['unit_price']).size()
train[train.unit_price<0] # seemingly, these negatives are some sort of adjustments of debt


# In[ ]:


## Because there's no negatives in the test-dataset
    # test.groupby(['unit_price']).size()
# Remove negative unit price, in the training-dataset, to make life simpler. 
train = train[train.unit_price>=0]


# In[ ]:


# Start Feature Engineering
# Converting time into integer of minutes, like in TUTORIAL 3
#train['min_hour'] = train['time'].apply(lambda x: x.split(':')[0])
#train['min_minute'] = train['time'].apply(lambda x: x.split(':')[1])
#train['min_hour'] = train['min_hour'].apply(lambda x : int(x) * 60 )
#train['min_minute'] = train['min_minute'].apply(lambda x: int(x))
#train['tot_min'] = train['min_hour'] + train['min_minute']


# ## Filtering out all the unnecessary columns: *date*,*time*,*description*, *min_hour*,*min_minute*, *customer_id* and *invoice_id*, but
# ## I'm keeping: *stock_id*,*country*,*unit_price*, and *tot_min* as  predictors
# Why only include some columns as predictors? Well, in my newbie-way of thinking:
# 1.  Each *country* has a preference for the type of products (*stock_id*) they purchase. E.g., saudi-arabia only buys 2 types of products: glass jar and plasters (DISCLAIMER: the sample size is limited). 
#     Go check! **train[train.country=='saudi arabia']**
# 2. Correspondingly, there is a trend between  *country* and the money they spend (*unit_price*)  E.g., UK and singapore are the two "big-spenders".
#     Go check!  **train.groupby(['country', 'unit_price']).size().reset_index(name='cnt').sort_values(by ='unit_price', ascending=False).head(15)**
# 3. As discussed inTUTORIAL3, there's a possible correlation between time of the day (*tot_min*) and purchases
# 

# In[ ]:


# So, these are the only columns I'll use as predictors
pred_cols = ['unit_price','customer_id','country', 'stock_id']
train_candidates = train[pred_cols]


# Now, technically,* stock_id*, *customer_id*, and *country* are all categoricals, right? So, technically, they should all be one-hot-encoded or something. However, when I tried doing it, only *country* was encoded (as shown below). 
# 

# In[ ]:


# One-hot-encode Country
onehotencoded_candidates = pd.get_dummies(train_candidates)
onehotencoded_candidates.shape


# It turns out, that *stock_id *and *customer_id* are considered as continuous, integers

# In[ ]:


train.info()


# Should I then brute-force these variables (*stock_id* and *customer_id*) into categoricals? Having read some things like https://roamanalytics.com/2016/10/28/are-categorical-variables-getting-lost-in-your-random-forests/, the answer may be **no**
# 
# Why? because as explained here http://rnowling.github.io/machine/learning/2015/08/10/random-forest-bias.html it seems that straight-forward **one**-hot-encoding doesn't scale nicely with variable importances when it comes to **decision trees** and **random forests**. On the other hand, if the categoricals are engineered into **other integer**-encoding (which I think they are in the current state), then it's scaled nicely. Also, this scaling is more apparent for larger numbers of variables. 
# 
# So I'm curious how many unique elements are in  each variable. Checking for uniqueness would help me decide whether to include them as predictors or not.

# In[ ]:


# Check for uniqueness of each relevant variable
train.apply(pd.Series.nunique).sort_values()
# Looks like country has the least number of uniqueness from original data. This might qualify for 1-hot-encoding(?), as described in the blogs above.


# Anyways, here is my training data set

# In[ ]:


# Here is my training dataset
train_X = onehotencoded_candidates
train_y = train['quantity']


# Oh, but I'm curous if I could check for RMSLE between the predicted- and the true- target, so I'm going to split up this training set more

# In[ ]:


# Splitting this training dataset further: to make train_train and train_test. This way I'd be able check for the "goodness of learning"
from sklearn.model_selection import train_test_split
train_train_X, train_test_X, train_train_y, train_test_y = train_test_split(train_X, train_y,random_state=0)


# In[ ]:


### Using RMSLE as validation

def rmsle(y_true,y_pred):
    return np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred))**2))


# ## Using random forest regressor out of the box

# In[ ]:


## Try Random Forests "out of the box"
from sklearn.ensemble import RandomForestRegressor

Forest_model_oob = RandomForestRegressor(n_jobs =-1, random_state=0)
Forest_model_oob.fit(train_train_X,train_train_y)
train_test_predictions = Forest_model_oob.predict(train_test_X)
print('Using Forest_model_oob, the rmsle is : {} \n'.format(rmsle(train_test_y,
                                                               train_test_predictions)))
pprint.pprint(Forest_model_oob.get_params())


# ### Now try hyperparameter tuning
# As explained here, https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74 , there are tons of things that can be tuned.
# When I tried to "plug-&-chug"-tuning all of the parameters, the search wouldn't finish in 6 hours, and my Kaggle kernel was stopped (timed out).
# 
# Looks like I need to be picky about which parameters to tune, so I started off following this one: https://www.analyticsvidhya.com/blog/2015/06/tuning-random-forest-model/
# 

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV
# Number of n_estimators
n_est = [60]
# Number of features to consider at every split
max_features = ['auto', 'sqrt','log2']
# Number of max_leaf_nodes
max_leaf_nodes = [5,1000,5000, 7500, 10000]
# how deep?
max_depth = [10, 25, 50, 75, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4, 50, 75]
# Create the random grid
random_grid = {'max_features': max_features,
               'n_estimators': n_est,
               'max_leaf_nodes': max_leaf_nodes,
               'max_depth': max_depth,
               'min_samples_leaf': min_samples_leaf,
               }
pprint.pprint(random_grid)


# In[ ]:


# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 25 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 50, cv = 3,
                               verbose=2, random_state=0, n_jobs = -1)
# Fit the random search model
rf_random.fit(train_train_X, train_train_y)


# ## Best parameters from the random search

# In[ ]:


rf_random.best_params_


# ## Evaluation of random search

# In[ ]:


def rmsle(y_true,y_pred):
    return np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred))**2))


# In[ ]:


def evaluate(model, train_test_X, train_test_y):
    predictions = model.predict(train_test_X)
    errors = abs(predictions - train_test_y)
    print('Model Performance:')
    print('Average deviation: {:0.1f} from actual quantity.'.format(np.mean(errors))) 
    print('rmsle= {:0.6f}'.format(rmsle(train_test_y,predictions)))
    return rmsle(train_test_y,predictions)


# In[ ]:


base_model = RandomForestRegressor(random_state = 0, n_jobs=-1)
base_model.fit(train_train_X, train_train_y)
base_rmsle = evaluate(base_model, train_test_X, train_test_y)


# In[ ]:


best_random = rf_random.best_estimator_
random_rmsle = evaluate(best_random, train_test_X, train_test_y)


# In[ ]:


print('With Random_search, the RMSLE is improved by {:0.1f}%.'.format( 100 * (random_rmsle - base_rmsle) / base_rmsle))


# Huh?....after 1 hour of random searching I'm not getting any better RMSLE?

# ## Anyhow let's just take a firstshot at the test dataset, using basemodel

# In[ ]:


test_candidates = test[pred_cols] # including the needed columns in test 
test_X = pd.get_dummies(test_candidates) # onehotcoding country
test_prediction = base_model.predict(test_X)


# In[ ]:


my_submission = pd.DataFrame({'id':test.id,
                             'quantity':test_prediction})
my_submission.to_csv('submission.csv', index=False)


# ## do a gridsearch based on the values provided by random search

# In[ ]:


#from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
#param_grid = {
#    'max_depth': [75, 100, 125],
#    'max_features': [2, 3],
#    'min_samples_leaf': [1, 2, 3],
#    'max_leaf_nodes': [6000, 7500, 8000],
#    'n_estimators': [50, 75, 90]
#}
# Create a based model
#rf = RandomForestRegressor()
# Instantiate the grid search model, 2-fold CV
#grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
#                          cv = 3, n_jobs = -1, verbose = 2)


# 

# ## fit the model, display and evaluate performance

# In[ ]:


# Fit the grid search to the data
#grid_search.fit(train_train_X, train_train_y)
#grid_search.best_params_
#{'max_depth': 100,
# 'max_features': 'auto',
# 'min_samples_leaf': 2,
# 'min_samples_split': 7500,
# 'n_estimators': 60}


# In[ ]:


#best_grid = grid_search.best_estimator_
#grid_rmsle = evaluate(best_grid, train_test_X, train_test_y)


# In[ ]:


#print('With grid_search after random_search, the RMSLE is improved by {:0.1f}%.'.format( 100 * (random_rmsle - base_rmsle) / base_rmsle))

