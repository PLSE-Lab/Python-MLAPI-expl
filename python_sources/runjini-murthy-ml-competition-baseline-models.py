#!/usr/bin/env python
# coding: utf-8

# # Runjini Murthy - Machine Learning Competition - Baseline Models
# Last updated: 04/01/2018
# ***

# In[1]:


import numpy as np
import pandas as pd
from datetime import datetime
import os
import gc
import re
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
pal = sns.color_palette()


# In[2]:


FILE_DIR = '../input/hawaiiml-data'

for f in os.listdir(FILE_DIR):
    print('{0:<30}{1:0.2f}MB'.format(f, 1e-6*os.path.getsize(f'{FILE_DIR}/{f}')))


# In[3]:


train = pd.read_csv(f'{FILE_DIR}/train.csv', encoding='ISO-8859-1')
test = pd.read_csv(f'{FILE_DIR}/test.csv', encoding='ISO-8859-1')
submission = pd.read_csv(f'{FILE_DIR}/sample_submission.csv', encoding='ISO-8859-1')


# In[5]:


train.head()


# In[6]:


test.head()


# In[7]:


submission.head()


# In[9]:


# From this point, begin using the modeling method described in Machine Learning tutorial.
print(train.columns)


# In[59]:


# Now begin modeling work.  Choose a target, y.
y = train.quantity

# Now, choose the predictor columns, which is X.
train_predictors = ['unit_price', 'customer_id', 'stock_id']
X = train[train_predictors]
print("Training model set up.")


# Test model 2.  SOMETHING IS NOT CORRECT IN THIS SET UP.  TRY TO SET UP TWO MODELS AT ONCE TO COMPARE RMSLE.
#train_predictors2 = ['stock_id', 'customer_id', 'id']
#X2 = train[train_predictors2]
#print("Training model 2 set up.")


# In[60]:


# Now we import the Decision Tree model from the scikit learn library.
from sklearn.tree import DecisionTreeRegressor

# Define model
train_model = DecisionTreeRegressor()

# Fit model
train_model.fit(X, y)

print("Model fitted.")

# Setup model 2
#train_model2 = DecisionTreeRegressor()

# Fit model
#train_model2.fit(X2, y)

#print("Model 2 fitted.")


# In[61]:


# Now we make predictions.
print("Making predictions for the following 5 item IDs:")
print(X.head())
print("Model predictions are")
print(train_model.predict(X.head()))

# Now we make predictions.
#print("Now we are making predictions for the following 5 item IDs in Model 2:")
#print(X2.head())
#print("Model 2 predictions are")
#print(train_model2.predict(X2.head()))


# In[62]:


# Now calculate error based on a sample of the data so that the model can be used for data it hasn't seen yet.
from sklearn.model_selection import train_test_split

# split data into training and validation data, for both predictors and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)
# Define model
train_model = DecisionTreeRegressor()
# Fit model
train_model.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = train_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))


# Model 2 split
#train_X2, val_X2, train_y2, val_y2 = train_test_split(X2, y,random_state = 0)
# Define model
#train_model2 = DecisionTreeRegressor()
# Fit model
#train_model2.fit(train_X2, train_y2)

# get predicted prices on validation data
#val_predictions2 = train_model2.predict(val_X2)
#print(mean_absolute_error(val_y2, val_predictions2))


# In[63]:


# Calculate optimal number of nodes.
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

def get_mae(max_leaf_nodes, predictors_train, predictors_val, targ_train, targ_val):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(predictors_train, targ_train)
    preds_val = model.predict(predictors_val)
    mae = mean_absolute_error(targ_val, preds_val)
    return(mae)

# compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
    
# The optimal number of nodes is 5000.


# In[64]:


train_model.fit(train_X, train_y)
# Treat the test data in the same way as training data. In this case, pull same columns.
test_X = test[train_predictors]
# Use the model to make predictions
predicted_quantity = train_model.predict(test_X)
# We will look at the predicted quantity to ensure we have something sensible.
print(predicted_quantity)

#train_model2.fit(train_X2, train_y2)
# Treat the test data in the same way as training data. In this case, pull same columns.
#test_X2 = test[train_predictors2]
# Use the model to make predictions
#predicted_quantity2 = train_model2.predict(test_X2)
# We will look at the predicted quantity to ensure we have something sensible.
#print(predicted_quantity2)

my_submission = pd.DataFrame({'id': test.id, 'quantity': predicted_quantity})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)


# In[ ]:


#from sklearn.ensemble import RandomForestRegressor
#from sklearn.metrics import mean_absolute_error
 
#forest_model = RandomForestRegressor()
#forest_model.fit(train_X, train_y)
#forest_preds = forest_model.predict(val_X)
#print(mean_absolute_error(val_y, forest_preds))
#print(forest_preds)

# Use the model to make predictions
#forest_preds_quantity = forest_model.predict(test_X)
# We will look at the predicted prices to ensure we have something sensible.
#print(forest_preds_quantity)

#my_submission = pd.DataFrame({'id': test.id, 'quantity': predicted_quantity})
# you could use any filename. We choose submission here
#my_submission.to_csv('submission.csv', index=False)


# In[66]:


# RMLSE error calculator - Model 1
import numpy as np
np.random.seed(0)

def rmsle(val_y, train_y):
    return np.sqrt(np.mean((np.log1p(val_y) - np.log1p(train_y))**2))

# create random ground truth values
val_y = 2**np.random.uniform(0, 10, 10)
print('Actual Values: \n', val_y)

# create noisy predictions
train_y = np.random.normal(15+val_y, 20, 10)
print('Predicted Values: \n', train_y)

# calculate error
print(f'RMSLE: {rmsle(val_y, train_y):0.5f}' )

# RMSE error
def rmse(val_y, train_y):
    return np.sqrt(np.mean((val_y - train_y)**2))

rmsle(val_y, train_y) == rmse(np.log1p(train_y), np.log1p(val_y))

