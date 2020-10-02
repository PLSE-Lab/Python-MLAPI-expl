#!/usr/bin/env python
# coding: utf-8

# # Introduction
# **This will be your workspace for the [Machine Learning course](https://www.kaggle.com/learn/machine-learning).**
# 
# You will need to translate the concepts to work with the data in this notebook, the Iowa data. Each page in the Machine Learning course includes instructions for what code to write at that step in the course.
# 
# # Write Your Code Below

# In[ ]:


import pandas as pd

main_file_path = '../input/house-prices-advanced-regression-techniques/train.csv' # this is the path to the Iowa data that you will use
lowa_data = pd.read_csv(main_file_path)
print(lowa_data.describe())

# Run this code block with the control-enter keys on your keyboard. Or click the blue botton on the left
print('Some output from running this cell')


# In[ ]:


print(lowa_data.columns)


# In[ ]:


lowa_price_data = lowa_data.SalePrice
print(lowa_price_data.head())


# In[ ]:


columns_of_interest = ['SaleType','SaleCondition']
two_columns_of_data = lowa_data[columns_of_interest]
print(two_columns_of_data.describe())


# In[ ]:


y = lowa_data.SalePrice
lowa_predictors = ['LotArea','YearBuilt','1stFlrSF','2ndFlrSF', 'FullBath','BedroomAbvGr','TotRmsAbvGrd']
X = lowa_data[lowa_predictors]
from sklearn.tree import DecisionTreeRegressor
lowa_model = DecisionTreeRegressor()
lowa_model.fit(X,y)


# In[ ]:


print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(lowa_model.predict(X.head()))


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
# split data into training and validation data, for both predictors and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)
# Define model
lowa_model = DecisionTreeRegressor()
# Fit model
lowa_model.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = lowa_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))


# In[ ]:


from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

def get_mae(max_leaf_nodes, predictors_train, predictors_val, targ_train, targ_val):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(predictors_train, targ_train)
    preds_val = model.predict(predictors_val)
    mae = mean_absolute_error(targ_val, preds_val)
    return(mae)


# In[ ]:


# compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor()
forest_model.fit(train_X, train_y)
lowa_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, melb_preds))


# In[ ]:


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Read the data
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

# pull data into target (y) and predictors (X)
train_y = train.SalePrice
predictor_cols = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']

# Create training predictors data
train_X = train[predictor_cols]

my_model = RandomForestRegressor()
my_model.fit(train_X, train_y)


# In[ ]:


# Read the test data
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
# Treat the test data in the same way as training data. In this case, pull same columns.
test_X = test[predictor_cols]
# Use the model to make predictions
predicted_prices = my_model.predict(test_X)
# We will look at the predicted prices to ensure we have something sensible.
print(predicted_prices)


# In[ ]:


my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)


# 
# **If you have any questions or hit any problems, come to the [Learn Discussion](https://www.kaggle.com/learn-forum) for help. **
# 
# **Return to [ML Course Index](https://www.kaggle.com/learn/machine-learning)**

# 

# 

# 

# 

# 
