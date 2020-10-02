#!/usr/bin/env python
# coding: utf-8

# # Intro
# **This is your workspace for Kaggle's Machine Learning course**
# 
# You will build and continually improve a model to predict housing prices as you work through each tutorial.
# 
# The tutorials you read use data from Melbourne. The Melbourne data is not available in this workspace.  Instead, you will translate the concepts to work with the data in this notebook, the Iowa data.
# 
# # Write Your Code Below
# 

# In[17]:


import pandas as pd

main_file_path = '../input/train.csv' # this is the path to the Iowa data that you will use
data = pd.read_csv(main_file_path)

# The cod below will help you see how output appears when you run a code block
# Run this code block with the control-enter keys on your keyboard. Or click the blue botton on the left
print(data.describe())
data.info()
data.columns
data_SalePrice = data.SalePrice
print(data_SalePrice.head())
columns_of_interest = ['LotArea', 'YearBuilt']
two_columns_of_interest = data[columns_of_interest]
two_columns_of_interest.describe()

y = data.SalePrice
iowa_predictors = ['GarageCars', 'Fireplaces', 'TotRmsAbvGrd', 'FullBath', 'TotalBsmtSF', 'YearBuilt', 'OverallCond', 'LotArea']
X = data[iowa_predictors]

# Building model
"""
steps are:
1. Define the model. what kind will it be?
2. Fit. Capture patterns from provided data.
3. Predict.
4. Evaluate. Determine how accurate the model's predictions are.
"""

from sklearn.tree import DecisionTreeRegressor

# define model 
iowa_model = DecisionTreeRegressor()

# fit model
iowa_model.fit(X, y)

print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(iowa_model.predict(X.head()))

# I built a model, but how good is it?

# metric for summarizing model quality, one called MAE (mean absolute error)
# the prediction for each house is error = actual - predicted
# so if a house costs $150,000 and you predicted it would cost 100,000, the error is 
# 50,000

from sklearn.tree import DecisionTreeRegressor
iowa_model = DecisionTreeRegressor()
iowa_model.fit(X, y)

from sklearn.metrics import mean_absolute_error
predicted_home_prices = iowa_model.predict(X)
mean_absolute_error(y, predicted_home_prices)

from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
# define
iowa_model = DecisionTreeRegressor()
# fit
iowa_model.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = iowa_model.predict(val_X)

print("mae: ", mean_absolute_error(val_y, val_predictions))

from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

# I don't understand this whole section.
# Go back to this...

def get_mae(max_leaf_nodes, predictors_train, predictors_val, targ_train, targ_val):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(predictors_train, targ_train)
    preds_val = model.predict(predictors_val)
    mae = mean_absolute_error(targ_val, preds_val)
    return(mae)

for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
    
# Building a random forest

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor()
forest_model.fit(train_X, train_y)
iowa_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, iowa_preds))


# 
# **If you have any questions or hit any problems, come to the [Learn Discussion](https://www.kaggle.com/learn-forum) for help. **
# 
# **Return to [ML Course Index](https://www.kaggle.com/learn/machine-learning)**

# 
