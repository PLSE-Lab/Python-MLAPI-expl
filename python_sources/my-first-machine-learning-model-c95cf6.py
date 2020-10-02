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

# In[4]:


import pandas as pd

main_file_path = '../input/train.csv' # this is the path to the Iowa data that you will use
data = pd.read_csv(main_file_path)

# The cod below will help you see how output appears when you run a code block
# Run this code block with the control-enter keys on your keyboard. Or click the blue botton on the left
print('hello world')
import pandas as pd

main_file_path = '../input/train.csv' # this is the path to the Iowa data that you will use
data = pd.read_csv(main_file_path)

# The cod below will help you see how output appears when you run a code block
# Run this code block with the control-enter keys on your keyboard. Or click the blue botton on the left
print(data.describe())
print(data.columns)
data_price = data.SalePrice
print(data_price)
important_columns=['LotArea','Condition1','Condition2','BedroomAbvGr']
important_columns1=data[important_columns]
important_columns1.describe()
y = data.SalePrice

data_predictors = ['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']
x = data[data_predictors]
from sklearn.tree import DecisionTreeRegressor
#define
data_model=DecisionTreeRegressor()
#fit model
data_model.fit(x,y)
print("Making predictions for the following 5 houses:")
print(x.head())
print("The predictions are")
print(data_model.predict(x.head()))
from sklearn.metrics import mean_absolute_error

predicted_home_prices = data_model.predict(x)
mean_absolute_error(y, predicted_home_prices)

from sklearn.model_selection import train_test_split

# split data into training and validation data, for both predictors and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_x, val_x, train_y, val_y = train_test_split(x, y,random_state = 0)
# Define model
data_model = DecisionTreeRegressor()
# Fit model
data_model.fit(train_x, train_y)

# get predicted prices on validation data
val_predictions = data_model.predict(val_x)
print(mean_absolute_error(val_y, val_predictions))

def get_mae(max_leaf_nodes, predictors_train, predictors_val, targ_train, targ_val):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(predictors_train, targ_train)
    preds_val = model.predict(predictors_val)
    mae = mean_absolute_error(targ_val, preds_val)
    return(mae)
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_x, val_x, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
    
    
    #random forest
    from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor()
forest_model.fit(train_x, train_y)
data_preds = forest_model.predict(val_x)
print(mean_absolute_error(val_y, data_preds))
my_submission = pd.DataFrame({'Id': data.Id, 'SalePrice': data_price})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)


# 
# **If you have any questions or hit any problems, come to the [Learn Discussion](https://www.kaggle.com/learn-forum) for help. **
# 
# **Return to [ML Course Index](https://www.kaggle.com/learn/machine-learning)**

# 
