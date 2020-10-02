#!/usr/bin/env python
# coding: utf-8

# # Introduction
# **This will be your workspace for Kaggle's Machine Learning education track.**
# 
# You will build and continually improve a model to predict housing prices as you work through each tutorial.  Fork this notebook and write your code in it.
# 
# The data from the tutorial, the Melbourne data, is not available in this workspace.  You will need to translate the concepts to work with the data in this notebook, the Iowa data.
# 
# Come to the [Learn Discussion](https://www.kaggle.com/learn-forum) forum for any questions or comments. 
# 
# # Write Your Code Below
# 
# 

# In[ ]:


import pandas as pd

main_file_path = '../input/train.csv'
data = pd.read_csv(main_file_path)
print('hello world')


# In[ ]:



print(data.describe())


# In[ ]:


print(data.columns)


# In[ ]:


price_data = data.SalePrice
print(price_data.head())


# In[ ]:


columns_of_interest = ['LotArea','1stFlrSF','2ndFlrSF']
two_columns_of_data = data[columns_of_interest]


# In[ ]:


# columns_of_interest = ['LotArea','1stFlrSF','2ndFlrSF']
# two_columns_of_data = data[columns_of_interest]
two_columns_of_data = data[['LotArea','1stFlrSF','2ndFlrSF']]


# In[ ]:


two_columns_of_data.describe()


# In[ ]:


# choosing the prediction target
y=data.SalePrice


# In[ ]:


predictors = ['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']


# In[ ]:


X = data[predictors]


# In[ ]:


from sklearn.tree import DecisionTreeRegressor

#define model
house_model = DecisionTreeRegressor()

# fit
house_model.fit(X,y)


# In[ ]:


print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(house_model.predict(X.head()))


# In[ ]:


# model validation

from sklearn.tree import DecisionTreeRegressor

#define model
house_model = DecisionTreeRegressor()

# fit
house_model.fit(X,y)


# In[ ]:


from sklearn.metrics import mean_absolute_error

predicted_home_prices = house_model.predict(X)
mean_absolute_error(y, predicted_home_prices)


# In[ ]:


from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

#define model
house_model = DecisionTreeRegressor()

#fit model
house_model.fit(train_X, train_y)


# In[ ]:


# get prediction on validation data
val_predictions = house_model.predict(val_X)
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
for max_leaf_nodes in [5,25, 50,100, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))


# In[ ]:


#random forest

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor()
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, melb_preds))


# In[ ]:


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Read the data
train = pd.read_csv('../input/train.csv')

# pull data into target (y) and predictors (X)
train_y = train.SalePrice
predictor_cols = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']

# Create training predictors data
train_X = train[predictor_cols]

my_model = RandomForestRegressor()
my_model.fit(train_X, train_y)


# In[ ]:


# Read the test data
test = pd.read_csv('../input/test.csv')
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


# In[ ]:




