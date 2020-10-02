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
data = pd.read_csv(main_file_path)

# Run this code block with the control-enter keys on your keyboard. Or click the blue botton on the left
print('Some output from running this cell')


# In[ ]:


import pandas as pd
melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'
data = pd.read_csv(melbourne_file_path)

print(data.describe())


# In[ ]:


melbourne_data = pd.read_csv(melbourne_file_path)

print(melbourne_data.columns)

melbourne_price_data = melbourne_data.Price

print(melbourne_price_data.head())

two_columns_of_data = ["Landsize","Rooms"]

print(melbourne_data[two_columns_of_data])

print(melbourne_data[two_columns_of_data].describe())


# In[ ]:


import pandas as pd

from sklearn.tree import DecisionTreeRegressor

main_file_path = '../input/house-prices-advanced-regression-techniques/train.csv' # this is the path to the Iowa data that you will use
data = pd.read_csv(main_file_path)

print(data.columns)

y = data.SalePrice

desired_cols =["LotArea","YearBuilt","1stFlrSF","2ndFlrSF","FullBath","BedroomAbvGr","TotRmsAbvGrd"]

X = data[desired_cols]

training_model = DecisionTreeRegressor()

training_model.fit(X,y)

print("The data on which we predict as below")
print(X.head())

print('The predicted data as below')
print(training_model.predict(X.head()))


# In[ ]:


import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

main_file_path = '../input/house-prices-advanced-regression-techniques/train.csv' # this is the path to the Iowa data that you will use
data = pd.read_csv(main_file_path)


y = data.SalePrice

desired_cols =["LotArea","YearBuilt","1stFlrSF","2ndFlrSF","FullBath","BedroomAbvGr","TotRmsAbvGrd"]

X = data[desired_cols]

train_X,val_X,train_y,val_y = train_test_split(X,y,random_state=0)

train_model = DecisionTreeRegressor()

train_model.fit(train_X,train_y)

val_predict = train_model.predict(val_X)

print(mean_absolute_error(val_y,val_predict))




# In[ ]:


from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

def get_mae(max_leaf_nodes, predictors_train, predictors_val, targ_train, targ_val):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(predictors_train, targ_train)
    preds_val = model.predict(predictors_val)
    mae = mean_absolute_error(targ_val, preds_val)
    return(mae)
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor()

forest_model.fit(train_X,train_y)

train_predict = forest_model.predict(val_X)

print(mean_absolute_error(val_y,train_predict))


# In[ ]:



import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Read the data
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

print(train)

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
