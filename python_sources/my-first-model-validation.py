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

# In[36]:


# Libraries
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Import Iowa real estate data and put it in a variable
main_file_path = '../input/train.csv'
data = pd.read_csv(main_file_path)

# Set the prediction target
y = data.SalePrice

# Set predictors as Lot size, 1st and 2nd floor Sq footage, number of bedrooms above ground, and year built 
predictors = ['LotArea', '1stFlrSF', '2ndFlrSF', 'BedroomAbvGr', 'YearBuilt']
x = data[predictors]

# Define model
model = DecisionTreeRegressor()

# Fit model (Basic)
model.fit(x, y)
predictions = model.predict(x)

# Run and display an initial mean absolute error
mae1 = mean_absolute_error(y, predictions)
output1 = 'MAE 1 = ' + repr(mae1)

print(output1)  

# run it again with training and validation data

train_x, val_x, train_y, val_y = train_test_split(x, y, random_state = 0)
# Define model
model = DecisionTreeRegressor()
# Fit model
model.fit(train_x, train_y)

# get predicted prices on validation data
val_predictions = model.predict(val_x)

# Run and display the mean absolute error using the second model
mae2 = mean_absolute_error(val_y, val_predictions)
output2 = 'MAE 2 = ' + repr(mae2)

print(output2)

