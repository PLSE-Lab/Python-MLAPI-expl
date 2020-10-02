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

# In[61]:


import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
main_file_path = '../input/train.csv'
data = pd.read_csv(main_file_path)
column_of_interest = ['SalePrice']
sales_data = data[column_of_interest]
two_columns = ['SaleType','SaleCondition']
sale_two_columns = data[two_columns]
#print(data.describe())
#print(data.columns)
#print(sales_data.head())
#print(sale_two_columns.describe())
list_x = ['LotArea' ,'YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']
x = data[list_x]
y = data.SalePrice
#define model 
my_model = DecisionTreeRegressor()
# fit model
my_model.fit(x,y)
#print('Data selected for prediction: ')
#print(x.head())
#print('Result predection:')
#print(my_model.predict(x.head()))
# Calulate mean absolute error (MAE)
Predicted_house_price = my_model.predict(x)
MAE_traindata=mean_absolute_error(y,Predicted_house_price)
print('MAE_traindata:',MAE_traindata)
#Building validation data - this is done outside the given sample data
#Decision Tree model
train_x,val_x,train_y,val_y = train_test_split(x,y,random_state =0)

validation_model = DecisionTreeRegressor()
validation_model.fit(train_x,train_y)

# get predicted prices on validation data

validation_predection = validation_model.predict(val_x)
MAE_validationdata=mean_absolute_error(val_y,validation_predection)
print('MAE_validationdata:', MAE_validationdata)

# get_mae - decision model 
def get_mae(max_leaf_nodes, predictors_train, predictors_val, targ_train, targ_val):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(predictors_train, targ_train)
    preds_val = model.predict(predictors_val)
    mae = mean_absolute_error(targ_val, preds_val)
    return(mae)

for max_leaf_nodes in [2,10,100,50,1000,20,30,40,53,45,55,90,60,150]:
    my_mae = get_mae(max_leaf_nodes,train_x,val_x,train_y,val_y)
    print('max_leaf_nodes:' ,max_leaf_nodes, '     ' , 'My_MAE:' ,my_mae)

#Random Forest model:
#define

random_model = RandomForestRegressor()
random_model.fit(train_x,train_y)

random_model_predection = random_model.predict(val_x)
MAE_Random_model = mean_absolute_error(val_y,random_model_predection)
print('MAE_Random_model: ', MAE_Random_model)

# get_mae - random model 
def get_mae(max_leaf_nodes, predictors_train, predictors_val, targ_train, targ_val):
    model = RandomForestRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(predictors_train, targ_train)
    preds_val = model.predict(predictors_val)
    mae = mean_absolute_error(targ_val, preds_val)
    return(mae)

for max_leaf_nodes in [2,10,100,50,1000,20,30,40,53,45,55,90,60,150]:
    my_mae_random = get_mae(max_leaf_nodes,train_x,val_x,train_y,val_y)
    print('max_leaf_nodes:' ,max_leaf_nodes, '     ' , 'My_MAE:' ,my_mae_random)


# In[72]:


# Submission

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Train data

main_file_path = '../input/train.csv'
data = pd.read_csv(main_file_path)
list_x = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']
x = data[list_x]
y = data.SalePrice

# Test Data

test_data_path = '../input/test.csv'
data = pd.read_csv(test_data_path)
#print(data.columns)
predictor_columns = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']
test_x = data[predictor_columns] 
# test_y to be found

# Predict price
prediction_variable = RandomForestRegressor()
prediction_variable.fit(x,y)
Predicted_price = prediction_variable.predict(test_x)
print('Predicted_price:' , Predicted_price)

# Submission file creation

my_submission = pd.DataFrame({'Id': data.Id, 'SalePrice': Predicted_price})
my_submission.to_csv('submission_rakesh.csv', index=False)

