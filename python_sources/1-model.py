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

# In[9]:


import pandas as pd # the most important part of the Pandas library is the DataFrame. 
#A DataFrame holds the type of data you might think of as a table. 
#This is similar to a sheet in Excel, or a table in a SQL database. 
main_file_path = '../input/train.csv' #directory where the data is
data = pd.read_csv(main_file_path) # convert the data into a DataFrame using pd

y=data.SalePrice # prediction target
d_pred=['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd'] 
# select predictors
X=data[d_pred] #all the predictors will be a vector X

from sklearn.tree import DecisionTreeRegressor #scikit-learn library to create models for the Data.Frames 
my_model=DecisionTreeRegressor()#define model
my_model.fit(X,y) #fit model
print("Prediction for the first eight houses:")
print(X.head())
print("The predictions are:")
print(my_model.predict(X.head())) #predict inside the training data to see how it works

from sklearn.metrics import mean_absolute_error as MAE #Model Validation
#MAE-take absolute value of each error and average it: On average, our predictions are off by about X
predicted_prices=my_model.predict(X) #error=actual-predicted so we create variable "predicted_prices"
MAE(y,predicted_prices) #mean_absolute_error between actual y and predicted_prices

#this model is inaccurate because it is "in-sample"scores. The prediction was done on the data we used,
#not on a new data so it will appear to be accurate although it is not. 
#Solution: We can divide the data into two sets, one for model construction and the other one to 
#test the model on it, so called new data "validation data".

from sklearn.model_selection import train_test_split as tts# split data into training and validation data,
#for both predictors and target. The split is based on a random number generator. 
# Supplying a numeric value to the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = tts(X, y,random_state = 0)
my_model=DecisionTreeRegressor() #defining model
my_model.fit(train_X,train_y) #fitting the model
val_pred=my_model.predict(val_X) # get predicted prices on validation data
print(MAE(val_y, val_pred))

#two problems:Overfitting "making the tree too deep" or underfitting "making the tree too shallow"
#use utility function to specify how many leaves are needed
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.tree import DecisionTreeRegressor
def get_mae(max_leaf_nodes, predictors_train, predictors_val, targ_train, targ_val):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(predictors_train, targ_train)
    preds_val = model.predict(predictors_val)
    mae = MAE(targ_val, preds_val)
    return(mae)
# compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
    
from sklearn.ensemble import RandomForestRegressor #import Random Forest function which has multiple trees
my_forest_model=RandomForestRegressor () #specify what model is needed
my_forest_model.fit(train_X,train_y)# fit the Random Forest model on the training data
prediction_values=my_forest_model.predict(val_X) #use the fitted model to predict X values 
#using validation set
print(MAE(val_y,prediction_values)) #returns MAE much better than the DecisionTree MAEs

import numpy as np #NumPy arrays facilitate advanced mathematical and other types of 
#operations on large numbers of data.
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
training_data = pd.read_csv('../input/train.csv') # Read the data
train_y = training_data.SalePrice # pull data into target (y) and predictors (X)
predictor_cols = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']
train_X = training_data[predictor_cols] # Create training predictors data
model_for_comp = RandomForestRegressor()
model_for_comp.fit(train_X, train_y)
# Read the test data
testing_data = pd.read_csv('../input/test.csv')
# Treat the test data in the same way as training data. In this case, pull same columns.
test_X = testing_data[predictor_cols]
# Use the model to make predictions
predicted_prices = model_for_comp.predict(test_X)
# We will look at the predicted prices to ensure we have something sensible.
print(predicted_prices)
my_submission = pd.DataFrame({'Id': testing_data.Id, 'SalePrice': predicted_prices}) #the end result needs to be
#submitted in csv format so using pandas create a DataFrame with this data, and then use the dataframe's 
#to_csv method to write the submission file
my_submission.to_csv('submission.csv', index=False) #the argument index=False to 
#prevent pandas from adding another column in our csv file.

