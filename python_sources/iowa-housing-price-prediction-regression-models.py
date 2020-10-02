#!/usr/bin/env python
# coding: utf-8

# **A Deeper Look Into The Housing Price Trends In Iowa, USA **
# 
# The predictive models used in this study were :
# 
# * Random Forest Regression
# * Decision Trees
# * XGBoost (Gradient Boosted Decision Trees)
# 
# It was observed that the mean absolute error (MAE) of the XGBoost model was lesser compared to the other two, i.e. decision trees and random forest. I have also added the code that was used to build the decision tree model, for a better comparative analysis.
# 
# Note: After imputation, the error value was found to be even lesser in all cases. It's therefore recommended to impute the columns with null values (NaN), so as to get the least error possible
# 
# *Special thanks to Dan Becker, for providing the datasets necessary for analysis and guiding me through, when clarifications with respect to certain data points had to be made *
# 
# Cheers,
# Karthik
# 

# In[ ]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Read the data
train = pd.read_csv('../input/train.csv')

# Pulling training data into target (y) and predictors (X)
train_y = train.SalePrice
predictor_cols = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']

# Create training predictors data
train_X = train[predictor_cols]

my_model = RandomForestRegressor()
my_model.fit(train_X, train_y)

# Read the test data
test = pd.read_csv('../input/test.csv')
# Treat the test data in the same way as training data. In this case, pull same columns.
test_X = test[predictor_cols]
# Use the model to make predictions
predicted_prices = my_model.predict(test_X)
# We will look at the predicted prices to ensure we have something sensible.
print("The predicted price values for Iowa are as follows: \n", predicted_prices)

#Visualization plot for the training data
plt.figure(figsize = (15,6))
plt.bar(train.YearBuilt, train_y)
plt.xlabel("Year In Which The Houses Were Built")
plt.ylabel("Price (USD)")
plt.show()

#Visalization plot for the test data
plt.figure(figsize = (15,6))
plt.bar(test.YearBuilt, predicted_prices)
plt.xlabel("Year In Which The Houses Were Built")
plt.ylabel("Predicted Price (USD)")
plt.show()

#Observation: Random forest model is not reliable without data splitting and imputation. It also gives a mean absolute error of ~24200 which is a anomaly number

#-----------------------------------------------------------------------------------------------------------


# In[ ]:


#File Submission
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
my_submission.to_csv('submission_karthikramesh_1_iowa_housing.csv', index=False)


# In[ ]:


#Generating a dataframe of the Iowa dataset obtained from the government
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
import pandas as pd

iowa_file_path = '../input/train.csv'
iowa_training_data = pd.read_csv(iowa_file_path)
print(iowa_training_data.columns)
print("\n")

#Defining the label/target variable
y = iowa_training_data.SalePrice
print(y.head(5))
print("\n")

#Defining the features/predictors
iowa_predictors = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
x = iowa_training_data[iowa_predictors]
print(x.describe())
print("\n")

#Defining the prediction model and fitting the model on the training dataset
train_x, val_x, train_y, val_y = train_test_split(x, y, random_state = 0)
iowa_model = DecisionTreeRegressor()
iowa_model.fit(train_x, train_y)

#Initiating the prediction function for forecasting prices for the validation dataset comprising information on Iowa's housing
print("Making predictions for houses in Iowa, on their prices:")
print(val_x)
print("The values are:")
val_predictions = iowa_model.predict(val_x)
print(val_predictions)
print("\n")

#Calculating the mean absolute error
print("The mean absolute error is:", mean_absolute_error(val_y, val_predictions))

#Utility function for evaluating the model and finding the maximum leaf nodes the model must have to attain the least MAE (mean absolute error)
def get_mae(max_leaf_nodes, trainX, valX, trainY, valY):
    model = DecisionTreeRegressor(max_leaf_nodes = max_leaf_nodes, random_state = 0)
    model.fit(trainX, trainY)
    pred_val = model.predict(valX)
    mae = mean_absolute_error(valY, pred_val)
    return(mae)

#Comparing the models with differing values of maximum leaf nodes, by calling the utility function
for max_leaf_nodes in [5, 50, 500, 5000]:
    mae_obtained = get_mae(max_leaf_nodes, train_x, val_x, train_y, val_y)
    print("Max Leaf Nodes: %d \t\t Mean Absolute Error: %d" %(max_leaf_nodes, mae_obtained))
    
#Observation: With 50 leaf nodes in the decision tree, the mean absolute error is the least with the value 27825!


# In[ ]:


#Handling missing values

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split

# Read the data
train = pd.read_csv('../input/train.csv')

# Pulling data into target (y) and predictors (x)
train_y = train.SalePrice
train_x = train.drop(['SalePrice'], axis = 1)
train_x_numerical = train_x.select_dtypes(exclude = ['object'])

#Defining the mean absolute error function after splitting the dataset into training data and test data, fitting the random forest model and predicting the prices
train_X, test_X, train_Y, test_Y = train_test_split(train_x_numerical, train_y, train_size=0.7, test_size=0.3, random_state=0)
def score_dataset(train_X, test_X, train_Y, test_Y):
    model = RandomForestRegressor()
    model.fit(train_X, train_Y)
    predictions = model.predict(test_X)
    return mean_absolute_error(test_Y, predictions)


#Defining the imputation function and calculating the mean absolute error(MAE):
my_imputer = Imputer()
imputed_X_training_set = my_imputer.fit_transform(train_X)
imputed_X_test_set = my_imputer.transform(test_X)
print("The mean absolute error after imputation of datasets is:")
print(score_dataset(imputed_X_training_set, imputed_X_test_set, train_Y, test_Y))

#Observation: The mean absolute error using the Random Forest model after imputation is 18985.42


# In[ ]:


#Implementing the XGBoost Model
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import Imputer

#Importing the parent dataset as a dataframe
iowa_data = pd.read_csv('../input/train.csv')
iowa_data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = iowa_data.SalePrice
x = iowa_data.drop(['SalePrice'], axis=1).select_dtypes(exclude = ['object'])

#Splitting the data into training dataset and test dataset
train_X, test_X, train_Y, test_Y = train_test_split(x, y, train_size = 0.7, test_size = 0.3, random_state = 0)

#Imputation of the predictor and target data
my_imputer = Imputer()
train_X = my_imputer.fit_transform(train_X)
test_X = my_imputer.transform(test_X)

#Defining the XGBoost model and fitting the training dataset
my_model = XGBRegressor(n_estimators = 1000, learning_rate = 0.05)
my_model.fit(train_X, train_Y, early_stopping_rounds = 5, eval_set = [(test_X, test_Y)], verbose = False)

#Predicting the prices using the test dataset
iowa_housing_price_prediction = my_model.predict(test_X)
print("The predicted prices for Iowa housing are as follows: " + "\n\n", iowa_housing_price_prediction)
print("\n")

#Calculating the mean absolute error (MAE)
print("The mean absolute error for this model is:", mean_absolute_error(test_Y, iowa_housing_price_prediction))


#Observation: The mean absolute error is lesser with a value of 18068.821


# In[ ]:




