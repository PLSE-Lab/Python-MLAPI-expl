#!/usr/bin/env python
# coding: utf-8

# # Hello MLWorld, an Introduction to Machine Learning.
# **A workspace for the [Machine Learning course](https://www.kaggle.com/learn/machine-learning).**

# # Setup.
# Performing necessary imports for use in notebook.

# In[ ]:


import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

main_file_path = '../input/house-prices-advanced-regression-techniques/train.csv' # Path to Iowa house sale data
data = pd.read_csv(main_file_path)


# # Selecting Columns.
# Demonstration of subsetting via dot-notation and index.

# In[ ]:


##### Select single column
# Grab column from series using name of column via dot-notation
iowa_price_data = data.SalePrice

# Peep first 5 rows of price
print(iowa_price_data.head())

##### Select multiple columns
# declare list, specifying column name(s)
given_columns = ['YrSold', 'LotArea']

# subset DF with said list
two_selected_columns = data[given_columns] 

# verify that we have the selected columns
print(two_selected_columns.describe())


# # Setting your Prediction Target and Predictors.
# Choose which column you would like to predict. 
# By convention, the _prediction target_ is called **y** and _predictors_ **X**.

# In[ ]:


# Set prediction target
y = data.SalePrice

# Set predictors
iowa_sale_predictors = ["LotArea",
                        "YearBuilt",
                        "1stFlrSF",
                        "2ndFlrSF",
                        "FullBath",
                        "BedroomAbvGr",
                        "TotRmsAbvGrd"]

# Subset dataset by predictors
X = data[iowa_sale_predictors]

print(X.head())


# # Building Your Model.
# Importing function from **tree** module of **scikit-learn** to _Define_ and  _Fit_ the model.

# In[ ]:


# from sklearn.tree import DecisionTreeRegressor # as imported in setup

# Define model
iowa_model = DecisionTreeRegressor()

# Fit model with predictors and prediction
iowa_model.fit(X, y)


# # Using the Model to Predict.
# The argument passed to the 'predict' method below informs what the model is predicting for.

# In[ ]:


# Use model to predict
prediction = iowa_model.predict(X)

print("Predictions are for the following 5 houses:")
print(X.head())

print("Predictions are:")
print(prediction[:5])

print("Original values are:")
print(y.head())


# # Evaluating the Accuracy of Your Model.
# A first look at using Mean Absolute Error (MAE) as a metric for summarizing model quality.

# In[ ]:


# from sklearn.metrics import mean_aboslute_error # as imported in setup

# Calculate MAE
_iowa_MAE = mean_absolute_error(prediction, y)
print("The MAE of inital prediction is %d"%(_iowa_MAE))


# # Okay, but what about the feasibility of the Data?
# At this point, we are calculating MAE as an _in-sample_ score - Building the model and calculating the MAE on the same dataset -  **very bad**.   
# This a major issue, as the model has been trained to find patterns in the dataset, so all of the model's predictions derived from those patterns will _appear_ accurate against the training data.
# 
# # Try Again; Split your Data.
# A way to smash the pattern bias to split the data into _training_ and  _validation_ datasets, testing the model's prediction accuracy against data it hasn't been exposed to.

# In[ ]:


# from sklearn.metrics import mean_absolute_error # as imported in setup

# data is split into training and validation sets, for predictors & target
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0) #random_state=0 so the split will be the same every time

# Define Model
iowa_model = DecisionTreeRegressor()
# Fit Model on training data
iowa_model.fit(train_X, train_y)

# Predict on validation data
val_predicts = iowa_model.predict(val_X)
iowaV2_MAE = mean_absolute_error(val_y, val_predicts)

print("The MAE of model on training split: "+ str(iowaV2_MAE))


# # Overfitting and Underfitting; _The Sweet Spot_
# Overfitting a model occurs when the noise of the data is captured, and the model  fits the dataset _too well_, therefore does not provide the most accurate predictions on unseen data - low bias, high variance.  
# Underfitting causes the underlying trend in the data to be missed, usually caused by an excessively simple model - high bias, low variance.  
# ![http://i.imgur.com/2q85n9s.png](http://i.imgur.com/2q85n9s.png)
# Validation and Cross-Validation can be used to pit models against one another to find the _sweet spot_, represented in the figure as the low point of the red line.

# In[ ]:


# from sklearn.metrics import mean_absolute_error
# from sklearn.tree import DecisionTreeRegressor

# function to build & train model, returns MAE 
def get_mae(max_leaf_nodes, predictors_train, predictors_val, targ_train, targ_val):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(predictors_train, targ_train)
    preds_val = model.predict(predictors_val)
    mae = mean_absolute_error(targ_val, preds_val)
    return(mae)

# Loop over range of values to find sweet spot
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d \t\t Mean Absolute Error: %d" %(max_leaf_nodes, my_mae))


# # Implementing a Complex Model
#  To avoid the pitfalls of using a single decision tree, we can use the power of many trees - a forest - using the **random forest** modeling technique, to produce a more accurate prediction.

# In[ ]:


# from sklearn.ensemble import RandomForestRegressor # as imported in setup
# from sklearn.metrics import mean_absolute_error

# Define model
iowa_forest_model = RandomForestRegressor()
# Fit model
iowa_forest_model.fit(train_X, train_y)
# Predict
iowa_predicts = iowa_forest_model.predict(val_X)

#calculate MAE from new model
ifm_mae = mean_absolute_error(val_y, iowa_predicts)
old_mae = get_mae(80, train_X, val_X, train_y, val_y)
print("Random Forest Regressor MAE is %d" % (ifm_mae))
print("Best run of Decision Tree Regressor MAE is %d" % (old_mae))


# # Submission.

# In[ ]:


# Get test data
path = '../input/house-prices-advanced-regression-techniques/%s.csv'

train = pd.read_csv(path % ('train'))
test = pd.read_csv(path % ('test'))
# Handle test data in the same manner as training data, select same columns
predictor_cols = ["LotArea",
                    "YearBuilt",
                    "1stFlrSF",
                    "2ndFlrSF",
                    "FullBath",
                    "BedroomAbvGr",
                    "TotRmsAbvGrd",
                    "OverallQual"]
train_y = train.SalePrice
train_X = train[predictor_cols]
test_X = test[predictor_cols]

rfr_model = RandomForestRegressor()
rfr_model.fit(train_X, train_y)
# Predict
predicted_prices = rfr_model.predict(test_X)

# Peep for sanity 
print(predicted_prices)

# Prepare submission file
rfr_submission = pd.DataFrame({'ID': test.Id, 'SalePrice': predicted_prices})
rfr_submission.to_csv('submission.csv', index=False)


# # Gaining information from your data.
# Cheatsheet of **functions, methods and attributes from this notebook** with their types/return types to extract information and verify data (sanity check) among other things.

# # pandas (pandas)
# #### Reads a CSV file, creates DataFrame (DF) with data from read.
# read_csv = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv') 
# print (read_csv.head())
# 
# #### get description of DF (rt: DataFrame)
# describe = data.describe() 
# print(describe)
# 
# #### get all columns (rt: list)
# columns = data.columns
# print(columns)
# 
# #### get first 5 rows
# head = data.head()
# print(head)
# 
# #### get last 5 rows
# tail = data.tail()
# print(tail)
# 
# # scikit-learn (sklearn)
# ## tree (sklearn.tree)
# #### Creates model implementing decision tree regression
# arbitrary_model = DecisionTreeRegressor()
# 
# ## ensemble (sklearn.ensemble)
# arbitrary_model2 = RandomForestRegressor()
# 
# #### Train model (uses libsvm under the hood)
# arbitrary_model.fit(X, y) # X = predictor, y = prediction target
# 
# #### Use model to predict Y for values passed by argument
# arb_prediction = arbitrary_model.predict(X.head())
#     
# ## metrics (sklearn.metrics)
# mean_absolute_error(y, arb_prediction)
# 
# #### model_selection (sklearn.model_selection)
# train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

# 
# **If you have any questions or hit any problems, come to the [Learn Discussion](https://www.kaggle.com/learn-forum) for help. **
# 
# **Return to [ML Course Index](https://www.kaggle.com/learn/machine-learning)**

# 
