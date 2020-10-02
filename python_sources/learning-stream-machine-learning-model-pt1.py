#!/usr/bin/env python
# coding: utf-8

# # Introduction
# **This is my workspace for Kaggle's Machine Learning education track.**
# 
# This is were i have transfered the knowledge from the workbooks onto building a model to predict the sale price of Iowa realestate data.
# 
# # The Code Is Below

# In[1]:


# Import the data and see what it contains
import pandas as pd

iowa_file_path = '../input/train.csv'
# read the data and store data in DataFrame titled iowa_data
iowa_data = pd.read_csv(iowa_file_path) 
# print a summary of the data in Iowa data
print(iowa_data.describe())


# ## Selecting and filtering in pandas

# In[2]:


# What columns are in the data
print(iowa_data.columns)


# In[3]:


# what are the types of prices
iowa_price_data = iowa_data.SalePrice
iowa_price_data.head()


# In[4]:


# Select two columns of interest
CoI =['YrSold','PoolArea']
two_columns_of_data = iowa_data[CoI]
two_columns_of_data.head()


# ## Scikit-Learn Model
# Here we build a decision tree to predict the sale price

# In[5]:


# Create a list of features and targets
from sklearn.tree import DecisionTreeRegressor

target = iowa_data['SalePrice']
features = iowa_data[['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']]


# In[6]:


# Define model
iowa_model = DecisionTreeRegressor()

# Fit model
iowa_model.fit(features, target)


# In[7]:


#Lets cheat and predict the price of some houses we used in the training model.
print(iowa_model.predict(features.head()))


# ## Model Validation
# Lets start again, and build another decision tree model. This time we split the data to train and test the model.

# In[8]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Split the data
train_X, val_X, train_y, val_y = train_test_split(features, target, random_state = 0)
# Define a new model
iowa_model = DecisionTreeRegressor()
# Fit model with the training data
iowa_model.fit(train_X, train_y)

# Make predicted prices on validation data
val_predictions = iowa_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))


# ## Underfitting, Overfitting and Model Optimization

# In[9]:


# Define the mean absolute error function
def get_mean_absolute_error(max_leaf_nodes, predictors_train, predictors_val, targ_train, targ_val):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(predictors_train, targ_train)
    preds_val = model.predict(predictors_val)
    mae = mean_absolute_error(targ_val, preds_val)
    return(mae)

for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mean_absolute_error(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))


# In[11]:


for max_leaf_nodes in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
    my_mae = get_mean_absolute_error(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))


# ## Random Forests
# Lets improve the model by using a random forest.

# In[12]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor()
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, melb_preds))


# We made some minor improvement by using a random forest.

# ## Submitting From A Kernel

# In[13]:


# Read the test data
train = pd.read_csv('../input/train.csv')
train_y = train.SalePrice

predictor_cols = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']
# Create training predictors data
train_X = train[predictor_cols]

my_model = RandomForestRegressor()
my_model.fit(train_X, train_y)

test = pd.read_csv('../input/test.csv')
# Treat the test data in the same way as training data. In this case, pull same columns.
test_X = test[predictor_cols]

predicted_prices = my_model.predict(test_X)

my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




