#!/usr/bin/env python
# coding: utf-8

# # Notebook from the tutorial
# This is a notebook when practicing the tutorial from Dan Becker
# [https://www.kaggle.com/dansbecker/model-validation](https://www.kaggle.com/dansbecker/model-validation)
# 
# It is done for educational purposes. It is a very simple model
# 

# # Introduction
# We first download data as follows

# In[3]:


import pandas as pd

main_file_path = '../input/train.csv'
data = pd.read_csv(main_file_path)
print(data.describe())


# We can test the data to see we have not made any error

# In[4]:


print(sorted(data.columns))


# We can further look at price data as these are the data of interest

# In[5]:


price_data = data.SalePrice
print(price_data.head())


# The columns of interest are the ones that help to predict the price. Intuitively, they should be at least the following

# In[9]:


columns_of_interest = ['TotRmsAbvGrd', 'TotRmsAbvGrd']
two_columns_of_data = data[columns_of_interest]
two_columns_of_data.describe()


# AS usual, we denote by y the variable of interest that we want to predict

# In[10]:


y= data.SalePrice


# We also regroup our predictors

# In[11]:


predictors = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 
              'BedroomAbvGr', 'TotRmsAbvGrd']
X = data[predictors]


# We will now use a Decision Tree Regressor model to make some prediction!

# We use sklearn library which is very easy to use.... We will call the fit function method to estimate the parameters of our model as follows:

# In[13]:


from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(X,y)


# Once we have estimated our model. We call it, we have **'trained'** our model, we can use it and make some prediction as follows;

# In[15]:


print('Making predictions for the following 5 houses:')
print(X.head())
print("the predictions are")
print(model.predict(X.head()))


# However, we are missing a point here. We need to avoid overfitting. In order to do so, we need to split our data set into a training and a validation sample set. This can be easily done with the scikit learn function train_test_split as follows

# In[16]:


from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)


# Once this is done, we redo as previously and train (or fit) our model

# In[17]:


model = DecisionTreeRegressor()
model.fit(train_X, train_y)


# We can again make some predictions

# In[ ]:


val_predictions = model.predict(val_X)


# We can also display some error metrics as follows

# In[19]:


from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(val_y, val_predictions))


# If we want something more fancier, we can write some simple functions to display some more statistics

# In[20]:


from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

def get_mae(max_leaf_nodes, train_X, train_y, test_X, test_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=42)
    model.fit(train_X, train_y)
    y_pred = model.predict(test_X)
    return mean_absolute_error(test_y, y_pred)

for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae= get_mae(max_leaf_nodes, train_X, train_y, val_X, val_y)
    print( "Max leaf nodes: %d \\ Mean absolute Error %d" %(max_leaf_nodes, my_mae))


# # A better model
# sofar, we have used one of the simplest model, the decision tree regressor. We can use ensemble models like RandomForest Regressor that performs better. Random Forest is in a nutshell similar to more advanced models like xgboost or lightgbm that are used to win Kaggle competitions. Random forest combines various decision tree models to avoid overffitting and making it more robust. It works as follows

# In[21]:


from sklearn.ensemble import RandomForestRegressor

for max_leaf_nodes in [5, 50, 500, 5000]:
    model2 = RandomForestRegressor(max_leaf_nodes=max_leaf_nodes)
    model2.fit(train_X, train_y)
    y_preds = model2.predict(val_X)
    my_mae= mean_absolute_error(val_y, y_preds)
    print( "Max leaf nodes: %d \\ Mean absolute Error %d" %(max_leaf_nodes, my_mae))


# We can now do the full analysis to get more idea of our  RandomForest Regressor model and see that it works better as follows:

# In[23]:


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


# Once we are happy with our model, we can make a prediction to the Kaggle competition as follows

# In[24]:


# Read the test data
test = pd.read_csv('../input/test.csv')
# Treat the test data in the same way as training data. In this case, pull same columns.
test_X = test[predictor_cols]
# Use the model to make predictions
predicted_prices = my_model.predict(test_X)
# We will look at the predicted prices to ensure we have something sensible.
print(predicted_prices)


# ## Write your results for submission
# Once we have done all the analysis, we can write a file with our submission results and submit it to the Kaggle competition easily

# In[25]:


my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)


# At this stage we are done with the modelling and prediction... Now it will be your turn to play with Kaggle competition
# *** THIS IS THE END **
# 
