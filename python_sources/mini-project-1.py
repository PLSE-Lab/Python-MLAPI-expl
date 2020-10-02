#!/usr/bin/env python
# coding: utf-8

# ## **Selecting Data for Modeling**
#  
# We'll start by picking a few variables using our intuition
# To choose variables/columns, we'll need to see a list of all columns in the dataset. That is done with the columns property of the DataFrame (the bottom line of code below).
# 
# ## **Selecting The Prediction Target**
# 
# We can pull out a variable with dot-notation. This single column is stored in a Series, which is broadly like a DataFrame with only a single column of data.
# 
# We'll use the dot notation to select the column we want to predict, which is called the prediction target. By convention, the prediction target is called y. So the code we need to save the house prices in the iowa data is

# In[ ]:


import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


# Path of the file to read
iowa_file_path = '../input/home-data-for-ml-course/train.csv'

home_data = pd.read_csv(iowa_file_path)
# Create target object and call it y
y = home_data.SalePrice


# ## Choosing "Features"
# The columns that are inputted into our model (and later used to make predictions) are called "features." In our case, those would be the columns used to determine the home price. Sometimes, you will use all columns except the target as features. Other times you'll be better off with fewer features.
# 
# For now, we'll build a model with only a few features. 
# 
# We select multiple features by providing a list of column names inside brackets. Each item in that list should be a string (with quotes).

# In[ ]:


# Create X
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[features]


# Let's quickly review the data we'll be using to predict house prices using the describe method and the head method, which shows the top few rows.

# In[ ]:


#X.describe()
X.head()


# In[ ]:


# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Specify Model
iowa_model = DecisionTreeRegressor(random_state=1)
# Fit Model
iowa_model.fit(train_X, train_y)

# Make validation predictions and calculate mean absolute error
val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE: {:,.0f}".format(val_mae))

# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.machine_learning.ex5 import *
#print("\nSetup complete")


# In[ ]:


def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)


# ## Compare Different Tree Sizes
# A loop that tries the following values for *max_leaf_nodes* from a set of possible values.
# 
# Call the *get_mae* function on each value of max_leaf_nodes. Store the output in some way that allows us to select the value of `max_leaf_nodes` that gives the most accurate model on your data.

# In[ ]:


candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
# loop to find the ideal tree size from candidate_max_leaf_nodes
for x in candidate_max_leaf_nodes:
    my_mae = get_mae(x,train_X,val_X,train_y,val_y)
    print("MAX leaf nodes: %d \t Mean absolute error: %d"%(x,my_mae))
    #score=min(x)

# Store the best value of max_leaf_nodes (it will be either 5, 25, 50, 100, 250 or 500)
best_tree_size = 100

#step_1.check()


# ## Fit Model Using All Data
# We know the best tree size. If we were going to deploy this model in practice, we would make it even more accurate by using all of the data and keeping that tree size.  That is, we don't need to hold out the validation data now that we've made all your modeling decisions.

# In[ ]:


final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size,random_state=1)
# fit the final model
final_model.fit(X,y)


# Many machine learning models allow some randomness in model training. Specifying a number for random_state ensures we get the same results in each run. This is considered a good practice. we use any number, and model quality won't depend meaningfully on exactly what value you choose.
# 
# We now have a fitted model that we can use to make predictions.
# 
# In practice, we'll want to make predictions for new houses coming on the market rather than the houses we already have prices for. But we'll make predictions for the first few rows of the training data to see how the predict function works.

# In[ ]:


print("Making predictions for the following 5 houses:")
#print(X.head())
print(y.head())
print("The predictions are")
print(iowa_model.predict(X.head()))

