#!/usr/bin/env python
# coding: utf-8

# # Intro
# **This is your workspace for Kaggle's Machine Learning course**
# 
# You will build and continually improve a model to predict housing prices as you work through each tutorial.
# 
# The tutorials you read use data from Melbourne. The Melbourne data is not available in this workspace.  Instead, you will translate the concepts to work with the data in this notebook, the Iowa data.
# 
# # Write Your Code Below
# 

# In[3]:


import pandas as pd

main_file_path = '../input/train.csv' # this is the path to the Iowa data that you will use
data = pd.read_csv(main_file_path)

# The cod below will help you see how output appears when you run a code block
# Run this code block with the control-enter keys on your keyboard. Or click the blue botton on the left
#print('hello world')
#pd.set_option('display.max_columns', 500)
data.describe() 


# 
# 
# **If you have any questions or hit any problems, come to the [Learn Discussion](https://www.kaggle.com/learn-forum) for help. **
# 
# **Return to [ML Course Index](https://www.kaggle.com/learn/machine-learning)**

# 

# In[4]:


data.columns


# In[5]:


price_data = data.SalePrice
print(price_data.head())


# In[6]:


columns_of_interest = ['MoSold', 'LotArea']
two_columns_of_data = data[columns_of_interest]


# In[7]:


two_columns_of_data.describe()


# In[8]:


y = data.SalePrice


# In[9]:


columns_of_interest = ['LotArea', 'YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']
X= data[columns_of_interest]


# In[10]:


from sklearn.tree import DecisionTreeRegressor

# Define model
model = DecisionTreeRegressor()

# Fit model
model.fit(X, y)


# In[11]:


print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(model.predict(X.head()))


# In[12]:


from sklearn.metrics import mean_absolute_error

predicted_home_prices = model.predict(X)
mean_absolute_error(y, predicted_home_prices)


# In[13]:


from sklearn.model_selection import train_test_split

# split data into training and validation data, for both predictors and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)
# Fit model
model.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))


# In[14]:


from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

def get_mae(max_leaf_nodes, predictors_train, predictors_val, targ_train, targ_val):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(predictors_train, targ_train)
    preds_val = model.predict(predictors_val)
    mae = mean_absolute_error(targ_val, preds_val)
    return(mae)


# In[16]:


for max_leaf_nodes in [5, 50, 100, 500, 2500,5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))


# In[17]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor()
forest_model.fit(train_X, train_y)
preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, preds))


# In[18]:


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


# In[19]:


# Read the test data
test = pd.read_csv('../input/test.csv')
# Treat the test data in the same way as training data. In this case, pull same columns.
test_X = test[predictor_cols]
# Use the model to make predictions
predicted_prices = my_model.predict(test_X)
# We will look at the predicted prices to ensure we have something sensible.
print(predicted_prices)


# In[20]:


my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)


# In[ ]:




