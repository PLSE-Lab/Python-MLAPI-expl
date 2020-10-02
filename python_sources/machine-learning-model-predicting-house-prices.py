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
# 
# 

# In[ ]:


import pandas as pd

main_file_path = '../input/train.csv'
data = pd.read_csv(main_file_path)
print('hello world')


# In[ ]:


data.describe()


# In[ ]:


print(data.columns)


# In[ ]:


data.head()


# In[ ]:


data_saleprice=data['SalePrice']
data_saleprice.head()


# In[ ]:


col=['SalePrice', 'LotArea']
two_columns_of_data=data[col]


# In[ ]:


two_columns_of_data.head()


# In[ ]:


two_columns_of_data.describe()


# ### Prediction target (y)

# In[ ]:


y=data.SalePrice


# ### Predictors

# In[ ]:


import tensorflow as tf
# converting object into categorical
data['SaleCondition']=data['SaleCondition'].astype('category')
data['Utilities']=data['Utilities'].astype('category')
data['RoofMatl']=data['RoofMatl'].astype('category')


# In[ ]:


# converting categorical col into numerical
data['SaleCondition']=data['SaleCondition'].cat.codes
data['Utilities']=data['Utilities'].cat.codes
data['RoofMatl']=data['RoofMatl'].cat.codes
data.head()
print(data['SaleCondition'].unique())
print(data['Utilities'].unique())
print(data['RoofMatl'].unique())


# In[ ]:


data.dtypes


# In[ ]:


data.isnull().sum()


# In[ ]:


preds=['LotArea','Utilities','RoofMatl','YearBuilt','1stFlrSF', '2ndFlrSF','BedroomAbvGr','FullBath','KitchenAbvGr','SaleCondition','OverallQual','OverallCond','TotRmsAbvGrd']
X = data[preds]


# The steps to building and using a model are:
# 
# - Define: What type of model will it be? A decision tree? Some other type of model? Some other parameters of the model type are specified too.
# - Fit: Capture patterns from provided data. This is the heart of modeling.
# - Predict: Just what it sounds like
# - Evaluate: Determine how accurate the model's predictions are.

# ## Decision trees

# In[ ]:


from sklearn.tree import DecisionTreeRegressor

# define model
data_model = DecisionTreeRegressor()

# fit model
data_model.fit(X,y)


# In[ ]:


print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(data_model.predict(X.head()))


# In[ ]:


from sklearn.metrics import mean_absolute_error

predicted_home_prices = data_model.predict(X)
mean_absolute_error(y, predicted_home_prices)


# In[ ]:


from sklearn.model_selection import train_test_split

# split data into training and validation data, for both predictors and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we run the script
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)

# get predicted prices on validation data
val_predictions = data_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))


# When we divide the houses amongst many leaves, we also have fewer houses in each leaf. Leaves with very few houses will make predictions that are quite close to those homes' actual values, but they may make very unreliable predictions for new data (because each prediction is based on only a few houses).
# 
# This is a phenomenon called **overfitting**, where a model matches the training data almost perfectly, but does poorly in validation and other new data. On the flip side, if we make our tree very shallow, it doesn't divide up the houses into very distinct groups.
# 
# At an extreme, if a tree divides houses into only 2 or 4, each group still has a wide variety of houses. Resulting predictions may be far off for most houses, even in the training data (and it will be bad in validation too for the same reason). When a model fails to capture important distinctions and patterns in the data, so it performs poorly even in training data, that is called **underfitting.**
# ![Overfitting and underfitting](http://i.imgur.com/2q85n9s.png)

# In[ ]:


from sklearn.metrics import mean_absolute_error
def get_mae(max_leaf_nodes, predictors_train, predictors_val, targ_train, targ_val):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(predictors_train, targ_train)
    preds_val = model.predict(predictors_val)
    mae = mean_absolute_error(targ_val, preds_val)
    return(mae)


# In[ ]:


# compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))


# ## Random Forests

# In[ ]:


from sklearn.ensemble import RandomForestRegressor

forest_model = RandomForestRegressor()
forest_model.fit(train_X, train_y)
data_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, data_preds))


# In[ ]:


## Read the test data
test = pd.read_csv('../input/test.csv')


# In[ ]:


import tensorflow as tf
# converting object into categorical
test['SaleCondition']=test['SaleCondition'].astype('category')
test['SaleCondition']=test['SaleCondition'].cat.codes
test['RoofMatl']=test['RoofMatl'].astype('category')
test['RoofMatl']=test['RoofMatl'].cat.codes
test['Utilities']=test['Utilities'].astype('category')
test['Utilities']=test['Utilities'].cat.codes


# In[ ]:


test.dtypes


# In[ ]:


# Treat the test data in the same way as training data. In this case, pull same columns.
test_X = test[preds]
# Use the model to make predictions
predicted_prices = forest_model.predict(test_X)
# We will look at the predicted prices to ensure we have something sensible.
print(predicted_prices)


# ## Prepare Submission File
# 

# In[ ]:


my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
my_submission.to_csv('submission.csv', index=False)


# In[ ]:




