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

# In[ ]:


import pandas as pd
main_file_path = '../input/train.csv'
data = pd.read_csv(main_file_path)

#Column Of enire DataSet - 
print(data.columns)


# In[ ]:


# House sell Price -> Predict vector
home_SalePrice = data.SalePrice
home_SalePrice.head()


# In[ ]:


# Select Important Clumn
column_of_interest = ['YearBuilt','1stFlrSF','2ndFlrSF','FullBath','TotRmsAbvGrd','LotArea','OverallCond','TotalBsmtSF','BedroomAbvGr']
#column_of_interest = [x for x in data.columns if data[x].dtype == int or data[x].dtype == float]
#column_of_interest.pop()
coi_data = data[column_of_interest]
#coi_data.dropna()


# In[ ]:


# Predict Vector Y
y = home_SalePrice

# Traing Data
X = coi_data


# In[ ]:


from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(X,y)


# In[ ]:


print(data.SalePrice.head())
model.predict(X.head())


# In[ ]:


from sklearn.metrics import mean_absolute_error

predicted_home_prices = model.predict(X)
mean_absolute_error(y, predicted_home_prices)


# In[ ]:


from sklearn.model_selection import train_test_split

# split data into training and validation data, for both predictors and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)
# Define model
model = DecisionTreeRegressor()
# Fit model
model.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))


# In[ ]:


from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

def get_mae(max_leaf_nodes, predictors_train, predictors_val, targ_train, targ_val):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(predictors_train, targ_train)
    preds_val = model.predict(predictors_val)
    mae = mean_absolute_error(targ_val, preds_val)
    return(mae)


# In[ ]:


import matplotlib.pyplot as plt

# compare MAE with differing values of max_leaf_nodes
x_Me, y_Me = [],[]
for max_leaf_nodes in range(40,60):
#     global x_Me,y_Me
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
    x_Me = x_Me + [max_leaf_nodes]
    y_Me = y_Me + [my_mae]
plt.plot(x_Me, y_Me)
plt.show()
print(min(y_Me))


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import  mean_absolute_error

RF_model = RandomForestRegressor()
RF_model.fit(train_X, train_y)
house_predict = RF_model.predict(val_X)
print(mean_absolute_error(val_y, house_predict))


# In[ ]:


# Read the test data
test = pd.read_csv('../input/test.csv')
# Treat the test data in the same way as training data. In this case, pull same columns.
bool_na = test['TotalBsmtSF'].isnull()
test_X = test[column_of_interest]
test_X = test_X.fillna(0)
# Use the model to make predictions
predicted_prices = RF_model.predict(test_X)
# We will look at the predicted prices to ensure we have something sensible.
print(predicted_prices)


# In[ ]:


my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
# you could use any filename. We choose submission here
my_submission.to_csv('House Prices - Random_forest.csv', index=False)


# In[ ]:




