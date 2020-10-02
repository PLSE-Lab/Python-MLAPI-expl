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
#print('hello world')


# In[ ]:


print(data.describe())


# In[ ]:


#Print a list of the columns
print(data.columns)


# In[ ]:


#From the list of columns, find a name of the column with the sales prices of the homes.
SalePrice_data = data.SalePrice
print(SalePrice_data.head())


# In[ ]:


#Pick any two variables and store them to a new DataFrame 
columns_of_interest = ['LotArea','GrLivArea']
two_columns_of_data = data[columns_of_interest]


# In[ ]:


#We can verify that we got the columns we need with the describe command.
two_columns_of_data.describe()


# In[ ]:


#Selecting the target variable which I wants to predict. 
y = SalePrice_data
#Create a list of the names of the predictors we will use in the initial model.
predictors = ['LotArea',
'YearBuilt',
'1stFlrSF',
'2ndFlrSF',
'FullBath',
'BedroomAbvGr',
'TotRmsAbvGrd']

X = data[predictors]


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
# split data into training and validation data, for both predictors and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X,y,random_state=0)

#Define model
Iowa_housing_model = DecisionTreeRegressor()

#fit model
Iowa_housing_model.fit(train_X,train_y)

# get predicted prices on validation data
val_predictions = Iowa_housing_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))


# In[ ]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

def get_mae(max_leaf_nodes, predictors_train, predictors_val, target_train,target_val):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(predictors_train,target_train)
    preds_val = model.predict(predictors_val)
    mae = mean_absolute_error(target_val, preds_val)
    return (mae)


# In[ ]:


#compare mean_absolute_error with different tree leaf nodes
for max_leaf_nodes in [5,50,500,5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print('max_leaf_nodes:{}\t\t mean_absolute_error:{}'.format(max_leaf_nodes, my_mae))
    


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor()
forest_model.fit(train_X, train_y)
Iowa_model_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, Iowa_model_preds))


# In[ ]:


#read the test data
test = pd.read_csv('../input/test.csv')

#pull same columns for prediction test
test_X = test[predictors]

#use the model to predict prices
predicted_prices = forest_model.predict(test_X)

#print predicted prices
print(predicted_prices)


# In[ ]:


#submit prediction for test data
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
my_submission.to_csv('submission.csv', index = False)

