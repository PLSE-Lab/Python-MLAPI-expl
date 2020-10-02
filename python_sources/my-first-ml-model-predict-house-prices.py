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

# In[1]:


import pandas as pd

main_file_path = '../input/train.csv'
data = pd.read_csv(main_file_path)
print('hello world')


# In[3]:


# describe the data given
print(data.describe())


# In[4]:


# to only view the upper part of data instead of whole data
print(data.head())

# to know the columns in the given data
print(data.columns)


# In[2]:


# for filtering the data and selecting only two coloumns

two_col = ['LotArea', 'SalePrice']
two_col_data = data[two_col]
print(two_col_data.head())


# In[5]:


#modeling the data using only specific coloumns (using the above techniques)

#prediction target

y = data.SalePrice

SalePrice_predictors = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

#by convention, given data for modeling is represented with X

X = data[SalePrice_predictors]



# In[6]:


# building the model

from sklearn.tree import DecisionTreeRegressor

#define model
my_model = DecisionTreeRegressor()
#Fit model
my_model.fit(X,y)


# In[ ]:


#making predictions
print("Making the predictions for following house prices:")
print(X.head())
print("The predictions for first five houses are:")
print(my_model.predict(X.head()))
print("The predictions of House prices as follows:")
print(my_model.predict(X))



# In[ ]:


# saving the predicted prices
predicted_prices = my_model.predict(X)

#finding the absolute error
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y, predicted_prices)


# Up to the above, we have used the In-sample to check the absolute errors. Using the Same data for model building and predictions is not a recommended thing. As when we change the data, results will diverge a lot and will cost the company. 
# 
# Using the sklearn, train_test_split is a valid solution to separate the validation to check the accuracy of our model. The data separated from modeling building is used for validation, hence called** validation data.**
# 
# 
# 

# In[ ]:


from sklearn.model_selection import train_test_split

#this will split the data into training part and validation part. The training part of data
# is used to train the model and while validation part to verify the model using the validation data.

train_X, val_X, train_y, val_y = train_test_split(X,y,random_state=0)
# random_state ensures that we get same split everytime we use with same data
# it can be any random number

my_model = DecisionTreeRegressor()
#fit model
my_model.fit(train_X, train_y)
# predictions 
House_price_predictions = my_model.predict(val_X)
print(mean_absolute_error(val_y, House_price_predictions))


# In[ ]:





# In[ ]:


# adjusting the moaximum leaf nodes

from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

def get_mae(max_leaf_nodes, predict_train, predict_val, target_train, target_val):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(predict_train, target_train)
    predictions = model.predict(predict_val)
    mae = mean_absolute_error(target_val, predictions)
    return(mae)


# In[ ]:


for max_leaf_nodes in [5,36,40,50,70,100,500,1000,5000]:
    new_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max Leaf Nodes = %d \t\t Mean Absolute Error =%d" %(max_leaf_nodes, new_mae))


# In[ ]:


print(data.isnull().sum())

