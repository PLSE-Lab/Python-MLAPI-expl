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
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer

main_file_path = '../input/train.csv'
data = pd.read_csv(main_file_path)
#print(data.columns)
y = data.SalePrice
#print(price.head())
columns_of_interest = [
    'LotArea',
    'YearBuilt',
    '1stFlrSF',
    '2ndFlrSF',
    'FullBath',
    'BedroomAbvGr',
    'TotRmsAbvGrd'
]
X = data[columns_of_interest]
model = RandomForestRegressor()
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)

my_imputer = Imputer()
imputed_X_train = my_imputer.fit_transform(train_X)
imputed_X_test = my_imputer.transform(val_X)
model.fit(imputed_X_train,train_y)
val_predicts = model.predict(imputed_X_test)
print("Mean Absolute Error from Imputation:")
#print(score_dataset(imputed_X_train, imputed_X_test, y_train, y_test))
mean_absolute_error(val_y, val_predicts)
#print(y.head())


# In[ ]:




