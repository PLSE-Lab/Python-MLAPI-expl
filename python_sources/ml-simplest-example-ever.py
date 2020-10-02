#!/usr/bin/env python
# coding: utf-8

# # Introduction
# This code shows how to make a simple classifier using `pandas` and `sklearn` 
# 
# ## Write Your Code Below

# In[ ]:


import pandas as pd

main_file_path = '../input/house-prices-advanced-regression-techniques/train.csv' # this is the path to the Iowa data that you will use
data = pd.read_csv(main_file_path)

# print a summary of the data
data.describe()


# In[ ]:


# defining prediction target as 'y'
y = data.SalePrice
y.head()


# In[ ]:


# choosing predictors (columns that will be used to predict)
predictors = ["LotArea",
                "YearBuilt",
                "1stFlrSF",
                "2ndFlrSF",
                "FullBath",
                "BedroomAbvGr",
                "TotRmsAbvGrd"]

X = data[predictors]
X.head()


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error


# split data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

# define model
my_model = DecisionTreeRegressor()

# fit model
my_model.fit(train_X, train_y)

# make predictions
val_predictions = my_model.predict(val_X)

# mean absolute error
print("MEAN ABSOLUTE ERROR: ")
print(mean_absolute_error(val_y, val_predictions))


# In[ ]:


# Read the test data
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

# Treat the test data in the same way as training data. In this case, pull same columns.
test_X = test[predictors]

# Use the model to make predictions
predicted_prices = my_model.predict(test_X)

# We will look at the predicted prices to ensure we have something sensible.
print(predicted_prices)

my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})

# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)

