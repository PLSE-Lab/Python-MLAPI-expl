# In this kernel, I will use Linear Regression to predict house prices
# I learned the techniques I use in this kernel from DataCamp

# import necessary packages
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# import data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# divide data into predictor and target variables
train_X = train.drop('SalePrice', axis=1)
train_y = train.SalePrice
test_X = test

# one-hot encoding categorical variables for analysis
onehot_train_X = pd.get_dummies(train_X)
onehot_test_X = pd.get_dummies(test_X)
train_X, test_X = onehot_train_X.align(onehot_test_X, join='left', axis=1)

# impute missing values with the column's mean value
my_imputer = SimpleImputer()
train_X = my_imputer.fit_transform(train_X)
test_X = my_imputer.transform(test_X)

# use cross-validation and print the scores for each
reg = LinearRegression()
cv_scores = cross_val_score(reg, train_X, train_y, cv=5)
print(cv_scores)

# define the model
reg = LinearRegression()
reg.fit(train_X, train_y)
predictions = reg.predict(test_X)

# creating the submission file
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice':predictions})
my_submission.to_csv('submission.csv', index=False)

