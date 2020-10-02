# In this kernel, I will use Lasso Regression to predict house prices
# I learned the techniques I use in this kernel from DataCamp

# The Lasso regression technique will select important features of the dataset.
# It shrinks the coefficients of less important features to 0, so they have no impact.


# import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score, GridSearchCV

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

# Fine-tuning the alpha value for lasso regression
alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
param_grid = {'alpha':alphas}
lasso = Lasso()
lasso_cv = GridSearchCV(lasso, param_grid, cv=5)
lasso_cv.fit(train_X, train_y)
print("Tuned lasso regression parameters: {}".format(lasso_cv.best_params_))
print("Best score is {}".format(lasso_cv.best_score_))

   

# IMPLEMENTING LASSO REGRESSION
lasso = Lasso(alpha = 0.1, normalize=True)
lasso.fit(train_X, train_y)
lasso_coef = lasso.coef_
print(lasso_coef)
preds = lasso.predict(test_X)


# preparing predictions for submission
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice':preds})
my_submission.to_csv('submission.csv', index=False)








