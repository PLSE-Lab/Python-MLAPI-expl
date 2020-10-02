#################
### My Koryto ###
#################

## XGBoost ML model to predict the Titanic competition results.

## Setup

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Read the data
X = pd.read_csv('../input/modified-titanic-train-dataset/train.csv', index_col='PassengerId')
X_test_full = pd.read_csv('../input/modified-test-titanic-data/test.csv', index_col='PassengerId')

X.dropna(axis=0, subset=['Survived'], inplace=True)
y = X.Survived              
X.drop(['Survived'], axis=1, inplace=True)

# Break off validation set from training data
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                                random_state=0)
# Select low cardinality columns
low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and 
                        X_train_full[cname].dtype == "object"]

# Select numeric columns
numeric_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

# Keep selected columns only
my_cols = low_cardinality_cols + numeric_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()

# One-hot encode the data
X_train = pd.get_dummies(X_train)
X_valid = pd.get_dummies(X_valid)
X_test = pd.get_dummies(X_test)
X_train, X_valid = X_train.align(X_valid, join='left', axis=1)
X_train, X_test = X_train.align(X_test, join='left', axis=1)

## XGBoost

from xgboost import XGBRegressor

# Define the basic model
my_model_1 = XGBRegressor(random_state=0)

# Fit the model
my_model_1.fit(X_train, y_train)

## MAE check

from sklearn.metrics import mean_absolute_error

# Get predictions
predictions_1 = my_model_1.predict(X_valid)

# Calculate MAE
mae_1 = mean_absolute_error(predictions_1, y_valid)

print("Mean Absolute Error:" , mae_1)


## Improve the model

my_model_2 = XGBRegressor(n_estimators = 10000, learn_rate = 0.05)

# Fit the model
my_model_2.fit(X_train, y_train, early_stopping_rounds=10, eval_set=[(X_valid, y_valid)], verbose = False) 

# Get predictions
predictions_2 = my_model_2.predict(X_valid)

# Calculate MAE
mae_2 = mean_absolute_error(predictions_2, y_valid)
print("Mean Absolute Error:" , mae_2)

## Output to CSV
predtest = my_model_2.predict(X_test)

output = pd.DataFrame({'PassengerId': X_test.index,
                       'Survived': predtest})
output.to_csv('submission.csv', index=False)