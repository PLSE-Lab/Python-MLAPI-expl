# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
X = pd.read_csv('../input/home-data-for-ml-course/train.csv', index_col='Id') 
X_test = pd.read_csv('../input/home-data-for-ml-course/test.csv', index_col='Id')

# Remove rows with missing target, separate target from predictors
X.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X.SalePrice
X.drop(['SalePrice'], axis=1, inplace=True)

# To keep things simple, we'll drop columns with missing values
cols_with_missing = [col for col in X.columns if X[col].isnull().any()] 
X.drop(cols_with_missing, axis=1, inplace=True)
X_test.drop(cols_with_missing, axis=1, inplace=True)

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y,
                                                      train_size=0.8, test_size=0.2,
                                                      random_state=0)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# function for comparing different approaches
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)

# Fill in the lines below: drop columns in training and validation data
drop_X_train = X_train.select_dtypes(exclude = 'object')
drop_X_valid = X_valid.select_dtypes(exclude = 'object')

object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]

# Columns that can be safely label encoded
good_label_cols = [col for col in object_cols if 
                   set(X_train[col]) == set(X_valid[col])]
        
# Problematic columns that will be dropped from the dataset
bad_label_cols = list(set(object_cols)-set(good_label_cols))

from sklearn.preprocessing import LabelEncoder

# Drop categorical columns that will not be encoded
label_X_train = X_train.drop(bad_label_cols, axis=1)
label_X_valid = X_valid.drop(bad_label_cols, axis=1)

# Apply label encoder 
my_labelEncoder = LabelEncoder()
for col in good_label_cols:
    label_X_train[col] = my_labelEncoder.fit_transform(label_X_train[col])
    label_X_valid[col] = my_labelEncoder.transform(label_X_valid[col])

object_nunique = list(map(lambda col: X_train[col].nunique(), object_cols))
d = dict(zip(object_cols, object_nunique))

# Print number of unique entries by column, in ascending order
sorted(d.items(), key=lambda x: x[1])


low_cardinality_cols = [col for col in object_cols if X_train[col].nunique() < 10]

# Columns that will be dropped from the dataset
high_cardinality_cols = list(set(object_cols)-set(low_cardinality_cols))

print('Categorical columns that will be one-hot encoded:', low_cardinality_cols)
print('\nCategorical columns that will be dropped from the dataset:', high_cardinality_cols)


from sklearn.preprocessing import OneHotEncoder

# Use as many lines of code as you need!
# Apply one-hot encoder to each column with categorical data
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[low_cardinality_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[low_cardinality_cols]))


# One-hot encoding removed index; put it back
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

# Remove categorical columns (will replace with one-hot encoding)
num_X_train = X_train.drop(object_cols, axis=1)
num_X_valid = X_valid.drop(object_cols, axis=1)

# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

X_test.fillna(method ='pad', inplace=True)
OHcolstest = pd.DataFrame(OH_encoder.transform(X_test[low_cardinality_cols]))
OHcolstest.index = X_test.index
numXtest = X_test.drop(object_cols, axis=1).fillna(0)

from xgboost import XGBRegressor
model = XGBRegressor(n_estimators = 800,learning_rate = 0.05, max_depth=3, min_child_weight=1,gamma=0,
     subsample=0.8,
     colsample_bytree=0.8)
model.fit(OH_X_train, y_train, early_stopping_rounds=5, 
             eval_set=[(OH_X_valid, y_valid)], 
             verbose=False)
OH_X_test = pd.concat([numXtest, OHcolstest], axis=1)
prediction = model.predict(OH_X_test)

output = pd.DataFrame({'Id': OH_X_test.index,
                       'SalePrice': prediction})
output.to_csv('submission.csv', index=False)
print("End of Code")

"""
######################################## tuning hyperparameters using GridSearchCV #####################
from sklearn.model_selection import GridSearchCV
param_test1 = {
    'subsample':[i/10.0 for i in range(6,10)],
    'colsample_bytree':[i/10.0 for i in range(6,10)]
}

gsearch1 = GridSearchCV(estimator = XGBRegressor(n_estimators = 800,learning_rate = 0.05, max_depth=3, min_child_weight=1,gamma=0,
     subsample=0.6,
     colsample_bytree=0.6), param_grid = param_test1,n_jobs=4,iid=False, cv=5)
gsearch1.fit(OH_X_train, y_train)

print(gsearch1.cv_results_)



param_test1 = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
}

from sklearn.model_selection import GridSearchCV
gsearch1 = GridSearchCV(estimator = XGBRegressor(n_estimators = 800,learning_rate = 0.05, max_depth=6, min_child_weight=5,gamma=0,
     subsample=0.8,
     colsample_bytree=0.8), param_grid = param_test1,n_jobs=4,iid=False, cv=5)
gsearch1.fit(OH_X_train, y_train)

print(gsearch1.cv_results_)
, gsearch1.best_params_, gsearch1.best_score_


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
def get_score(n_estimators):
    # Replace this body with your own code
    my_pipeline = Pipeline(steps=[('model', XGBRegressor(n_estimators = n_estimators,learning_rate = 0.025, random_state = 0))])
    scores = -1 * cross_val_score(my_pipeline, OH_X_train, y_train,cv=3,scoring='neg_mean_absolute_error')
    return scores.mean()

results = {}
for i in range(1,20):
    results[50*i] = get_score(50*i)


import matplotlib.pyplot as plt


plt.plot(results.keys(), results.values())
plt.show()
print("End of Code")

"""