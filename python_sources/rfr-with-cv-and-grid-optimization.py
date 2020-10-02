# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import numpy as np
from sklearn.cross_validation import StratifiedKFold, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import ParameterGrid

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

def rmsle(y_true, y_pred):
    assert y_true.size == y_pred.size
    terms_to_sum = [(np.log(y_pred + 1) - np.log(y_true + 1)) ** 2]
    return (np.sum(terms_to_sum) / y_true.size) ** 0.5
    
CLASS = False  # Whether classification or regression
SCORE_MIN = True  # Optimizing score through minimum
rfr = RandomForestRegressor(n_estimators=50)  # Regressor
k = 5  # Number of folds
best_score = 10
best_params = None

train_name = '../input/train.csv'
test_name = '../input/test.csv'
submission_name = '../input/sample_submission.csv'
submission_col = 'SalePrice'
submission_target = 'test_sub1.csv'

# Read files
train = pd.DataFrame.from_csv(train_name)
train = train.fillna(-1)
test = pd.DataFrame.from_csv(test_name)
test = test.fillna(-1)
submission = pd.DataFrame.from_csv(submission_name)
# Extract target
target = train['SalePrice']
del train['SalePrice']

# Label nominal variables to numbers
columns = train.columns.values
nom_numeric_cols = ['MSSubClass']
dummy_train = []
dummy_test = []
for col in columns:
    # Only works for nominal data without a lot of factors
    if train[col].dtype.name == 'object' or col in nom_numeric_cols:
        dummy_train.append(pd.get_dummies(train[col].values.astype(str), col))
        dummy_train[-1].index = train.index
        dummy_test.append(pd.get_dummies(test[col].values.astype(str), col))
        dummy_test[-1].index = test.index
        del train[col]
        del test[col]
train = pd.concat([train] + dummy_train, axis=1)
test = pd.concat([test] + dummy_test, axis=1)

# Use only common columns
columns = []
for col_a in train.columns.values:
    if col_a in test.columns.values:
        columns.append(col_a)
train = train[columns]
test = test[columns]

# CV
train = np.array(train)
target = np.array(target)
test = np.array(test)
print(train.shape, test.shape)

if CLASS:
    kfold = StratifiedKFold(target, k)
else:
    kfold = KFold(train.shape[0], k)

param_grid = {'max_features': [0.1, 0.2, 0.4], 'max_depth': [20, 40, 60]}

for params in ParameterGrid(param_grid):
    print(params)
    rfr.max_depth = params['max_depth']
    rfr.max_features = params['max_features']
    rmsle_score = []
    for train_index, test_index in kfold:
        X_train, X_test = train[train_index], train[test_index]
        y_train, y_test = target[train_index], target[test_index]
        rfr.fit(X_train, y_train)
        rmsle_score.append(rmsle(y_test, rfr.predict(X_test)))
    if SCORE_MIN:
        if best_score > np.mean(rmsle_score):
            print(np.mean(rmsle_score))
            print('new best')
            best_score = np.mean(rmsle_score)
            best_params = params
    else:
        if best_score < np.mean(rmsle_score):
            print(np.mean(rmsle_score))
            print('new best')
            best_score = np.mean(rmsle_score)
            best_params = params

rfr.n_estimators = 300
rfr.max_depth = best_params['max_depth']
rfr.max_features = best_params['max_features']
rfr.fit(train, target)
submission[submission_col] = rfr.predict(test)
submission.to_csv(submission_target)

print('Done!')
# Any results you write to the current directory are saved as output.