# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

#kivaschenko
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

path_train = '../input/train.csv'
path_test = '../input/test.csv'
path_submission = '../input/sample_submission.csv'
path_predict = 'predict_submission.csv'

train = pd.read_csv(path_train)
test = pd.read_csv(path_test)
submission = pd.read_csv(path_submission)

# Created the target data array: y_true
y_true = train.SalePrice.apply(np.log1p)

# Count the columns where more nulls: columns_isnull
columns_isnull = []
for col in train.columns:
    isnul = train[col].isnull().sum()
    lendth = len(train[col])
    il = isnul/lendth
    if il >= .5:
        columns_isnull.append(col)
print("Columns that have more nulls: {}".format(columns_isnull))

# Concatenate both sets train and test to df: df
df = pd.concat([train, test])

# Dropping the columns where more part of values are nulls:
df.drop(columns_isnull, axis=1, inplace=True)

# Slising from df only object data types columns: object_df
object_df = df.select_dtypes(include=['object']).copy()

# Create numerical features from categorical:
dummie = pd.get_dummies(object_df, drop_first=True)
df = pd.concat([df, dummie], axis=1, join_axes=[df.index])

# Dropping the object columns:
object_columns = object_df.columns
df.drop(object_columns, axis=1, inplace=True)

# Create new train set: train
train = df[df.SalePrice.notnull()]

# Create new test set: test
test = df[df.SalePrice.isnull()]

# Drop 'Id' and 'SalePrice' columns:
train.drop(['Id','SalePrice'], axis=1, inplace=True)
test.drop(['Id','SalePrice'], axis=1, inplace=True)

# Check the similarity of the training and test sets with the number of columns
print('The number of columns of train and test set is similar? --', \
      train.shape[1] == test.shape[1])

# Assign main variates to predict model:     
X = train.values
X_pred = test.values
y = np.ravel(y_true.values)
print(
    "X.shape: {}".format(X.shape),
    "X_pred.shape: {}".format(X_pred.shape),
    "y.shape: {}".format(y.shape)
)
# Check target array:
print("The target array: {}".format(y))

# Assigne Imputer class: imp
imp = Imputer(missing_values='NaN', strategy='median', axis=0)

# Impute the missing data
X_imputed = imp.fit_transform(X)
X_pred_imputed = imp.fit_transform(X_pred)

# Create pipeline: pipe
pipe = Pipeline(
	[
	('scaler', StandardScaler()), 
	('regressor', RandomForestRegressor())
	]
)

# Define params to grid search checking: param_grid
param_grid = {
    'regressor__n_estimators': [x for x in range(10, 160, 10)],
    'regressor__max_depth': [x for x in range(4, 13, 1)]
}

# Split data set
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, random_state=1)

# Make grid search: grid
grid = GridSearchCV(pipe, param_grid, cv=10)
grid.fit(X_train, y_train)

print("Test set score: {:.5f}".format(grid.score(X_test, y_test)))
print("Best parameters: {}".format(grid.best_params_))
print("Best cross-validation score: {:.5f}".format(grid.best_score_))
print("Best estimator:\n{}".format(grid.best_estimator_))

# Predict and write the result
predicted_y = grid.best_estimator_.predict(X_pred_imputed)
submission.SalePrice = np.expm1(predicted_y)
submission.to_csv(path_predict, header=True, index=False)