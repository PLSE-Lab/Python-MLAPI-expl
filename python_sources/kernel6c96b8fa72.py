# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from itertools import product

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

#import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

# Read data and set dataframes
X = pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv')
X_test = pd.read_csv('/kaggle/input/home-data-for-ml-course/test.csv')

COLUMNS_TO_DROP = ['MSZoning', 'KitchenQual']
MAX_CARDINALITY_FEATURES = 8

X.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X.SalePrice
X.drop(['SalePrice'], axis=1, inplace=True)

X.drop(COLUMNS_TO_DROP, axis=1, inplace=True)
X_test.drop(COLUMNS_TO_DROP, axis=1, inplace=True)

# Split in train and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=0)

# Select different kynds of columns
obj_cols = [col for col in X.columns if X[col].dtype == 'object']
num_cols = list(set(X.columns).difference(set(obj_cols)))
good_label_cols = [col for col in obj_cols if set(X_train[col]) == set(X_valid[col])]
low_cardinality_cols = [col for col in good_label_cols if (X[col].nunique() < MAX_CARDINALITY_FEATURES) and (not X[col].isnull().any())]

# Get model with numerical columns
num_X_train = X_train[num_cols]
num_X_valid = X_valid[num_cols]

num_X_test = X_test[num_cols]

# Get model with other columns
obj_X_train = X_train[low_cardinality_cols]
obj_X_valid = X_valid[low_cardinality_cols]

obj_X_test = X_test[low_cardinality_cols]

# Preprocess phase 1: impute numerical columns with missing values
print('Preprocessing 1: imputing columns with missing values...')
my_imputer = SimpleImputer(strategy='median')
imp_X_train = pd.DataFrame(my_imputer.fit_transform(num_X_train))
imp_X_valid = pd.DataFrame(my_imputer.transform(num_X_valid))

imp_X_test = pd.DataFrame(my_imputer.transform(num_X_test))

# Fill in the lines below: imputation removed column names; put them back
imp_X_train.columns = num_X_train.columns
imp_X_valid.columns = num_X_valid.columns
imp_X_train.index = num_X_train.index
imp_X_valid.index = num_X_valid.index

imp_X_test.columns = num_X_test.columns
imp_X_test.index = num_X_test.index

# Preprocessing phase 2: select columns with categorical data for one-hot encoding
# Fit the one-hot encoder to the model
print('Preprocessing 2: encoding categorical data...')
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_X_train = pd.DataFrame(OH_encoder.fit_transform(obj_X_train))
OH_X_valid = pd.DataFrame(OH_encoder.transform(obj_X_valid))

for col in obj_X_test.columns:
    if (obj_X_test[col].isin([np.nan])).any():
        print('col:',col)
        print(obj_X_test[col][obj_X_test[col].isin([np.nan])])
        print()
        
OH_X_test = pd.DataFrame(OH_encoder.transform(obj_X_test))

# OH encoding removes index: put it back
OH_X_train.index = obj_X_train.index
OH_X_valid.index = obj_X_valid.index

OH_X_test.index = obj_X_test.index

# Get model with numerical (imputed) and encoded categorical columns
#imp_X_train = pd.DataFrame()
#imp_X_valid = pd.DataFrame()
#OH_X_train = pd.DataFrame()
#OH_X_valid = pd.DataFrame()
prep_X_train = pd.concat([imp_X_train, OH_X_train], axis=1)
prep_X_valid = pd.concat([imp_X_valid, OH_X_valid], axis=1)

prep_X_test = pd.concat([imp_X_test, OH_X_test], axis=1)

n_estimators_values = [800, 850, 900, 950, 1000]
max_leaf_nodes_values = [650, 700, 750, 800]
criterion_values = ['mse'] #['mae', 'mse']

results = dict()
best_val = 1e30
best_params = None

#print(imp_X_train.shape)
#print(OH_X_train.shape)
#print(prep_X_train.shape)
        
print('Starting validation...')
for n_estimators, max_leaf_nodes, criterion in product(n_estimators_values, max_leaf_nodes_values, criterion_values):
    
    params = tuple([n_estimators, max_leaf_nodes, criterion])
    if params not in results:
        model = RandomForestRegressor(random_state=0, n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, criterion=criterion)
        model.fit(prep_X_train, y_train)
        MEA = mean_absolute_error(model.predict(prep_X_valid), y_valid)
        
        results[params] = MEA
    else:
        MEA = results[params]
    
    print("n_est=%d, max_leaf=%d, crit=%s, MEA=%lf" % (n_estimators, max_leaf_nodes, criterion, MEA))
    if best_val > MEA:
        best_val = MEA
        best_params = [n_estimators, max_leaf_nodes, criterion]

# Print best parameters resulted
print(best_params, best_val)

# Fit final model
model = RandomForestRegressor(random_state=0, n_estimators=best_params[0], max_leaf_nodes=best_params[1], criterion=best_params[2])
model.fit(prep_X_train, y_train)

y_preds = model.predict(prep_X_test)

# The lines below shows how to save predictions in format used for competition scoring
# Just uncomment them.

output = pd.DataFrame({'Id': X_test.Id, 'SalePrice': y_preds})
output.to_csv('submission.csv', index=False)