# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split



#Now including categorical Data as well and with XGBoost model

train_data=pd.read_csv('../input/train.csv')
test_data=pd.read_csv('../input/test.csv')

train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)

target = train_data.SalePrice

cols_with_missing=[col for col in train_data.columns if train_data[col].isnull().any()]
candidate_train_predictors=train_data.drop(['Id','SalePrice']+cols_with_missing, axis=1)
candidate_test_predictors=test_data.drop(['Id']+cols_with_missing,axis=1)

low_cardinality_cols=[cname for cname in candidate_train_predictors.columns if candidate_train_predictors[cname].nunique()<10 and candidate_train_predictors[cname].dtype=='object']
numeric_cols=[cname for cname in candidate_train_predictors.columns if candidate_train_predictors[cname].dtype in ['int64','float64']]

my_columns=low_cardinality_cols+numeric_cols
train_predictors=candidate_train_predictors[my_columns]
test_predictors=candidate_test_predictors[my_columns]

one_hot_encoded_training_predictors = pd.get_dummies(train_predictors)
one_hot_encoded_test_predictors = pd.get_dummies(test_predictors)
final_train, final_test = one_hot_encoded_training_predictors.align(one_hot_encoded_test_predictors, join='left', axis=1)


from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer()
final_train = my_imputer.fit_transform(final_train)
final_test = my_imputer.transform(final_test)

train_X, test_X, train_y, test_y = train_test_split(final_train, target, test_size=0.25)

#my_model=XGBRegressor()
#my_model.fit(final_train,target, verbose=False)
from sklearn.metrics import mean_absolute_error

my_model = XGBRegressor(n_estimators=1500, learning_rate=0.05)
my_model.fit(train_X, train_y, early_stopping_rounds=5, eval_set=[(test_X, test_y)], verbose=False)
predictions = my_model.predict(test_X)
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))
prediction=my_model.predict(final_test)
output=pd.DataFrame({'Id':test_data.Id, 'SalePrice':prediction})
output.to_csv('submission_categorical_XGBoost.csv',index=False)


import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.