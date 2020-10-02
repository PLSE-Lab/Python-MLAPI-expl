import pandas as pd
import numpy as np

from sklearn.preprocessing import Imputer
from xgboost import XGBRegressor

train_input_data = pd.read_csv('../input/train.csv')
test_input_data = pd.read_csv('../input/test.csv')

train_input_data.dropna(axis = 0 , subset = ['Survived'] , inplace = True)
train_Survived = train_input_data.Survived

cols_with_missing_values = [cols for cols in train_input_data.columns
                            if train_input_data[cols].isnull().any()]
                            
predictors_train = train_input_data.drop(['PassengerId' , 'Survived'] + cols_with_missing_values , axis = 1)
predictors_test = test_input_data.drop(['PassengerId'] + cols_with_missing_values , axis = 1)

low_cardinality_cols = [cname for cname in predictors_train.columns
                        if predictors_train[cname].nunique() < 5 and
                        predictors_train[cname].dtypes == 'object']

numeric_cols = [cname for cname in predictors_train.columns
                if predictors_train[cname].dtypes != 'object']

cols_predictors = low_cardinality_cols + numeric_cols

actual_predictors_train = predictors_train[cols_predictors]
actual_predictors_test = predictors_test[cols_predictors]

one_hot_encoded_actual_predictors_train = pd.get_dummies(actual_predictors_train)
one_hot_encoded_actual_predictors_test = pd.get_dummies(actual_predictors_test)

final_predictors_train , final_predictors_test = one_hot_encoded_actual_predictors_train.align(one_hot_encoded_actual_predictors_test , join = 'inner' , 
                                                axis = 1)

titanic_model = XGBRegressor()
titanic_model.fit(final_predictors_train , train_Survived)

predict_Survived = titanic_model.predict(final_predictors_test)
rounded_predict_Survived = predict_Survived.round()
integer_predict_Survived = rounded_predict_Survived.astype(int)

titanic_model_submission = pd.DataFrame({'PassengerId': test_input_data.PassengerId , 
                                        'Survived': integer_predict_Survived})
titanic_model_submission.to_csv('titanic_model_submission.csv' , index = False)