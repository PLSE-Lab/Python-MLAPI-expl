import numpy as np 
import pandas as pd 
import os

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

#features - Pclass, Sex, Age, Sib Sp, Parch, Fare
y_train = train.iloc[:,[1]].values

#Encode the categorical data
train_predictors = train[train.columns[[2,4,5,6,7,9]]]
test_predictors = test[test.columns[[1,3,4,5,6,8]]]

one_hot_encoded_training_predictors = pd.get_dummies(train_predictors)
one_hot_encoded_test_predictors = pd.get_dummies(test_predictors)
X_train, X_test = one_hot_encoded_training_predictors.align(one_hot_encoded_test_predictors, join='left', axis=1)

#Feature Scaling
from sklearn.preprocessing import Imputer
my_imputer = Imputer()
X_train = my_imputer.fit_transform(X_train)
X_test = my_imputer.transform(X_test)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
y_pred1 = []
for i in y_pred:
    if i[0] < 0.5:
        y_pred1.append(0)
    else:
        y_pred1.append(1)
        
my_submission = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': y_pred1})
my_submission.to_csv('submission.csv', index=False)