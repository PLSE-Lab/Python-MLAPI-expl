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

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV# Create the parameter grid based on the results of random search 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (MinMaxScaler, LabelEncoder)
from sklearn.metrics import f1_score
from sklearn.linear_model import ElasticNetCV


def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    f1 = f1_score(test_labels, predictions)
    return f1

def mode_age_estimator(x, y):    
    model_enet = ElasticNetCV(cv=10)
    model_enet.fit(x, y) 
    return model_enet


if __name__ == "__main__":
    
    param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110, 150],
    'max_features': [2, 3, 4],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300]
    }
    
    # Loading data
    training_data = pd.read_csv('/kaggle/input/titanic/train.csv')
    test_data =  pd.read_csv('/kaggle/input/titanic/test.csv')
    
    training_data = training_data.drop(columns='Cabin')
    training_data = training_data.dropna()
    
    m = test_data.isna().any(axis=1)
    age_model = mode_age_estimator(training_data[['SibSp', 'Fare', 'Parch']],
                                   training_data['Age'])
    xage = test_data.loc[m, ['SibSp', 'Fare', 'Parch']]
    xage = xage.fillna(method='ffill')
    test_data.at[m, 'Age'] = age_model.predict(xage)
    
    X = training_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
    X_pred = test_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
    X_pred = X_pred.fillna(method='ffill')
    
    category_cols = X.select_dtypes(include=['object']).columns
    for col in category_cols:
        label_encoder = LabelEncoder().fit(X[col])
        X[col] = label_encoder.transform(X[col])
        X_pred[col] = label_encoder.transform(X_pred[col])
        
    y = pd.DataFrame(training_data['Survived'])  
        
    
    X_scaler = MinMaxScaler().fit(X)
    X.loc[:, :] = X_scaler.transform(X)
    X_pred.loc[:, :] = X_scaler.transform(X_pred)
    
    test_size = 0.2
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size)
        
    test_size = 0.2
    X_train, X_cross, y_train, y_cross = train_test_split(X_train, y_train,
                                                          test_size=test_size)
        
    # Create a based model
    rf = RandomForestClassifier()# Instantiate the grid search model
    grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                              cv = 5, n_jobs = -1)
    
    # Fit the grid search to the data
    grid_search.fit(X_train.values, y_train.values.ravel())
    grid_search.best_params_
    
    best_grid = grid_search.best_estimator_
    training_accuracy = evaluate(best_grid, X_train.values, y_train.values.ravel())
    validation_accuracy = evaluate(best_grid, X_cross.values, y_cross.values.ravel())
    test_accuracy = evaluate(best_grid, X_test.values, y_test.values.ravel())
    
    print(training_accuracy, validation_accuracy, test_accuracy)
    
    
    y_pred = best_grid.predict(X_pred)
    
    Submission = pd.DataFrame(np.vstack((test_data.PassengerId.values, y_pred)).T,
                              columns=['PassengerId', 'Survived'])
    Submission.to_csv('submission.csv', index=False)
    
