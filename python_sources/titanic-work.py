# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

from sklearn.ensemble import RandomForestRegressor

# Read the data
train = pd.read_csv('../input/train.csv')

train = train.drop(["Cabin","Name","Ticket"], axis=1)
train=pd.get_dummies(train)

from sklearn.preprocessing import Imputer
my_imputer = Imputer()
train = my_imputer.fit_transform(train)
train = pd.DataFrame(data=train, columns=['PassengerId', 'Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S'])


# pull data into target (y) and predictors (X)
train_y = train.Survived
predictor_cols = ['Parch', 'Fare', 'Sex_female', 'Embarked_C', 'Embarked_Q']

# Create training predictors data
train_X = train[predictor_cols]

my_model = RandomForestRegressor()
my_model.fit(train_X, train_y)
# Read the test data
test = pd.read_csv('../input/test.csv')
test = test.drop(["Cabin","Name","Ticket"], axis=1)
test=pd.get_dummies(test)

test = my_imputer.fit_transform(test)
test = pd.DataFrame(data=test, columns=['PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S'])

test.head()

# Treat the test data in the same way as training data. In this case, pull same columns.
test_X = test[predictor_cols]
# Use the model to make predictions
predicted_survived = my_model.predict(test_X)
predicted_survived = [0 if p < 0.6 else 1 for p in predicted_survived]
# We will look at the predicted prices to ensure we have something sensible.

test["PassengerId"] = test["PassengerId"].astype("int")

#Submit
my_submission = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predicted_survived})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)