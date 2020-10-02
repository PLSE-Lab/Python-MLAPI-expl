import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load training data
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')

# Replace missing ages and fares with median in both training and test data
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())
test_data['Age'] = test_data['Age'].fillna(test_data['Age'].median())
test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].median())
# Replace 'male' with 1 and 'female' with 2 in 'Sex' column
train_data['Sex'] = train_data['Sex'].map({'male': 1, 'female': 2})
test_data['Sex'] = test_data['Sex'].map({'male': 1, 'female': 2})

# Organize data
predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Fare']
target = ['Survived']
predictors_columns = train_data[predictors]
target_columns = train_data[target]
train_X, val_X, train_y, val_y = train_test_split(predictors_columns, target_columns)
test_X = test_data[predictors]

# Lowest mean absolute error is 0.188 for number of leaf nodes = 50
titanic_dt = DecisionTreeRegressor(max_leaf_nodes=50)
titanic_dt.fit(train_X, train_y)
predictions_dt = titanic_dt.predict(val_X)
print(mean_absolute_error(val_y, predictions_dt))

# Mean absolute error is between 0.139 and 0.175
titanic_rf = RandomForestClassifier()
titanic_rf.fit(train_X, train_y.values.ravel())
predictions_rf = titanic_rf.predict(val_X)
print(mean_absolute_error(val_y, predictions_rf))

# We will use the random forest model
predictions_test = titanic_rf.predict(test_X)
my_submission = pd.DataFrame({'PassengerId': test_data.PassengerId,
                              'Survived': predictions_test})
my_submission.to_csv('submission.csv', index=False)                       
