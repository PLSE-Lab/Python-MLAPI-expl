"""
This code is mostly based from this tutorial:

https://www.dataquest.io/mission/74/getting-started-with-kaggle

"""

import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold


def scale(x, mu, sigma):
    return (x - mu) / sigma


train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

# Clean data
# 1. Add missing values with the median, or most popular value.
train['Age'] = train['Age'].fillna(train['Age'].median())
train['Embarked'] = train['Embarked'].fillna(train['Embarked'].describe().top)

test['Age'] = train['Age'].fillna(train['Age'].median())
test['Embarked'] = test['Embarked'].fillna(train['Embarked'].describe().top)

# 2. 'Digitize' non-numeric values (e.g. sex, where they embarked, etc.).
train.loc[train['Sex'] == 'male', 'Sex'] = 0
train.loc[train['Sex'] == 'female', 'Sex'] = 1
train.loc[train['Embarked'] == 'S', 'Embarked'] = 0
train.loc[train['Embarked'] == 'C', 'Embarked'] = 1
train.loc[train['Embarked'] == 'Q', 'Embarked'] = 2

test.loc[test['Sex'] == 'male', 'Sex'] = 0
test.loc[test['Sex'] == 'female', 'Sex'] = 1
test.loc[test['Embarked'] == 'S', 'Embarked'] = 0
test.loc[test['Embarked'] == 'C', 'Embarked'] = 1
test.loc[test['Embarked'] == 'Q', 'Embarked'] = 2
test['Fare'] = test['Fare'].fillna(train['Fare'].median())

# Improve features
# 1. Generate new features.
train['FamilySize'] = train['SibSp'] + train['Parch']
test['FamilySize'] = test['SibSp'] + test['Parch']

# 2. Scale features.
train['AgeScaled'] = train['Age'].apply(scale, args=(train['Age'].mean(), train['Age'].std()))
train['FareScaled'] = train['Fare'].apply(scale, args=(train['Fare'].mean(), train['Fare'].std()))
test['AgeScaled'] = test['Age'].apply(scale, args=(test['Age'].mean(), test['Age'].std()))
test['FareScaled'] = test['Fare'].apply(scale, args=(test['Fare'].mean(), test['Fare'].std()))

# Setup learning
predictors = ['AgeScaled', 'Pclass', 'Sex', 'FareScaled', 'Embarked', 'FamilySize']
clf = LinearSVC()
kf = KFold(n_splits=3, random_state=1)

# Learn
predictions = []
for xx, yy in kf.split(train):
    train_predictors = (train[predictors].iloc[xx, :])
    train_target = train['Survived'].iloc[xx]
    clf.fit(train_predictors, train_target)
    test_predictions = clf.predict(train[predictors].iloc[yy, :])
    predictions.append(test_predictions)

# Accuracy
predictions = np.concatenate(predictions, axis=0)
total = predictions == train['Survived']
accuracy = len(total[total == True]) / float(len(predictions))
print("training accuracy: {}".format(accuracy))

# Precit on test data
clf_test = LinearSVC()
clf_test.fit(train[predictors], train['Survived'])
pred = clf_test.predict(test[predictors])

# Submit
submission = pd.DataFrame({'PassengerID': test['PassengerId'], 'Survived': pred})
submission.to_csv("kaggle.csv", index=False)
