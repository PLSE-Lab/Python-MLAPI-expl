import numpy as np
import pandas as pd
from sklearn import tree

# because we love numbers
gender_map = {'female': 0, 'male': 1}

# the list of features we will use for predictions
fields = ['Pclass', 'Sex', 'Age', 'Fare']

# import the training and test sets
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# cleanup the training set
train['Sex'] = train['Sex'].map(gender_map).astype(int)
train['Age'] = train['Age'].fillna(train['Age'].median())

# cleanup the test set
test['Sex'] = test['Sex'].map(gender_map).astype(int)
test['Age'] = test['Age'].fillna(test['Age'].median())
test['Fare'] = test['Fare'].fillna(test['Fare'].median())

# convert to arrays
target = train['Survived'].values
features = train[fields].values
test_features = test[fields].values

# build decision tree
dec_tree = tree.DecisionTreeClassifier(min_samples_split = 4)
dec_tree = dec_tree.fit(features, target)

# use model on test set
prediction = dec_tree.predict(test_features)

# use required format for predictions
pindex = test['PassengerId'].values
solution = pd.DataFrame(prediction, pindex, columns = ['Survived'])

# write predictions to file
solution.to_csv('solution.csv', index_label = 'PassengerId')