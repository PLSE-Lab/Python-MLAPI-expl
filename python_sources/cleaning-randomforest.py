import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Load
train = pd.read_csv('../input/train.csv')
Y_train = train['Survived'].astype('int32')
X_train = train.drop(['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
test = pd.read_csv('../input/test.csv')
X_test = test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# Embarked, only 2 rows de drop in the training set
X_train.drop(X_train['Embarked'].isnull())

# Set Fare to 0 for only one Test entry
X_test['Fare'][X_test['Fare'].isnull()] = 0

# Age, we replace Nan ages by a random number between mean and standard deviation
def fill_age(data):
	age_avg = data['Age'].mean()
	age_std = data['Age'].std()
	age_nan_count = data['Age'].isnull().sum()
	data['Age'][data['Age'].isnull()] = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_nan_count)

fill_age(X_train)
fill_age(X_test)

# Categorize variables
def categorize(data):
	data = pd.concat([data, pd.get_dummies(data['Pclass'], prefix='Pclass')], axis=1)
	data = data.drop('Pclass', axis=1)
	data = pd.get_dummies(data)
	return data

X_train = categorize(X_train)
X_test = categorize(X_test)

# Normalize
def normalize(data):
	fields_to_normalize = ['Age', 'SibSp', 'Parch', 'Fare']
	normalized = data[fields_to_normalize].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
	data = data.drop(fields_to_normalize, axis=1)
	data = pd.concat([data, normalized], axis=1)
	data = data.astype('float32')
	return data

X_train = normalize(X_train)
X_test = normalize(X_test)


## RandomForest
rf = RandomForestClassifier()
rf.fit( X_train, Y_train)

## Prediction
preds = rf.predict(X_test)
pd.DataFrame({
	'PassengerId': test['PassengerId'], 
	'Survived': preds
}).to_csv('gender_submission.csv', index=False, header=True)
