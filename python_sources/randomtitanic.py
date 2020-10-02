import pandas as pd
import numpy as np
from sklearn import ensemble as en

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

y = train['Survived']
train = train.drop('Survived', 1)
train = train.drop('Name', 1)
train = train.drop('PassengerId', 1)
train = train.drop('Ticket', 1)
train = train.drop('Cabin', 1)
train['Embarked'] = train['Embarked'].map({'C':2, 'Q':1, 'S':0}) 
train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mean())
train['Sex'] = train['Sex'].map({'female':1, 'male':0}) 
train['Age'] = train['Age'].fillna(train['Age'].mean())
X = train

clf = en.RandomForestClassifier()
clf.fit(X, y)
c = test.PassengerId
test = test.drop('Name', 1)
test = test.drop('PassengerId', 1)
test = test.drop('Ticket', 1)
test = test.drop('Cabin', 1)
test['Embarked'] = test['Embarked'].map({'C':2, 'Q':1, 'S':0}) 
test['Embarked'] = test['Embarked'].fillna(test['Embarked'].mean())
test['Sex'] = test['Sex'].map({'female':1, 'male':0}) 
test['Age'] = test['Age'].fillna(test['Age'].mean())
test['Fare'] = test['Fare'].fillna(test['Fare'].mean())

Xt = test
t = clf.predict(Xt)

output = pd.DataFrame([])
output['Survived'] = t
output['PassengerId'] = c

output.to_csv('pred.csv', header = ['Survived', 'PassengerId'], index = False) 
	