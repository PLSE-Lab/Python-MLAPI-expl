# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


import pandas as pd
import numpy as np

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


survived = train[train['Survived'] == 1]
not_survived = train[train['Survived'] == 0]


train__test__data = [train, test] 

for dataset in train__test__data:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.')
    



for dataset in train__test__data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', \
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    



title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Other": 5}
for dataset in train__test__data:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
     



for dataset in train__test__data:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)


for dataset in train__test__data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')


for dataset in train__test__data:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    

for dataset in train__test__data:
    dataset['Age'] =dataset['Age'].fillna(dataset['Age'].mean())
    dataset['Age'] = dataset['Age'].astype(int)
    
train['AgeBand'] = pd.cut(train['Age'], 5)



for dataset in train__test__data:
    dataset.loc[ dataset['Age'] <= 15, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 15) & (dataset['Age'] <= 30), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 30) & (dataset['Age'] <= 45), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 45) & (dataset['Age'] <= 60), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 60, 'Age'] = 4



for dataset in train__test__data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].mean())


train['FareBand'] = pd.qcut(train['Fare'], 4)


for dataset in train__test__data:
    dataset.loc[ dataset['Fare'] <= 8, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 8) & (dataset['Fare'] <= 15), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 15) & (dataset['Fare'] <= 30), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 30, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

for dataset in train__test__data:
    dataset['FamilySize'] = dataset['SibSp'] +  dataset['Parch'] + 1


for dataset in train__test__data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
    


features_drop = ['Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'FamilySize']
train = train.drop(features_drop, axis=1)
test = test.drop(features_drop, axis=1)
train = train.drop(['PassengerId', 'AgeBand', 'FareBand'], axis=1)



X_train = train.drop('Survived', axis=1)
y_train = train['Survived']
X_test = test.drop("PassengerId", axis=1).copy()



from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


clf =DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_predicted = clf.predict(X_test)

submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": y_predicted
    })

submission.to_csv('mohamedehab.csv', index=False)