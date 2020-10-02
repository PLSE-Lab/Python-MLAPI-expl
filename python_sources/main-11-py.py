import pandas as pd
import numpy as np
import csv as csv
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.ensemble import AdaBoostClassifier as ada

train = pd.read_csv('../input/train.csv', header=0)
test = pd.read_csv('../input/test.csv', header=0)

################################# Training Set #################################
train['Gender'] = train['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

if len(train.Embarked[ train.Embarked.isnull() ]) > 0:
    train.Embarked[ train.Embarked.isnull() ] = train.Embarked.dropna().mode().values

Ports = list(enumerate(np.unique(train['Embarked'])))
Ports_dict = { name : i for i, name in Ports }
train.Embarked = train.Embarked.map( lambda x: Ports_dict[x]).astype(int)

median_age = train['Age'].dropna().median()
if len(train.Age[ train.Age.isnull() ]) > 0:
    train.loc[ (train.Age.isnull()), 'Age'] = median_age

train = train.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1)

################################# Testing Set #################################
test['Gender'] = test['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

if len(test.Embarked[ test.Embarked.isnull() ]) > 0:
    test.Embarked[ test.Embarked.isnull() ] = test.Embarked.dropna().mode().values

test.Embarked = test.Embarked.map( lambda x: Ports_dict[x]).astype(int)

median_age = test['Age'].dropna().median()
if len(test.Age[ test.Age.isnull() ]) > 0:
    test.loc[ (test.Age.isnull()), 'Age'] = median_age

if len(test.Fare[ test.Fare.isnull() ]) > 0:
    median_fare = np.zeros(3)
    for f in range(0,3):
        median_fare[f] = test[ test.Pclass == f+1 ]['Fare'].dropna().median()
    for f in range(0,3):
        test.loc[ (test.Fare.isnull()) & (test.Pclass == f+1 ), 'Fare'] = median_fare[f]

ids = test['PassengerId'].values

test = test.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1)

################################# Classifier #################################
train_data = train.values
test_data = test.values

print('Training...')
forest = rfc(n_estimators = 1050, n_jobs = -1, criterion = "entropy", random_state = 777, max_features = None)
ada_forest = ada(forest, n_estimators = 4, random_state = 93)
ada_forest = forest.fit(train_data[0::, 1::], train_data[0::, 0])

print(ada_forest.score(train_data[0::, 1::], train_data[0::, 0]))

print('Predicting...')
output = ada_forest.predict(test_data).astype(int)

predictions_file = open("predictions.csv", "w")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print('Done!!!')
