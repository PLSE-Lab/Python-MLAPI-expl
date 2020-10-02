import math
import pandas as pd
import numpy as np
import csv as csv
from sklearn.ensemble import RandomForestClassifier as rfc

train = pd.read_csv('../input/train.csv', header=0)
test = pd.read_csv('../input/test.csv', header=0)

################################# Training Set #################################
# (Sex) set female to 0 and male to 1
train['Gender'] = train['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# (Embarked) fill empty values with mode
if len(train.Embarked[ train.Embarked.isnull() ]) > 0:
    train.Embarked[ train.Embarked.isnull() ] = train.Embarked.dropna().mode().values

# (Embarked) convert to numeric values
Ports = list(enumerate(np.unique(train['Embarked'])))
Ports_dict = { name : i for i, name in Ports }
train.Embarked = train.Embarked.map( lambda x: Ports_dict[x]).astype(int)

# (Age) fill missing values
if len(train.Age[ train.Age.isnull() ]) > 0:
    median_age = train['Age'].dropna().median()
    train.loc[ (train.Age.isnull()), 'Age'] = median_age
    age_b_size = 10
    aux = 0
    age_medians = np.zeros(10)
    for a in range(1, 11):
        age_medians[a - 1] = train.Age[ (train.Age > aux) & (train.Age <= age_b_size * a) ].median()
        aux += age_b_size
    aux = 0
    train.Age = train.Age.map( lambda x: age_medians[math.ceil(x / age_b_size) - 1] ).astype(int)

# dropping unnecessary attributes
train = train.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1)

################################# Testing Set #################################
# (Sex) set female to 0 and male to 1
test['Gender'] = test['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# (Embarked) fill empty values with mode
if len(test.Embarked[ test.Embarked.isnull() ]) > 0:
    test.Embarked[ test.Embarked.isnull() ] = test.Embarked.dropna().mode().values

# (Embarked) convert to numeric values
test.Embarked = test.Embarked.map( lambda x: Ports_dict[x]).astype(int)

# (Age) fill missing values
if len(test.Age[ test.Age.isnull() ]) > 0:
    median_age = test['Age'].dropna().median()
    test.loc[ (test.Age.isnull()), 'Age'] = median_age
    age_b_size = 10
    aux = 0
    age_medians = np.zeros(10)
    for a in range(1, 11):
        age_medians[a - 1] = test.Age[ (test.Age > aux) & (test.Age <= age_b_size * a) ].median()
        aux += age_b_size
    aux = 0
    test.Age = test.Age.map( lambda x: age_medians[math.ceil(x / age_b_size) - 1] ).astype(int)

# (Fare) filling empty values with median
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
forest = rfc(n_estimators = 100)
forest = forest.fit(train_data[0::, 1::], train_data[0::, 0])

print(forest.score(train_data[0::, 1::], train_data[0::, 0]))

print('Predicting...')
output = forest.predict(test_data).astype(int)

predictions_file = open("predictions.csv", "w")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print('Done!!!')
