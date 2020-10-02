import math
import pandas as pd
import numpy as np
import csv as csv
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.ensemble import AdaBoostClassifier as ada

train = pd.read_csv('../input/train.csv', header=0)
test = pd.read_csv('../input/test.csv', header=0)



Title_Dictionary = {

    "Dr": "2",
    "Capt" : "2",
    "Major" : "2",
    "Rev" : "2",
    "Col" : "2",
    "Jonkheer" : "3",
    "Don" : "3",
    "Sir" : "3",
    "Lady" : "3",
    "Master" : "3",
    "the Countess" : "3",
    "Dona" : "3",
    "Mme" : "1",
    "Mlle" : "1",
    "Ms" : "1",
    "Mr" : "0",
    "Mrs" : "1",
    "Miss" : "1"

}

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

# (Fare) filling empty values with median of Pclass and doing binning
bin_size = 20
bin_max = 100
bins = [20, 40, 60, 80, 100]
median_fare = np.zeros(3)
for f in range(0,3):
    median_fare[f] = train[ train.Pclass == f+1 ]['Fare'].dropna().median()
    if median_fare[f] > bin_max:
        median_fare[f] = bin_max

if len(train.Fare[ train.Fare.isnull() ]) > 0:
    for f in range(0,3):
        train.loc[ (train.Fare.isnull()) & (train.Pclass == f+1 ), 'Fare'] = median_fare[f]
train.loc[ train.Fare > bin_max, "Fare" ] = bin_max

train.Fare = train.Fare.map( lambda x: bins[math.ceil(x / bin_size) - 1] )


# encoding in dummy variable

train['Title'] = train['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())

train['Title'] = train.Title.map(Title_Dictionary)

titles_dummies = pd.get_dummies(train['Title'], prefix='Title')
train = pd.concat([train, titles_dummies], axis=1)

# removing the title variable
train.drop('Title', axis=1, inplace=True)

# dropping unnecessary attributes
train = train.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1)

features = list(train.columns)
train[features] = train[features].apply(lambda x: x/x.max(), axis=0)

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

# (Fare) filling empty values with median of Pclass and doing binning
bin_size = 20
bin_max = 100
bins = [20, 40, 60, 80, 100]
median_fare = np.zeros(3)
for f in range(0,3):
    median_fare[f] = test[ test.Pclass == f+1 ]['Fare'].dropna().median()
    if median_fare[f] > bin_max:
        median_fare[f] = bin_max

if len(test.Fare[ test.Fare.isnull() ]) > 0:
    for f in range(0,3):
        test.loc[ (test.Fare.isnull()) & (test.Pclass == f+1 ), 'Fare'] = median_fare[f]
test.loc[ test.Fare > bin_max, "Fare" ] = bin_max

test.Fare = test.Fare.map( lambda x: bins[math.ceil(x / bin_size) - 1] )

# encoding in dummy variable

test['Title'] = test['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())

test['Title'] = test.Title.map(Title_Dictionary)

titles_dummies = pd.get_dummies(test['Title'], prefix='Title')
test = pd.concat([test, titles_dummies], axis=1)

ids = test['PassengerId'].values

# removing the title variable
test.drop('Title', axis=1, inplace=True)

test = test.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1)

features = list(test.columns)
test[features] = test[features].apply(lambda x: x/x.max(), axis=0)

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
