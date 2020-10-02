import numpy as np
import pandas as pd
import csv
from sklearn.ensemble import RandomForestClassifier

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

train['Gender'] = train['Sex'].map( {'female':0,'male':1} ).astype(int)
if len(train.Embarked[ train.Embarked.isnull() ]) > 0:
    train.loc[ (train.Embarked.isnull()), 'Embarked' ] = train.Embarked.dropna().mode().values
Ports = list(enumerate(np.unique(train.Embarked)))
Port_dict = {name: i for i,name in Ports}
train.Embarked = train.Embarked.map(lambda x: Port_dict[x]).astype(int)

print(train.head(10))

median_age = train.Age.dropna().median()
if len(train.Age[ train.Age.isnull() ]) > 0:
    train.loc[ (train.Age.isnull()), 'Age' ] = median_age

print(train.head(10))

test['Gender'] = test['Sex'].map( {'female':0,'male':1} ).astype(int)
if len(test.Embarked[ test.Embarked.isnull() ]) > 0:
    test.loc[ (test.Embarked.isnull()), 'Embarked' ] = test.Embarked.dropna().mode().values
test.Embarked = test.Embarked.map(lambda x: Port_dict[x]).astype(int)
median_age = test.Age.dropna().median()
if len(test.Age[ test.Age.isnull() ]) > 0:
    test.loc[ (test.Age.isnull()), 'Age' ] = median_age
if len(test.Fare[ test.Fare.isnull() ]) > 0:
    median_fare = np.zeros(3)
    for i in range(3):
        median_fare[i] = test.Fare[ test.Pclass == i + 1 ].median()
    for i in range(3):
        test.loc[ (test.Fare.isnull()) & (test.Pclass == i + 1), 'Fare'] = median_fare[i]
print(test.head(10))