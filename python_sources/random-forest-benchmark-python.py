import pandas as pd
import numpy as np
import random
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv('../input/train.csv')


def trans_sex(sex):
    if sex == 'male':
        return 0
    if sex == 'female':
        return 1


def fillna_age(age):
    if np.isnan(age):
        age_mean = 29.6991176471
        age_std = 14.5264973323
        return random.gauss(age_mean, age_std)
    else:
        return age

#print(train.Fare.mean())
#print(train.Fare.std())


def fillna_fare(fare):
    if np.isnan(fare):
        fare_mean = 32.2042079686
        fare_std = 49.6934285972
        return random.gauss(fare_mean, fare_std)
    else:
        return fare

def trans_age(age):
    if np.isnan(age):
        age_mean = 29.6991176471
        age_std = 14.5264973323
        return random.gauss(age_mean, age_std)
    else:
        if age >= 80:return 8
        if age >= 70:return 7
        if age >= 60:return 6
        if age >= 50:return 5
        if age >= 40:return 4
        if age >= 30:return 3
        if age >= 20:return 2
        if age >= 10:return 1
        if age >= 7:return 0
        if age >= 4:return -1
        if age >= 1:return -2


def trans_embarked(embarked):
    if embarked == 'C':
        return 1
    elif embarked == 'Q':
        return 2
    elif embarked == 'S':
        return 3
    else:
        return 0


train.Sex = train.Sex.apply(trans_sex)
train.Age = train.Age.apply(fillna_age)
#train.Age = train.Age.apply(trans_age)
train.Embarked = train.Embarked.apply(trans_embarked)

train['Relative'] = train['SibSp'] + train['Parch']
train_label = train.Survived


del train['Survived']
del train['PassengerId']
del train['Name']
del train['Ticket']
del train['Cabin']


clf = RandomForestClassifier(n_estimators=1000)
clf = clf.fit(train,train_label)

scores = cross_val_score(clf, train, train_label)
print(scores.mean())


test = pd.read_csv('../input/test.csv')
test.Sex = test.Sex.apply(trans_sex)
test.Age = test.Age.apply(fillna_age)
test.Fare = test.Fare.apply(fillna_fare)
#train.Age = train.Age.apply(trans_age)
test.Embarked = test.Embarked.apply(trans_embarked)

test['Relative'] = test['SibSp'] + test['Parch']

result = pd.DataFrame()
result['PassengerId'] = test['PassengerId']
del test['PassengerId']
del test['Name']
del test['Ticket']
del test['Cabin']
print(test.head(5))
test_output = clf.predict(test)
result['Survived'] = test_output
print(result)
result.to_csv('Result.csv',index=False)