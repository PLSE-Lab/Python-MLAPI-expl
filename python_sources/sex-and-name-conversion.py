import numpy as np
import pandas as pd
from sklearn import cross_validation, tree, linear_model, svm, metrics
from sklearn.ensemble import GradientBoostingClassifier
import time

# Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, header=0)
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, header=0)

# Pclass - imputacao dos valores ausentes
mode_pclass = train['Pclass'].dropna().mode().values

train['Pclass'].fillna(mode_pclass[0], inplace=True)
test['Pclass'].fillna(mode_pclass[0], inplace=True)

# Name - imputacao dos valores ausentes

train['Name'].fillna('x, others.', inplace=True)
test['Name'].fillna('x, others.', inplace=True)

# Name - convert


def title_type(t):
    if t == 'Mr':
        return 0
    elif t == 'Capt':
        return 1
    elif t == 'Rev':
        return 2
    elif t == 'Master':
        return 3
    elif t == 'Mrs':
        return 4
    elif t == 'Mlle':
        return 5
    elif t == 'Miss':
        return 6
    elif t == 'Dr':
        return 7
    else:
        return 8


title = []
for name in train['Name']:
    t1 = name.split(', ')
    t2 = t1[1].split('.')
    title.append(t2[0])

title_int = []
for name in title:
    title_int.append(title_type(name))

train['Title'] = title_int

title = []
for name in test['Name']:
    t1 = name.split(', ')
    t2 = t1[1].split('.')
    title.append(t2[0])

title_int = []
for name in title:
    title_int.append(title_type(name))

test['Title'] = title_int


# Sex - imputacao dos valores ausentes

mode_sex = train['Sex'].dropna().mode().values

age_nan = train['Sex'][train['Sex'].isnull()]
for x in age_nan:
    if train['Title'][x[0]] in ['Mr', 'Capt', 'Master', 'Rev']:
        train.loc[x[0], 'Sex'] = 'male'
    elif train['Title'][x[0]] in ['Mrs', 'Mlle', 'Miss']:
        train.loc[x[0], 'Sex'] = 'female'
    else:
        train.loc[x[0], 'Sex'] = mode_sex

age_nan = test['Sex'][test['Sex'].isnull()]
for x in age_nan:
    if test['Title'][x[0]] in ['Mr', 'Capt', 'Master', 'Rev']:
        test.loc[x[0], 'Sex'] = 'male'
    elif test['Title'][x[0]] in ['Mrs', 'Mlle', 'Miss']:
        test.loc[x[0], 'Sex'] = 'female'
    else:
        test.loc[x[0], 'Sex'] = mode_sex


# Age - imputacao dos valores ausentes
mode_age = train['Age'].dropna().mode().values

train['Age'].fillna(mode_age[0], inplace=True)
test['Age'].fillna(mode_age[0], inplace=True)

# SibSp - imputacao dos valores ausentes
mode_sibSp = train['SibSp'].dropna().mode().values

train['SibSp'].fillna(mode_sibSp[0], inplace=True)
test['SibSp'].fillna(mode_sibSp[0], inplace=True)

# Parch - imputacao dos valores ausentes
mode_parch = train['Parch'].dropna().mode().values

train['Parch'].fillna(mode_parch[0], inplace=True)
test['Parch'].fillna(mode_parch[0], inplace=True)

# Embarked - imputacao dos valores ausentes
mode_embarked = train['Embarked'].dropna().mode().values

train['Embarked'].fillna(mode_embarked[0], inplace=True)
test['Embarked'].fillna(mode_embarked[0], inplace=True)

# Sex , Embarked - convert to numeric values
dummies = []
cols = ['Sex', 'Embarked']
for col in cols:
    dummies.append(pd.get_dummies(train[col]))
titanic_dummies = pd.concat(dummies, axis=1)
train = pd.concat((train, titanic_dummies), axis=1)
train = train.drop(['Sex', 'Embarked'], axis=1)

dummies = []
cols = ['Sex', 'Embarked']
for col in cols:
    dummies.append(pd.get_dummies(test[col]))
titanic_dummies = pd.concat(dummies, axis=1)
test = pd.concat((test, titanic_dummies), axis=1)
test = test.drop(['Sex', 'Embarked'], axis=1)

# removendo atributos:
train = train.drop(['Ticket', 'Cabin', 'Fare', 'Name'], axis=1)
test = test.drop(['Ticket', 'Cabin', 'Fare', 'Name'], axis=1)



##Submeter

x = train.values
answer = train['Survived'].values

x = np.delete(x, 1, axis=1)

clf = tree.DecisionTreeClassifier(criterion='gini')
clf.fit(x, answer)
test_result = clf.predict(test)

output = np.column_stack((test['PassengerId'],test_result))

df_results = pd.DataFrame(output.astype('int'),columns=['PassengerId','Survived'])
df_results.to_csv('titanic_results.csv',index=False)
