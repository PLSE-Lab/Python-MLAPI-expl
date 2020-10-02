# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
print (test.head())
del test['Cabin']
test.fillna(test['Age'].median(), inplace=True)

train = train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Survived']]
train.fillna(train['Age'].median(), inplace=True)
train = train.dropna()
#print(test.isnull().sum())
y = train['Survived']
train = train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]


X_train, X_valid, y_train, y_valid = train_test_split(train, y, test_size=0.2)


X_train.Sex = pd.Categorical(X_train.Sex)
X_train.Embarked = pd.Categorical(X_train.Embarked)

X_train['S_Code'] = X_train.Sex.cat.codes
X_train['E_Code'] = X_train.Embarked.cat.codes

del X_train['Sex']
del X_train['Embarked']


X_valid.Sex = pd.Categorical(X_valid.Sex)
X_valid.Embarked = pd.Categorical(X_valid.Embarked)

X_valid['S_Code'] = X_valid.Sex.cat.codes
X_valid['E_Code'] = X_valid.Embarked.cat.codes

del X_valid['Sex']
del X_valid['Embarked']

print (X_train.head(10), X_train.shape, y_train.shape)

clf = AdaBoostClassifier(n_estimators=100).fit(X_train, y_train)
print (clf.score(X_valid, y_valid))


test.Sex = pd.Categorical(test.Sex)
test.Embarked = pd.Categorical(test.Embarked)

test['S_Code'] = test.Sex.cat.codes
test['E_Code'] = test.Embarked.cat.codes

del test['Sex']
del test['Embarked']
pred = clf.predict(test[['Pclass', 'S_Code', 'Age', 'SibSp', 'Parch', 'Fare', 'E_Code']])
df_p = pd.DataFrame([test['PassengerId'], pred]).transpose()

df_p.to_csv('passenger.csv', header=['PassengerId', 'Survived'], index=False)