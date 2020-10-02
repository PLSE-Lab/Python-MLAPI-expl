# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import sklearn.svm as svm

def convertSex(s: str):
    if(s == 'female'):
        return 1
    return 0

def convertEmbarked(s: str):
    if(s=='S'):
        return 1
    if(s=='C'):
        return 2
    return 3

def filterData(train: object):
    train['Sex_n'] = train['Sex'].apply(convertSex)
    train['Embarked_n'] = train['Embarked'].apply(convertEmbarked)
    #train['Family_size'] = train['SibSp'] + train['Parch']
    ageMean = np.round(train['Age'].mean())
    train['Age'].fillna(value=ageMean, inplace=True)
    return train.drop(['PassengerId','Name','Ticket','Fare','Cabin', 'Sex', 'Embarked'], axis=1)

#Training
train = pd.read_csv("../input/train.csv")
train2 = filterData(train)
y = train2['Survived'].values
X = train2.drop('Survived', axis=1)
clf = svm.SVC()
clf.fit(X, y)
#Predict
test = pd.read_csv("../input/test.csv")
test2 = filterData(test)
result = clf.predict(test2.values)
#Write output
output = pd.DataFrame({'PassengerId': test['PassengerId'].values, 'Survived':result})
output.to_csv("result.csv", index=False)
