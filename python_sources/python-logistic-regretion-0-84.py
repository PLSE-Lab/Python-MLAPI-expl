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

from sklearn.linear_model import LogisticRegression

train = pd.read_csv("../input/train.csv")
teste = pd.read_csv("../input/test.csv")


train = train.drop(['PassengerId','Ticket'], axis=1)
teste = teste.drop(['PassengerId','Ticket'], axis=1)


# Embarked
embark_dummies_titanic  = pd.get_dummies(train['Embarked'])
train = train.join(embark_dummies_titanic)
train.drop(['Embarked'], axis=1,inplace=True)

embark_dummies_titanic  = pd.get_dummies(teste['Embarked'])
teste = teste.join(embark_dummies_titanic)
teste.drop(['Embarked'], axis=1,inplace=True)

train['S'][(train['S']!=1) & (train['C']!=1) & (train['Q']!=1)]=train['S'][train['S']==1].count()/train['S'].count()
train['Q'][(train['S']!=1) & (train['C']!=1) & (train['Q']!=1)]=train['Q'][train['Q']==1].count()/train['Q'].count()
train['C'][(train['S']!=1) & (train['C']!=1) & (train['Q']!=1)]=train['C'][train['C']==1].count()/train['C'].count()

teste['S'][(teste['S']!=1) & (teste['C']!=1) & (teste['Q']!=1)]=teste['S'][teste['S']==1].count()/teste['S'].count()
teste['Q'][(teste['S']!=1) & (teste['C']!=1) & (teste['Q']!=1)]=teste['Q'][teste['Q']==1].count()/teste['Q'].count()
teste['C'][(teste['S']!=1) & (teste['C']!=1) & (teste['Q']!=1)]=teste['C'][teste['C']==1].count()/teste['C'].count()


# Sex
person_dummies_titanic  = pd.get_dummies(train['Sex'])
train = train.join(person_dummies_titanic)
train.drop(['Sex'], axis=1,inplace=True)
train.drop(['male'], axis=1,inplace=True)

person_dummies_titanic  = pd.get_dummies(teste['Sex'])
teste = teste.join(person_dummies_titanic)
teste.drop(['Sex'], axis=1,inplace=True)
teste.drop(['male'], axis=1,inplace=True)


# Age
train["Age"] = train["Age"].fillna(train['Age'].mean())
teste["Age"] = train["Age"].fillna(train['Age'].mean())

train["Age2"] = train["Age"]*train["Age"]/1
teste["Age2"] = teste["Age"]*teste["Age"]/1

# Fare
train["Fare"] = train["Fare"].fillna(train['Fare'].mean())
teste["Fare"] = train["Fare"].fillna(train['Fare'].mean())

#train["Fare2"] = train["Fare"]*train["Fare"]/100
#teste["Fare2"] = teste["Fare"]*teste["Fare"]/100


# Class
class_dummies_titanic  = pd.get_dummies(train['Pclass'])
train = train.join(class_dummies_titanic)
train.drop(['Pclass'], axis=1,inplace=True)

class_dummies_titanic  = pd.get_dummies(teste['Pclass'])
teste = teste.join(class_dummies_titanic)
teste.drop(['Pclass'], axis=1,inplace=True)


# Family
train["Family"] = train["SibSp"]+train["Parch"]+1
train.drop(['Parch'], axis=1,inplace=True)

teste["Family"] = teste["SibSp"]+teste["Parch"]+1
teste.drop(['Parch'], axis=1,inplace=True)

train["Family2"] = train["Family"]*train["Family"]/1
teste["Family2"] = teste["Family"]*teste["Family"]/1


# Cabin
train['Cabin'].loc[~train['Cabin'].isnull()] = 1
train['Cabin'].loc[train['Cabin'].isnull()] = 0
teste['Cabin'].loc[~teste['Cabin'].isnull()] = 1
teste['Cabin'].loc[teste['Cabin'].isnull()] = 0


# Title
def name_extract(data):
    # extract Title from name
    data['Title'] = data.Name.str.extract('([A-Za-z]+)\.', expand=False)
    # delete rare title
    data['Title'] = data['Title'].replace(['Capt', 'Col','Lady', 'Countess',
        'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    data['Title'] = data['Title'].replace(['Mlle', 'Ms'], 'Miss')
    data['Title'] = data['Title'].replace('Mme', 'Mrs')
    data['Surname'] = data['Name'].apply(lambda x: str(x).split('.')[1].split(' ')[1])
    data['Surname'] = data.Surname.str.replace('(', '')
                                            
    return data.drop(['Name'], axis=1) 

train = name_extract(train)
teste = name_extract(teste)

title_dummies_titanic  = pd.get_dummies(train['Title'])
train = train.join(title_dummies_titanic)

title_dummies_titanic  = pd.get_dummies(teste['Title'])
teste = teste.join(title_dummies_titanic)


# Drop
train.drop(['Title'], axis=1,inplace=True)
train.drop(['Surname'], axis=1,inplace=True)



teste.drop(['Title'], axis=1,inplace=True)
teste.drop(['Surname'], axis=1,inplace=True)



# define training and testing sets

X_train = train.drop("Survived",axis=1)
Y_train = train["Survived"]
X_test  =teste

# Logistic Regression

logreg = LogisticRegression()

results=logreg.fit(X_train, Y_train)


Y_pred = logreg.predict(X_train)
Y_test = logreg.predict(X_test)


r=logreg.score(X_train, Y_train)

print(r)

test2 = pd.read_csv("../input/test.csv")

submission = pd.DataFrame({
        "PassengerId": test2["PassengerId"],
        "Survived": Y_test
    })
#submission.to_csv('titanic9.csv', index=False)