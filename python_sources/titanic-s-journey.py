import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

train.drop(['PassengerId','Name','Ticket'],axis=1)
test.drop(['Name','Ticket'],axis=1)

train.drop(['Embarked'], axis=1,inplace=True)
test.drop(['Embarked'], axis=1,inplace=True)

train["Fare"] = train["Fare"].fillna(train["Fare"].median())
test["Fare"] = test["Fare"].fillna(test["Fare"].median())

average_age_train = train['Age'].mean()
average_age_test = test['Age'].mean()
std_age_train = train['Age'].std()
std_age_test = test['Age'].std()
count_null_age_train = train['Age'].isnull().sum()
count_null_age_test = test['Age'].isnull().sum()

rand_train = np.random.randint(average_age_train-std_age_train,average_age_train+std_age_train,size=count_null_age_train)
rand_test = np.random.randint(average_age_test-std_age_test,average_age_test+std_age_test,size=count_null_age_test)

train['Age'][np.isnan(train['Age'])] = rand_train
test['Age'][np.isnan(test['Age'])] = rand_test

train.drop("Cabin",axis=1,inplace=True)
test.drop("Cabin",axis=1,inplace=True)

train['Family'] =  train["Parch"] + train["SibSp"]
train['Family'].loc[train['Family'] > 0] = 1
train['Family'].loc[train['Family'] == 0] = 0

test['Family'] =  test["Parch"] + test["SibSp"]
test['Family'].loc[test['Family'] > 0] = 1
test['Family'].loc[test['Family'] == 0] = 0

train = train.drop(['SibSp','Parch'], axis=1)
test = test.drop(['SibSp','Parch'], axis=1)

def type_person(person):
    age,sex = person
    return 'child' if age < 16 else sex
    
train['Person'] = train[['Age','Sex']].apply(type_person,axis=1)
test['Person'] = test[['Age','Sex']].apply(type_person,axis=1)

train.drop(['Sex'],axis=1,inplace=True)
test.drop(['Sex'],axis=1,inplace=True)

person_dummies_train  = pd.get_dummies(train['Person'])
person_dummies_train.columns = ['Male','Female','Child']
person_dummies_train.drop(['Male'], axis=1, inplace=True)

person_dummies_test  = pd.get_dummies(test['Person'])
person_dummies_test.columns = ['Male','Female','Child']
person_dummies_test.drop(['Male'], axis=1, inplace=True)

train = train.join(person_dummies_train)
test = test.join(person_dummies_test)

train.drop(['Person'],axis=1,inplace=True)
test.drop(['Person'],axis=1,inplace=True)

pclass_dummies_train = pd.get_dummies(train['Pclass'])
pclass_dummies_train.columns = ['Class_1','Class_2','Class_3']
pclass_dummies_train.drop(['Class_3'], axis=1, inplace=True)

pclass_dummies_test  = pd.get_dummies(test['Pclass'])
pclass_dummies_test.columns = ['Class_1','Class_2','Class_3']
pclass_dummies_test.drop(['Class_3'], axis=1, inplace=True)

train.drop(['Pclass'],axis=1,inplace=True)
test.drop(['Pclass'],axis=1,inplace=True)

train = train.join(pclass_dummies_train)
test = test.join(pclass_dummies_test)

X_train = train.drop(["Name","Ticket","PassengerId","Survived"],axis=1)
Y_train = train["Survived"]
X_test  = test.drop(["PassengerId","Name","Ticket"],axis=1).copy()

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)

submission = pd.DataFrame({"PassengerId": test["PassengerId"],"Survived": Y_pred})
submission.to_csv('titanic.csv', index=False)
