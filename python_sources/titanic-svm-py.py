from __future__ import division
import pandas as pd
from sklearn import svm

import numpy as np




titanic = pd.read_csv("../input/train.csv")
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1

titanic["Embarked"] = titanic["Embarked"].fillna("S")
titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2
#print titanic["Cabin"].unique()

titanic_test = pd.read_csv("../input/test.csv")
titanic_test["Age"] = titanic_test["Age"].fillna(titanic_test["Age"].median())
titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic_test["Fare"].median())
titanic_test.loc[titanic_test["Sex"] == "male", "Sex"] = 0
titanic_test.loc[titanic_test["Sex"] == "female", "Sex"] = 1

titanic_test["Embarked"] = titanic_test["Embarked"].fillna("S")
titanic_test.loc[titanic_test["Embarked"] == "S", "Embarked"] = 0
titanic_test.loc[titanic_test["Embarked"] == "C", "Embarked"] = 1
titanic_test.loc[titanic_test["Embarked"] == "Q", "Embarked"] = 2

features = ["Pclass", "Sex", "Age", "SibSp", "Parch","Fare","Embarked"]


C = 0.7
clf = svm.SVC(kernel='linear',C=C).fit(titanic[features],titanic["Survived"])
prediction = clf.predict(titanic_test[features])

submission = pd.DataFrame({"PassengerId":titanic_test["PassengerId"],
                           "Survived":prediction})
submission.to_csv("titanic_result.csv", index=False)