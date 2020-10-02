#this is prediction based on decision tree in python



#importing

import pandas as pd

import numpy as np

from sklearn import tree



#reading

train=pd.read_csv("../input/train.csv")

test=pd.read_csv("../input/test.csv")



#copying test and adding target

test_one=test

test["Survived"]=0



#imputing features in train

train["Age"] = train["Age"].fillna(train["Age"].median())

train["Embarked"] = train["Embarked"].fillna("S")



#adding features in train

train["Child"]=float('NaN')

train["Child"][train["Age"] < 18] = 1

train["Child"][train["Age"] >= 18] = 0



train["Family_size"] = train["SibSp"] + train["Parch"] + 1



#converting features in train

train["Sex"][train["Sex"] == "male"] = 0

train["Sex"][train["Sex"] == "female"] = 1



train["Embarked"][train["Embarked"] == "S"] = 0

train["Embarked"][train["Embarked"] == "C"] = 1

train["Embarked"][train["Embarked"] == "Q"] = 2



#imputing features in test

test.Fare[152] = test.Fare.median()

test["Age"] = test["Age"].fillna(test["Age"].median())



#converting features in test

test["Sex"][test["Sex"] == "male"] = 0

test["Sex"][test["Sex"] == "female"] = 1



test["Embarked"][test["Embarked"] == "S"] = 0

test["Embarked"][test["Embarked"] == "C"] = 1

test["Embarked"][test["Embarked"] == "Q"] = 2



#adding features in test

test["Child"]=float('NaN')

test["Child"][test["Age"] < 18] = 1

test["Child"][test["Age"] >= 18] = 0



test["Family_size"] = test["SibSp"] + test["Parch"] + 1



#creating the numpy arrays for target and features

target = train["Survived"].values

features_one = train[["Pclass", "Sex", "Child", "Family_size"]].values



#fitting the decision tree

nkd_tree = tree.DecisionTreeClassifier()

nkd_tree = nkd_tree.fit(features_one, target)

    

#observing the importance and score of the features

print(nkd_tree.feature_importances_)

print(nkd_tree.score(features_one, target))



#extracting the features from the test set

test_features = test[["Pclass", "Sex", "Child", "Family_size"]].values



#making prediction on test set

nkd_prediction = nkd_tree.predict(test_features)



#creating DataFrame

PassengerId = np.array(test["PassengerId"]).astype(int)

nkd_solution = pd.DataFrame(nkd_prediction, PassengerId, columns = ["Survived"])

print(nkd_solution)



#writing solutions to a csv file with the name nkd_submission.csv

nkd_solution.to_csv("nkd_submission.csv", index_label = ["PassengerId"])
