import numpy as np
import pandas as pd
from sklearn import tree

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"]= 1

train["Age"] = train["Age"].fillna(train["Age"].median())
train["Embarked"] = train["Embarked"].fillna("S")

train["Embarked"][train["Embarked"]=="S"] = 0
train["Embarked"][train["Embarked"]=="C"] = 1
train["Embarked"][train["Embarked"]=="Q"] = 2

#target and features_one decision arrays
target = train["Survived"].values
features_one = train[["Pclass","Age","Sex","Fare"]].values

#Fitting the decision tree
my_tree = tree.DecisionTreeClassifier()
my_tree = my_tree.fit(features_one,target)

print(my_tree.feature_importances_)