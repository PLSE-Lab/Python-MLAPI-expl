__author__ = 'ali'
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn import tree

# train = pd.read_csv('./train.csv')
# test = pd.read_csv('./test.csv')

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train["Child"] = float('NaN')
train.loc[train["Age"] < 18, 'Child'] = int(1)
# train["Child"] = train["Child"][train["Age"] < 18] = 1
train.loc[train["Age"] >= 18, 'Child'] = int(0)
# train["Child"][train["Age"] >= 18] = 0

# print(train[['Age', 'Child']])

train.loc[train["Sex"] == "male", 'Sex'] = 0
train.loc[train["Sex"] == "female", 'Sex'] = 1
# train["Sex"][train["Sex"] == "female"] = 1




# #todo: why?? impute the Embarked ...
# print(train["Embarked"])



# # Convert the Embarked classes to integer form
train["Embarked"] = train["Embarked"].fillna("S")
train.loc[train["Embarked"] == "S", 'Embarked'] = 0
train.loc[train["Embarked"] == "C", 'Embarked'] = 1
train.loc[train["Embarked"] == "Q", 'Embarked'] = 2

target = train["Survived"].values

train["Pclass"] = train["Pclass"].fillna(1)
train["Age"] = train["Age"].fillna(train.Age.median())
train["Fare"] = train["Fare"].fillna(train.Fare.median())
train["Sex"] = train["Sex"].fillna(train.Sex.median())

# print(train["Fare"])
features_one = train[["Pclass", "Sex", "Age", "Fare"]].values

# # # Fit your first decision tree: my_tree_one
# my_tree_one = tree.DecisionTreeClassifier()
# my_tree_one = my_tree_one.fit(features_one, target)


test["Pclass"] = test["Pclass"].fillna(test.Pclass.median())
test["Age"] = test["Age"].fillna(test.Age.median())
test["Fare"] = test["Fare"].fillna(test.Fare.median())

test.loc[test["Sex"] == "male", 'Sex'] = 0
test.loc[test["Sex"] == "female", 'Sex'] = 1

test["Sex"] = test["Sex"].fillna(test.Sex.median())
test["Embarked"] = test["Embarked"].fillna("S")

test.loc[test["Embarked"] == "S", 'Embarked'] = 0
test.loc[test["Embarked"] == "C", 'Embarked'] = 1
test.loc[test["Embarked"] == "Q", 'Embarked'] = 2


# test_features = test[["Pclass","Sex","Age","Fare"]].values
# my_prediction = my_tree_one.predict(test_features)
# print(my_prediction)
#
# PassengerId =np.array(test["PassengerId"]).astype(int)
# my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])
# print(my_solution)
# my_solution.to_csv("my_solution_one.csv", index_label = ["PassengerId"])
#
# Create a new array with the added features: features_two
# features_two = train[["Pclass","Age","Sex","Fare", "SibSp", "Parch", "Embarked"]].values
#
# Control overfitting by setting "max_depth" to 10 and "min_samples_split" to 5 : my_tree_two
# max_depth = 10
# min_samples_split = 5
# my_tree_two = tree.DecisionTreeClassifier(max_depth = 10, min_samples_split = 5, random_state = 1)
# my_tree_two = my_tree_two.fit(features_two, target)
#
# #Print the score of the new decison tree
# print(my_tree_two.score(features_two, target))
#
# # Create train_two with the newly defined feature

train["SibSp"] = train["SibSp"].fillna(train.SibSp.median())
train["Parch"] = train["Parch"].fillna(train.Parch.median())
train["family_size"] = train["SibSp"] + train["Parch"] + 1

train["Pclass"] = train["Pclass"].fillna(train.Pclass.median())

test["SibSp"] = test["SibSp"].fillna(test.SibSp.median())
test["Parch"] = test["Parch"].fillna(test.Parch.median())

test["family_size"] = test["SibSp"] + test["Parch"] + 1

test["Pclass"] = test["Pclass"].fillna(test.Pclass.median())

# float to int
# train["Embarked"] = train["Embarked"].astype(int)

# features_three = train[["Pclass", "Sex", "Age", "Fare", "family_size"]].values
# my_tree_three = tree.DecisionTreeClassifier()
# my_tree_three = my_tree_three.fit(features_three, target)
# test_features = test[["Pclass", "Sex", "Age", "Fare", "family_size"]].values
# my_prediction = my_tree_three.predict(test_features)


# PassengerId =np.array(test["PassengerId"]).astype(int)
# my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])
# # print(my_solution)
# my_solution.to_csv("my_solution_one.csv", index_label = ["PassengerId"])

# features_forest = train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]]
# forest = RandomForestClassifier(max_depth=10, min_samples_split=2, n_estimators=100, random_state=1)
# # print(features_forest["Embarked"].values)
# my_forest = forest.fit(features_forest, target)

# pred_forest = my_forest.predict(test_features)
#
# my_prediction = pred_forest
# PassengerId = np.array(test["PassengerId"]).astype(int)
# my_solution = pd.DataFrame(my_prediction, PassengerId, columns=["Survived"])
# # print(my_solution)
# my_solution.to_csv("my_solution_one.csv", index_label=["PassengerId"])

# print(train.info())

test_features = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values


features_logistics = train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]]

logreg = LogisticRegression()

logreg.fit(features_logistics, target)

Y_pred = logreg.predict(test_features)



PassengerId =np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(Y_pred, PassengerId, columns = ["Survived"])
# print(Y_pred)
my_solution.to_csv("my_solution_one.csv", index_label = ["PassengerId"])
