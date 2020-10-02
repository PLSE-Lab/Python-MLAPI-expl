__author__ = 'ali'
import pandas as pd
import numpy as np
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
train["Embarked"] = train["Embarked"].fillna("S")


# # Convert the Embarked classes to integer form
train.loc[train["Embarked"] == "S", 'Embarked'] = 0
train.loc[train["Embarked"] == "C", 'Embarked'] = 1
train.loc[train["Embarked"] == "Q", 'Embarked'] = 2



target = train["Survived"].values

train["Pclass"] = train["Pclass"].fillna(1)
train["Age"]= train["Age"].fillna(train.Age.median())
train["Fare"]= train["Fare"].fillna(train.Fare.median())
train["Sex"]= train["Sex"].fillna(train.Sex.median())

# print(train["Fare"])
features_one = train[["Pclass", "Sex", "Age", "Fare"]].values

# # # Fit your first decision tree: my_tree_one
my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(features_one, target)
#
#
# #work with test data:
#
# # Impute the missing value with the median
# test.Fare[152] =test.Fare.median()
#
test["Pclass"] = test["Pclass"].fillna(test.Pclass.median())
test["Age"]= test["Age"].fillna(test.Age.median())
test["Fare"]= test["Fare"].fillna(test.Fare.median())

test.loc[test["Sex"] == "male", 'Sex'] = 0
test.loc[test["Sex"] == "female", 'Sex'] = 1

test["Sex"]= test["Sex"].fillna(test.Sex.median())


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
features_two = train[["Pclass","Age","Sex","Fare", "SibSp", "Parch", "Embarked"]].values
#
#Control overfitting by setting "max_depth" to 10 and "min_samples_split" to 5 : my_tree_two
max_depth = 10
min_samples_split = 5
my_tree_two = tree.DecisionTreeClassifier(max_depth = 10, min_samples_split = 5, random_state = 1)
my_tree_two = my_tree_two.fit(features_two, target)
#
# #Print the score of the new decison tree
# print(my_tree_two.score(features_two, target))
#
# # Create train_two with the newly defined feature
train_two = train.copy()
train_two["family_size"] = train["SibSp"] + train["Parch"] + 1

test["family_size"] =  test["SibSp"] + test["Parch"] + 1
#
# # Create a new feature set and add the new feature
features_three = train_two[["Pclass", "Sex", "Age", "Fare","family_size"]].values
#
# # Define the tree classifier, then fit the model
my_tree_three = tree.DecisionTreeClassifier()
my_tree_three = my_tree_three.fit(features_three, target)

test_features = test[["Pclass", "Sex", "Age", "Fare", "family_size"]].values
my_prediction = my_tree_three.predict(test_features)
# print(my_prediction)

PassengerId =np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])
# print(my_solution)
my_solution.to_csv("my_solution_one.csv", index_label = ["PassengerId"])

#
# # Print the score of this decision tree
# # print(my_tree_three.score(features_three, target))




#
#
# # Import the `RandomForestClassifier`
# from sklearn.ensemble import RandomForestClassifier
#
# # We want the Pclass, Age, Sex, Fare,SibSp, Parch, and Embarked variables
# features_forest = train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
#
# # Building and fitting my_forest
# forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 100, random_state = 1)
# my_forest = forest.fit(features_forest, target)
#
# # Print the score of the fitted random forest
# print(my_forest.score(features_forest, target))
#
# # Compute predictions on our test set features then print the length of the prediction vector
# test_features = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
# pred_forest = my_forest.predict(test_features)
# print(len(pred_forest))