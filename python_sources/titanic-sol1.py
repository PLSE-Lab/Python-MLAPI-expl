# Import the Pandas & numpy library
import pandas as pd 
import numpy as np 
# Import 'tree' from scikit-learn library
from sklearn import tree 
# Import the `RandomForestClassifier`
from sklearn.ensemble import RandomForestClassifier

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# Create the column family_size 
test["family_size"] = test["SibSp"] + test["Parch"] + 1
train["family_size"] = train["SibSp"] + train["Parch"] + 1

# Convert the male and female groups to integer form
test["Sex"].loc[test["Sex"] == "male"] = 0
test["Sex"].loc[test["Sex"] == "female"] = 1
train["Sex"].loc[train["Sex"] == "male"] = 0
train["Sex"].loc[train["Sex"] == "female"] = 1
# Impute the Embarked variable
test["Embarked"] = test["Embarked"].fillna("S")
train["Embarked"] = train["Embarked"].fillna("S")
# Convert the Embarked classes to integer form

test["Embarked"].loc[test["Embarked"] == "S"] = 0
test["Embarked"].loc[test["Embarked"] == "C"] = 1
test["Embarked"].loc[test["Embarked"] == "Q"] = 2

train["Embarked"].loc[train["Embarked"] == "S"] = 0
train["Embarked"].loc[train["Embarked"] == "C"] = 1
train["Embarked"].loc[train["Embarked"] == "Q"] = 2

# Impute the missing value with the median
test["Fare"].loc[152] = test.Fare.median()
train["Age"].fillna(train["Age"].median(), inplace = True)
test["Age"].fillna(test["Age"].median(), inplace = True)
# Create the column Child
test["Child"] = 0
train["Child"] = 0
# Assign 1 to passengers under 12, 0 to those 12 or older. Print the new column.
test["Child"].loc[test["Age"] <  6.5] = 1
test["Child"].loc[test["Age"] >= 6.5] = 0
train["Child"].loc[train["Age"] <  6.5] = 1
train["Child"].loc[train["Age"] >= 6.5] = 0

test["Old"] = 0
train["Old"] = 0
# Assign 0 to passengers under 55, 1 to those 55 or older. Print the new column.
test["Old"].loc[test["Age"] <  55] = 0
test["Old"].loc[test["Age"] >= 55] = 1
train["Old"].loc[train["Age"] <  55] = 0
train["Old"].loc[train["Age"] >= 55] = 1

#Control overfitting by setting "max_depth" to 10 and "min_samples_split" to 5
# Define the tree classifier
my_tree = tree.DecisionTreeClassifier(max_depth = 30, min_samples_split = 5, random_state = 1)

target = train["Survived"].values
features_one = train[["Pclass", "Fare", "Sex", "Embarked", "family_size", "Child", "Parch", "Old"]].values
my_tree = my_tree.fit(features_one, target)

test_features = test[[ "Pclass", "Fare", "Sex", "Embarked", "family_size", "Child", "Parch", "Old"]].values
my_prediction = my_tree.predict(test_features)
print(my_prediction)

# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
PassengerId =np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])
print(my_solution)
# Check that your data frame has 418 entries
print(my_solution.shape)
# Write your solution to a csv file with the name my_solution.csv
my_solution.to_csv("my_solution_one.csv", index_label = ["PassengerId"])