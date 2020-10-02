import numpy as np
import pandas as pd
from sklearn import tree

from sklearn.ensemble import RandomForestClassifier
#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float32}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float32}, )

#Print to standard output, and see the results in the "log" section below after running your script
print("\n\nTop of the training data:")
print(train.head())

print("\n\nSummary statistics of training data")
print(train.describe())

#Any files you save will be available in the output tab below
train.to_csv('copy_of_the_training_data.csv', index=False)
test_one=test

# Initialize a Survived column to 0
test_one["Survived"]=0

# Set Survived to 1 if Sex equals "female" and print the `Survived` column from `test_one`
test_one["Survived"][test_one["Sex"] == 'female'] =1
print(test_one["Survived"])
#first prediction based on gender

train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] =1

# Impute the Embarked variable
train["Embarked"] = train["Embarked"].fillna("S")

train["Age"] = train["Age"].fillna(train["Age"].median())
# Convert the Embarked classes to integer form
train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"]=="C"] = 1
train["Embarked"][train["Embarked"]=="Q"] = 2


test["Sex"][test["Sex"] == "male"] = 0
test["Sex"][test["Sex"] == "female"] =1

# Impute the Embarked variable
test["Embarked"] = test["Embarked"].fillna("S")

test["Age"] = test["Age"].fillna(test["Age"].median())
# Convert the Embarked classes to integer form
test["Embarked"][test["Embarked"] == "S"] = 0
test["Embarked"][test["Embarked"]=="C"] = 1
test["Embarked"][test["Embarked"]=="Q"] = 2

#Print the Sex and Embarked columns
print(train["Sex"])
print(train["Embarked"])

target = train["Survived"].values
features_one = train[["Pclass", "Sex", "Age", "Fare"]].values

# Fit your first decision tree: my_tree_one
my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(features_one,target)

# Look at the importance and score of the included features
print(my_tree_one.feature_importances_)
print(my_tree_one.score(features_one,target))

# Impute the missing value with the median
test.Fare[152] = test.Fare.median()

# Extract the features from the test set: Pclass, Sex, Age, and Fare.
test_features = test[["Pclass","Sex","Age", "Fare"]].values

# Make your prediction using the test set
my_prediction = my_tree_one.predict(test_features)

# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
PassengerId =np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])
print(my_solution)

# Check that your data frame has 418 entries
print(my_solution.shape)

# Write your solution to a csv file with the name my_solution.csv
my_solution.to_csv("my_solution_one.csv", index_label = ["PassengerId"])
print(my_solution.shape)

features_two = train[["Pclass","Age","Sex","Fare","SibSp", "Parch","Embarked"]].values

#Control overfitting by setting "max_depth" to 10 and "min_samples_split" to 5 : my_tree_two
max_depth = 10
min_samples_split = 5
my_tree_two = tree.DecisionTreeClassifier(max_depth = 10, min_samples_split = 5, random_state = 1)
my_tree_two = my_tree_two.fit(features_two,target)

#Print the score of the new decison tree
print(my_tree_two.score(features_two,target))

train_two = train.copy()
train_two["family_size"] = train_two["SibSp"] + train_two["Parch"] +1

# Create a new feature set and add the new feature
features_three = train_two[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch","family_size"]].values

# Define the tree classifier, then fit the model
my_tree_three = tree.DecisionTreeClassifier()
my_tree_three = my_tree_three.fit(features_three,target)

# Print the score of this decision tree
print(my_tree_three.score(features_three, target))

# We want the Pclass, Age, Sex, Fare,SibSp, Parch, and Embarked variables
features_forest = train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values

# Building and fitting my_forest
forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 100, random_state = 1)
my_forest = forest.fit(features_forest,target)

# Print the score of the fitted random forest
print(my_forest.score(features_forest, target))

# Compute predictions on our test set features then print the length of the prediction vector
test_features = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
pred_forest = my_forest.predict(test_features)
print(len(pred_forest))
PassengerId =np.array(test["PassengerId"]).astype(int)
my_solution_forest = pd.DataFrame(pred_forest, PassengerId, columns = ["Survived"])
print(my_solution_forest)

# Check that your data frame has 418 entries
print(my_solution_forest.shape)

# Write your solution to a csv file with the name my_solution.csv
my_solution_forest.to_csv("my_solution_forest_one.csv", index_label = ["PassengerId"])
print(my_solution_forest.shape)

print(my_tree_two.score(features_two, target))
print(my_forest.score(features_two,target))

