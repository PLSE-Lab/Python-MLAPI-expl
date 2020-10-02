# Import the Pandas library
import pandas as pd
# Import the Numpy library
import numpy as np
# Import 'tree' from scikit-learn library
from sklearn import tree

# Load the train and test datasets to create two DataFrames
test = pd.read_csv("../input/test.csv")

updatedTest = test

# Create the column Child and assign to 'NaN'
updatedTest["Age"] = 2

# Assign 1 to passengers under 18, 0 to those 18 or older. Print the new column.
updatedTest["Age"][test["Age"] < 18] = 1
updatedTest["Age"][test["Age"] >= 18] = 0

# Initialize a Survived column to 0
updatedTest["Survived"] = 2

# Set Survived to 1 if Sex equals "female" and print the `Survived` column from `test_one`
updatedTest["Survived"][test["Sex"] == "female"] = 1
updatedTest["Survived"][test["Sex"] == "male"] = 0


updatedTest["Fare"] = test["Fare"].fillna(updatedTest["Fare"].median())

# Create the target and features numpy arrays: target, features_one
target = updatedTest["Survived"].values
test_features = updatedTest[["Pclass", "Survived",  "Fare"]].values

print (test_features)

# Fit your first decision tree: my_tree_one
my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit( test_features,target)

# Look at the importance and score of the included features
print(my_tree_one.feature_importances_)
print(my_tree_one.score(test_features, target))

# Make your prediction using the test set
my_prediction = my_tree_one.predict(test_features)

# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
PassengerId =np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])
print(my_solution)

# Check that your data frame has 418 entries
print(my_solution.shape)

# Write your solution to a csv file with the name my_solution.csv
my_solution.to_csv("my_solution.csv", index_label = ["PassengerId"])
sol = pd.read_csv("my_solution.csv")

print(sol)