# Import the Pandas library
import pandas as pd
# Import the Numpy library
import numpy as np
# Import 'tree' from scikit-learn library
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

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


# Convert the Embarked classes to integer form
updatedTest["Embarked"][updatedTest["Embarked"] == "S"] = 0
updatedTest["Embarked"][updatedTest["Embarked"] == "C"] = 1
updatedTest["Embarked"][updatedTest["Embarked"] == "Q"] = 2

# Create the target and features numpy arrays: target, features_one
target = updatedTest["Survived"].values
test_features = updatedTest[["Pclass", "Age", "Survived", "Fare", "SibSp", "Parch", "Embarked"]].values
#test_features = updatedTest[["Pclass", "Survived",  "Fare"]].values

print (test_features)



# Building and fitting my_forest
forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 100, random_state = 1)
my_forest = forest.fit(test_features, target)

# Print the score of the fitted random forest
print(my_forest.score(test_features, target))

# Compute predictions on our test set features then print the length of the prediction vector
test_features = test[["Pclass", "Age", "Survived", "Fare", "SibSp", "Parch", "Embarked"]].values
pred_forest = my_forest.predict(test_features)
print(len(pred_forest))


# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
PassengerId =np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(pred_forest, PassengerId, columns = ["Survived"])
print(my_solution)

# Check that your data frame has 418 entries
print(my_solution.shape)

# Write your solution to a csv file with the name my_solution.csv
my_solution.to_csv("my_solution.csv", index_label = ["PassengerId"])
sol = pd.read_csv("my_solution.csv")

print(sol)