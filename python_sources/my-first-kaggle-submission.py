import numpy as np
import pandas as pd

# Import 'tree' from scikit-learn library
from sklearn import tree 
#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#Print to standard output, and see the results in the "log" section below after running your script
print("\n\nTop of the training data:")
print(train.head())

print("\n\nSummary statistics of training data")
print(train.describe())

my_train = train

# Convert the male and female groups to integer form
my_train["Sex"][my_train["Sex"] == "male"] = 0
my_train["Sex"][my_train["Sex"] == "female"] = 1
# Impute the Embarked variable
my_train["Embarked"] = my_train["Embarked"].fillna("S")

# Convert the Embarked classes to integer form
my_train["Embarked"][my_train["Embarked"] == "S"] = 0
my_train["Embarked"][my_train["Embarked"] == "C"] = 1
my_train["Embarked"][my_train["Embarked"] == "Q"] = 2

my_train["Age"] = my_train["Age"].fillna(my_train["Age"].median())

print(my_train.head())


# Create the target and features numpy arrays: target, features_one
target = my_train["Survived"].values
features_one = my_train[["Pclass", "Sex", "Age", "Fare"]].values

# Fit your first decision tree: my_tree_one
my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(features_one, target)

# Look at the importance and score of the included features
print(my_tree_one.feature_importances_)
print(my_tree_one.score(features_one, target))
print("\n\nTop of the Test data:")
print(test.head())

test["Age"] = test["Age"].fillna(test["Age"].median())
test.Fare[152] = test.Fare.median()


test["Sex"][test["Sex"] == "male"] = 0
test["Sex"][test["Sex"] == "female"] = 1
# Extract the features from the test set: Pclass, Sex, Age, and Fare.
test_features = test[["Pclass", "Sex", "Age", "Fare"]].values

# Make your prediction using the test set and print them.
my_prediction = my_tree_one.predict(test_features)
print(my_prediction)

# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
PassengerId =np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])
print(my_solution)

# Check that your data frame has 418 entries
print(my_solution.shape)
my_solution.to_csv("5.csv", index_label = ["PassangerId", "Survived"])

#Any files you save will be available in the output tab below
#my_solution.to_csv('first_submission.csv', index=False)