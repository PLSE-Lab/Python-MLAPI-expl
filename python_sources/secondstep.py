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


test.Fare[152] = test.Fare.median()

train_two = train.copy()
train_two["family_size"] = train_two["SibSp"] + train_two["Parch"] +1
target = train_two["Survived"].values
# Create a new feature set and add the new feature
features_three = train_two[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch","family_size","Embarked"]].values



# Building and fitting my_forest
forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 100, random_state = 1)
my_forest = forest.fit(features_three,target)

# Print the score of the fitted random forest
print(my_forest.score(features_three, target))
test["family_size"] = test["SibSp"] + test["Parch"] +1
# Compute predictions on our test set features then print the length of the prediction vector
test_features = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch","family_size", "Embarked"]].values
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



