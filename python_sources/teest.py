import sys
import numpy as np
import pandas as pd
from sklearn import tree

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#Print to standard output, and see the results in the "log" section below after running your script
print("\n\nTop of the training data:")
print(train.head())

print("\n\nSummary statistics of training data")
print(train.describe())




#Any files you save will be available in the output tab below
train.to_csv('copy_of_the_training_data.csv', index=False)


# Convert the male and female groups to integer form
train["Sex"][ train["Sex"] == "male" ] = 0
train["Sex"][ train["Sex"] == "female"] = 1

test["Sex"][ test["Sex"] == "male" ] = 0
test["Sex"][ test["Sex"] == "female"] = 1


# Impute the Embarked variable
train["Embarked"] = train["Embarked"].fillna("S")
test["Embarked"] = test["Embarked"].fillna("S")

# Convert the Embarked classes to integer form
train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2


test["Embarked"][test["Embarked"] == "S"] = 0
test["Embarked"][test["Embarked"] == "C"] = 1
test["Embarked"][test["Embarked"] == "Q"] = 2


train["Age"] = train["Age"].fillna(29.7)
test["Age"] = test["Age"].fillna(29.7)
# Print the train data to see the available features
print("\n\ntraining data:")
print(train)


# Create the target and features numpy arrays: target, features_one
target = train["Survived"].values
features_one = train[["Pclass", "Sex", "Age", "Fare"]].values

# Fit your first decision tree: my_tree_one
my_tree_one = tree.DecisionTreeClassifier()

my_tree_one = my_tree_one.fit(features_one, target)
#my_tree_one = my_tree_one.fit(features_one, filtered_target)

# Look at the importance and score of the included 
print("\n\nfeature importances:")
print(my_tree_one.feature_importances_)
print("\n\nscore:")
print(my_tree_one.score(features_one, target))

# Impute the missing value with the median
test.Fare[152] = test.Fare.median()

# Extract the features from the test set: Pclass, Sex, Age, and Fare.
test_features = test[["Pclass", "Sex", "Age", "Fare"]].values

# Make your prediction using the test set
my_prediction = my_tree_one.predict(test_features)
print("\n\nmy_prediction", my_prediction)
# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
PassengerId =np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])
print("\n\nmy_solution",my_solution)

# Check that your data frame has 418 entries
print("\n\nsolution shape",my_solution.shape)

# Write your solution to a csv file with the name my_solution.csv
my_solution.to_csv("my_solution_one.csv", index_label = ["PassengerId"])



