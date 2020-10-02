import numpy as np
import pandas as pd

train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#Print to standard output, and see the results in the "log" section below after running your script
#print("\n\nTop of the training data:")
#print(train.head())

#print("\n\nSummary statistics of training data")
#print(train.describe())

#Any files you save will be available in the output tab below
#train.to_csv('copy_of_the_training_data.csv', index=False)


## Kaggle Python Tutorial on Machine Learning


## Chapter 2: Predicting with Decision Trees


## Exercise 1: Intro to decision trees
from sklearn import tree 

## Exercise 2: Cleaning and Formatting your Data

# (hidden step) Fill missing ages with median
train["Age"] = train["Age"].fillna(train["Age"].median())

#Convert the male and female groups to integer form
train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1

# (hidden step) same thing for gender
test["Sex"][test["Sex"] == "male"] = 0
test["Sex"][test["Sex"] == "female"] = 1

#Impute the Embarked variable
train["Embarked"] = train["Embarked"].fillna("S")

#Convert the Embarked classes to integer form
train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2

#Print the Sex and Embarked columns
#print(train["Embarked"])
#print(train["Sex"])


## Exercise 3: Creating your first decision tree

# Print the train data to see the available features
# print(train)

# Create the target and features numpy arrays: target, features_one
target = train["Survived"].values
features_one = train[["Pclass", "Sex", "Age", "Fare"]].values

# Fit your first decision tree: my_tree_one
my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(features_one, target)

# Look at the importance and score of the included features
print(my_tree_one.feature_importances_)
print(my_tree_one.score(features_one, target))


## Exercise 4: Interpreting your decision tree

# print(my_tree_one.feature_importances_)


## Exercise 5: Predict and submit to Kaggle

# Impute the missing value with the median
print(test.Fare[152])
test.Fare[152] = test["Fare"].median()
print("cleaned:", test.Fare[152])

# Extract the features from the test set: Pclass, Sex, Age, and Fare.
test_features = test[["Pclass", "Sex", "Age", "Fare"]].values

# Make your prediction using the test set
my_prediction = my_tree_one.predict(test_features)

# # Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
PassengerId = np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])
print(my_solution)

# # Check that your data frame has 418 entries
print(my_solution.shape)

# # Write your solution to a csv file with the name my_solution.csv
my_solution.to_csv("my_solution_one.csv", index_label = ["PassengerId"])


# ## Exercise 6: Overfitting and how to control it

# # Create a new array with the added features: features_two
# features_two = train[["Pclass","Age","Sex","Fare", "SibSp", "Parch", "Embarked"]].values

# #Control overfitting by setting "max_depth" to 10 and "min_samples_split" to 5 : my_tree_two
# max_depth = 10
# min_samples_split = 5
# my_tree_two = tree.DecisionTreeClassifier(max_depth = max_depth, min_samples_split = min_samples_split, random_state = 1)
# my_tree_two = my_tree_two.fit(features_two, target)

# #Print the score of the new decison tree
# print(my_tree_two.score(features_two, target))


# ## Exercise 7: Feature-engineering for our Titanic data set

# # Create train_two with the newly defined feature
# train_two = train.copy()
# train_two["family_size"] = train_two["SibSp"] + train_two["Parch"] + 1

# # Create a new feature set and add the new feature
# features_three = train_two[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "family_size"]].values

# # Define the tree classifier, then fit the model
# my_tree_three = tree.DecisionTreeClassifier()
# my_tree_three = my_tree_three.fit(features_three, target)

# # Print the score of this decision tree
# print(my_tree_three.score(features_three, target))