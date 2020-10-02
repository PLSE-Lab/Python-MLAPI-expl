import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#Print to standard output, and see the results in the "log" section below after running your script
print("\n\nTop of the training data:")
print(train.head())

print("\n\nSummary statistics of training data")
print(train.describe())

#Any files you save will be available in the output tab below
#train.to_csv('copy_of_the_training_data.csv', index=False)

# Convert the male and female groups to integer form
train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1

#Filling nan spaces in test Age
train["Age"] = train["Age"].fillna(np.nanmedian(train["Age"]))
#print(train["Age"])
#print(np.nanmedian(train["Age"]))

target = train["Survived"].values
features_forest = train[["Pclass", "Age", "Sex", "Fare"]].values

# Building and fitting my_forest
forest = RandomForestClassifier(max_depth = 20, min_samples_split=5, n_estimators = 500, random_state = 1)
my_forest = forest.fit(features_forest, target)

# Print the score of the fitted random forest
print(my_forest.score(features_forest, target))

# TEST START

print("\n\nSummary statistics of test data")
print(test.describe())

# Convert the male and female groups to integer form
test["Sex"][test["Sex"] == "male"] = 0
test["Sex"][test["Sex"] == "female"] = 1

#Filling nan spaces in test Age and Fare
test["Age"] = test["Age"].fillna(np.nanmedian(test["Age"]))
#print(test["Age"])
test["Fare"] = test["Fare"].fillna(np.nanmedian(test["Fare"]))
#print(test["Fare"])

print("\n\nSummary statistics of test data")
print(test.describe())

#Compute predictions on our test set features then print the length of the prediction vector
test_features = test[["Pclass", "Age", "Sex", "Fare"]].values
pred_forest = my_forest.predict(test_features)
print(len(pred_forest))

# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
PassengerId =np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(pred_forest, PassengerId, columns = ["Survived"])
print(my_solution)

# Check that your data frame has 418 entries
print(my_solution.shape)

# Write your solution to a csv file with the name my_solution.csv
my_solution.to_csv("my_solution_one.csv", index_label = ["PassengerId"])