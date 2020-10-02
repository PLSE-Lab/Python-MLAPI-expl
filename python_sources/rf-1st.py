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
print(test.describe())
from sklearn.ensemble import RandomForestClassifier


# We want the Pclass, Age, Sex, Fare,SibSp, Parch, and Embarked variables
target = train["Survived"].values
# Convert the male and female groups to integer form
train.loc[(train["Sex"] == "male"),"Sex"] = 0
train.loc[(train["Sex"] == "female"),"Sex"] = 1
# Impute the Embarked variable
train["Embarked"] = train["Embarked"].fillna("S")
train["Age"] = train["Age"].fillna(train["Age"].median())
# Convert the Embarked classes to integer form
train.loc[(train["Embarked"] == "S"),"Embarked"] = 0
train.loc[(train["Embarked"] == "C"),"Embarked"] = 1
train.loc[(train["Embarked"] == "Q"),"Embarked"] = 2


features_forest = train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values


# Building and fitting my_forest
forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 100, random_state = 1)
my_forest = forest.fit(features_forest, target)

# Print the score of the fitted random forest
print(my_forest.score(features_forest, target))

# Compute predictions on our test set features then print the length of the prediction vector
test["Age"] = test["Age"].fillna(test["Age"].median())
test["Fare"] = test["Fare"].fillna(test["Fare"].median())
test.loc[(test["Sex"] == "male"),"Sex"] = 0
test.loc[(test["Sex"] == "female"),"Sex"] = 1
test["Embarked"] = test["Embarked"].fillna("S")
test.loc[(test["Embarked"] == "S"),"Embarked"] = 0
test.loc[(test["Embarked"] == "C"),"Embarked"] = 1
test.loc[(test["Embarked"] == "Q"),"Embarked"] = 2

test_features = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
pred_forest = my_forest.predict(test_features)
print(len(pred_forest))

PassengerId =np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(pred_forest, PassengerId, columns = ["Survived"])

print(my_solution)

# Check that your data frame has 418 entries
print(my_solution.shape)

# Write your solution to a csv file with the name my_solution.csv
my_solution.to_csv("my_solution_two.csv", index_label = ["PassengerId"])
