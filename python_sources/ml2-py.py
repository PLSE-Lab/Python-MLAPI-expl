import numpy as np
import pandas as pd
from sklearn import tree
# Import the `RandomForestClassifier`
from sklearn.ensemble import RandomForestClassifier

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )



#Print to standard output, and see the results in the "log" section below after running your script
#print("\n\nTop of the training data:")
print(train.head())


train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1

# Impute the Embarked variable
train["Age"] = train["Age"].fillna(train["Age"].median())
train["Embarked"] = train["Embarked"].fillna("S")

# Convert the Embarked classes to integer form
train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2

# Create the column Child and assign to 'NaN'
train["Child"] = float('NaN')

# Assign 1 to passengers under 18, 0 to those 18 or older. Print the new column.
train["Child"][train["Age"] < 18] = 1
train["Child"][train["Age"] >= 18] = 0
#print("\n\nSummary statistics of training data")
#print(train.describe())
features_forest = train[["Sex", "Child", "Fare"]].values
target = train["Survived"].values

# Extract the features from the test set: Pclass, Sex, Age, and Fare.
test["Sex"][test["Sex"] == "male"] = 0
test["Sex"][test["Sex"] == "female"] = 1

# Impute the Embarked variable
test["Age"] = test["Age"].fillna(test["Age"].median())
test["Embarked"] = test["Embarked"].fillna("S")

# Convert the Embarked classes to integer form
test["Embarked"][test["Embarked"] == "S"] = 0
test["Embarked"][test["Embarked"] == "C"] = 1
test["Embarked"][test["Embarked"] == "Q"] = 2

# Create the column Child and assign to 'NaN'
test["Child"] = float('NaN')

# Assign 1 to passengers under 18, 0 to those 18 or older. Print the new column.
test["Child"][test["Age"] < 18] = 1
test["Child"][test["Age"] >= 18] = 0

# Impute the missing value with the median
test.Fare[152] = test["Fare"].median()

test_features = test[["Sex", "Child", "Fare"]].values


#RandomForest
forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 10, random_state = 1)
my_forest = forest.fit(features_forest, target)

my_prediction = my_forest.predict(test_features)

#Save the results
PassengerId =np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])
my_solution.to_csv("my_solution.csv", index_label = ["PassengerId"])




