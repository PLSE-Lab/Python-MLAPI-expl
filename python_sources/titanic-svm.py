import numpy as np
import pandas as pd

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

# Convert the male and female groups to integer form
train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1
test["Sex"][test["Sex"] == "male"] = 0
test["Sex"][test["Sex"] == "female"] = 1

# Impute the Embarked and Age variable
train["Embarked"] = train["Embarked"].fillna('S')
train["Age"] = train["Age"].fillna(train["Age"].median())
test["Embarked"] = test["Embarked"].fillna('S')
test["Age"] = test["Age"].fillna(test["Age"].median())
test["Fare"] = test["Fare"].fillna(test["Fare"].median())

# Convert the Embarked classes to integer form
train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2
test["Embarked"][test["Embarked"] == "S"] = 0
test["Embarked"][test["Embarked"] == "C"] = 1
test["Embarked"][test["Embarked"] == "Q"] = 2

#Print the Sex and Embarked columns
print(train["Sex"])
print(train["Embarked"])

# Import the `svm` library
from sklearn import svm

# We want the Pclass, Age, Sex, Fare,SibSp, Parch, and Embarked variables
features = train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
target = train["Survived"].values

print(features[0])

# Create SVM classification object 
svm_model = svm.SVC(kernel='linear', C=1, gamma=1) 
svm_model.fit(features, target)

# Print the score of the fitted svm
print(svm_model.score(features, target))

# Compute predictions on our test set features then print the length of the prediction vector
test_features = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
prediction = svm_model.predict(test_features)

PassengerId = np.array(test["PassengerId"]).astype(int)

print(PassengerId)

my_solution = pd.DataFrame( {'PassengerId': PassengerId,
                             'Survived': prediction} )

#pd.set_option('display.max_rows', 500)
my_solution.to_csv("titanic_trial_svm", index=False)
# print(len(pred_forest))