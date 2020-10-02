import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation

# Initialize our algorithm
alg = LogisticRegression(random_state=1)

# Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

# Fix train data
train["Age"] = train["Age"].fillna(train["Age"].median())

train.loc[train["Sex"] == "male", "Sex"] = 0
train.loc[train["Sex"] == "female", "Sex"] = 1

train["Embarked"] = train["Embarked"].fillna("S")
train.loc[train["Embarked"] == "S", "Embarked"] = 0
train.loc[train["Embarked"] == "C", "Embarked"] = 1
train.loc[train["Embarked"] == "Q", "Embarked"] = 2

train["Fare"] = train["Fare"].fillna(train["Fare"].median())

# Fix test data
test["Age"] = test["Age"].fillna(train["Age"].median())

test.loc[test["Sex"] == "male", "Sex"] = 0
test.loc[test["Sex"] == "female", "Sex"] = 1

test["Embarked"] = test["Embarked"].fillna("S")
test.loc[test["Embarked"] == "S", "Embarked"] = 0
test.loc[test["Embarked"] == "C", "Embarked"] = 1
test.loc[test["Embarked"] == "Q", "Embarked"] = 2

test["Fare"] = test["Fare"].fillna(train["Fare"].median())

# Predictors
predictors = ["Pclass", "Sex", "Age", "SibSp"]

# Train the algorithm using all the training data
alg.fit(train[predictors], train["Survived"])

# Make predictions using the test set.
predictions = alg.predict(test[predictors])
        
# Create a new dataframe with only the columns Kaggle wants from the dataset.
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predictions
    })

# Any files you save will be available in the output tab below
submission.to_csv('submission.csv', index=False)