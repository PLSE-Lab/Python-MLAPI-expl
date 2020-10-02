import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import re

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test_sample = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#Fill in missing values using median of column
train["Age"]=train["Age"].fillna(train["Age"].median())

#Change sex to numeric
train.loc[train["Sex"] == "male", "Sex"] = 0
train.loc[train["Sex"] == "female", "Sex"] = 1

#Fill in missing embarked values and convert to numeric
train["Embarked"]=train["Embarked"].fillna("S")

train.loc[train["Embarked"] == "S", "Embarked"] = 0
train.loc[train["Embarked"] == "C", "Embarked"] = 1
train.loc[train["Embarked"] == "Q", "Embarked"] = 2

#define which variables we'll use for predictions
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

#Do same for test sample
test_sample["Age"]=test_sample["Age"].fillna(test_sample["Age"].median())
test_sample["Fare"]=test_sample["Fare"].fillna(test_sample["Fare"].median())

test_sample.loc[test_sample["Sex"] == "male", "Sex"] = 0
test_sample.loc[test_sample["Sex"] == "female", "Sex"] = 1

test_sample["Embarked"]=test_sample["Embarked"].fillna("S")

test_sample.loc[test_sample["Embarked"] == "S", "Embarked"] = 0
test_sample.loc[test_sample["Embarked"] == "C", "Embarked"] = 1
test_sample.loc[test_sample["Embarked"] == "Q", "Embarked"] = 2

#function that takes title from name field
def get_title(name):
    # Use a regular expression to search for a title.  Titles always consist of capital and lowercase letters, and end with a period.
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

# Get all the titles and print how often each one occurs.
titles = train["Name"].apply(get_title)

#convert title to numeric field
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2, "Dona": 10}

for k,v in title_mapping.items():
    titles[titles == k] = v
    
#add new column to train set, then do same for test set
train["Title"] = titles

titles2 = test_sample["Name"].apply(get_title)

for k,v in title_mapping.items():
    titles2[titles2 == k] = v
    
test_sample["Title"] = titles2

train["FamilySize"] = train["SibSp"] + train["Parch"]
test_sample["FamilySize"] = test_sample["SibSp"] + test_sample["Parch"]

predictors.append("Title")
predictors.append("FamilySize")

#define two algorithms
algorithms = [
    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), predictors],
    [LogisticRegression(random_state=1), predictors],
    [RandomForestClassifier(random_state=1, n_estimators=100, min_samples_split=2, min_samples_leaf=3), predictors]
    ]

full_predictions = []
#for each algorithm, fit to train data, then predict on test data
for alg, predictors in algorithms:
    # Fit the algorithm using the full training data.
    alg.fit(train[predictors], train["Survived"])
    # Predict using the test dataset.  We have to convert all the columns to floats to avoid an error.
    predictions = alg.predict_proba(test_sample[predictors].astype(float))[:,1]
    full_predictions.append(predictions)

# The gradient boosting classifier generates better predictions, so we weight it higher.
predictions3 = (full_predictions[0] * 2 + full_predictions[2] + full_predictions[1]) / 4

#convert to binary classifier
predictions3[predictions3 > 0.5] = 1
predictions3[predictions3 <= 0.5] = 0

#make it an integer
predictions3 = predictions3.astype(int)

#create submission file
submission = pd.DataFrame({
        "PassengerId": test_sample["PassengerId"],
        "Survived": predictions3
    })
    
submission.to_csv("kaggle.csv", index=False)