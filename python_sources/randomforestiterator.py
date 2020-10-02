import numpy as np
import pandas as pd
#from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation as cv
from sklearn.cross_validation import KFold

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

# Cleaning the dataset
def harmonize_data(titanic):

    # Filling the blank data
    titanic["Age"] = titanic["Age"].fillna(titanic["Age"].mean())
    titanic["Fare"] = titanic["Fare"].fillna(titanic["Fare"].mean())
    titanic["Embarked"] = titanic["Embarked"].fillna("S")

    # Assigning binary form to data for calculation purpose
    titanic.loc[titanic["Sex"] == "male", "Sex"] = 1
    titanic.loc[titanic["Sex"] == "female", "Sex"] = 0
    titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
    titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
    titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2

    return titanic

# Creating submission file
def create_submission(dtc, train, test, predictors, filename):

    dtc.fit(train[predictors], train["Survived"])
    predictions = dtc.predict(test[predictors])

    submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predictions
    })

    submission.to_csv(filename, index=False)

# Defining the clean dataset
train_data = harmonize_data(train)
test_data  = harmonize_data(test)

# Feature enginnering
train_data["FamilySize"] = train_data["SibSp"]+train_data["Parch"]+1

test_data["FamilySize"] = test_data["SibSp"]+test_data["Parch"]+1

# Defining predictor
predictors = ["Sex", "Age", "Pclass", "FamilySize"]

#Applying method
max_score = 0
best_n = 0
for n in range(66,67):
    dtc_scr = 0.
    dtc = RandomForestClassifier(max_depth=n)
    for train, test in KFold(len(train_data), n_folds=10, shuffle=True):
        dtc.fit(train_data[predictors], train_data["Survived"])
        dtc_scr += dtc.score(train_data[predictors], train_data["Survived"])/10
    if dtc_scr > max_score:
        max_score = dtc_scr
        best_n = n

print(best_n, max_score)
dtc = RandomForestClassifier(max_depth=best_n)

# Creating submission
create_submission(dtc, train_data, test_data, predictors, "dtcsurvivors.csv")