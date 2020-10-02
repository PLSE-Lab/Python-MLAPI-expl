print('Importing libraries...')
import numpy as np
import pandas as pd
from sklearn import cross_validation as cv
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier

print('Fetching the training and test datasets...')
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test  = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

print('Cleaning the dataset...')
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

print('Defining submission file...')    
def create_submission(rfc, train, test, predictors, filename):
    rfc.fit(train[predictors], train["Survived"])
    predictions = rfc.predict(test[predictors])
    submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predictions
    })
    submission.to_csv(filename, index=False)

print('Defining the clean dataset...')    
train_data = harmonize_data(train)
test_data  = harmonize_data(test)

print('Performing feature enginnering...') 
train_data["PSA"] = train_data["Pclass"]*train_data["Sex"]*train_data["Age"]
train_data["SP"] = train_data["SibSp"]+train_data["Parch"]
test_data["PSA"] = test_data["Pclass"]*test_data["Sex"]*test_data["Age"]
test_data["SP"] = test_data["SibSp"]+test_data["Parch"]

print('Defining predictors...')
predictors = ["Pclass", "Sex", "Age", "PSA", "Fare", "Embarked", "SP"]

print('Finding best n_estimators for RandomForestClassifier...')
max_score = 0
best_n = 0
for n in range(1,300):
    rfc_scr = 0.
    rfc = RandomForestClassifier(n_estimators=n)
    for train, test in KFold(len(train_data), n_folds=10, shuffle=True):
        rfc.fit(train_data[predictors].T[train].T, train_data["Survived"].T[train].T)
        rfc_scr += rfc.score(train_data[predictors].T[test].T, train_data["Survived"].T[test].T)/10
    if rfc_scr > max_score:
        max_score = rfc_scr
        best_n = n
print(best_n, max_score)

print('Finding best max_depth for RandomForestClassifier...')
max_score = 0
best_m = 0
for m in range(1,100):
    rfc_scr = 0.
    rfc = RandomForestClassifier(max_depth=m)
    for train, test in KFold(len(train_data), n_folds=10, shuffle=True):
        rfc.fit(train_data[predictors].T[train].T, train_data["Survived"].T[train].T)
        rfc_scr += rfc.score(train_data[predictors].T[test].T, train_data["Survived"].T[test].T)/10
    if rfc_scr > max_score:
        max_score = rfc_scr
        best_m = m
print(best_n, max_score)

print('Applying method...')
rfc = RandomForestClassifier(n_estimators=best_n, max_depth=best_m)
print('Creating submission...')
create_submission(rfc, train_data, test_data, predictors, "rfcsurvivors.csv")
print('Submitted.')