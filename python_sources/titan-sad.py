import numpy as np
import pandas as pd
import csv as csv
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
import re
import operator
from sklearn.linear_model import LinearRegression
# Sklearn also has a helper that makes it easy to do cross validation
from sklearn.cross_validation import KFold

titanic = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())

# Initialize our algorithm with the default paramters
# n_estimators is the number of trees we want to make
# min_samples_split is the minimum number of rows we need to make a split
# min_samples_leaf is the minimum number of samples we can have at the place where a tree branch ends (the bottom points of the tree)
titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1
titanic["Embarked"] = titanic["Embarked"].fillna("S")
titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2


test["Age"] = test["Age"].fillna(test["Age"].median())
test.loc[test["Sex"] == "male", "Sex"] = 0
test.loc[test["Sex"] == "female", "Sex"] = 1
test["Embarked"] = test["Embarked"].fillna("S")
test.loc[test["Embarked"] == "S", "Embarked"] = 0
test.loc[test["Embarked"] == "C", "Embarked"] = 1
test.loc[test["Embarked"] == "Q", "Embarked"] = 2

ids = []
test_data=[] #Creat a variable called 'test_data'
for id in test["PassengerId"]: #Skip through each row in the csv file
    ids.append(id)
    # test_data.append(row[1:]) #adding each row to the data variable
test_data = np.array(test_data) #Then convert from a list to an array

# The columns we'll use to predict the target
predictors = ["Sex", "Pclass", "Age", "Parch", "Fare"]
# Initialize our algorithm class
alg = LinearRegression()
# Generate cross validation folds for the titanic dataset.  It return the row indices corresponding to train and test.
# We set random_state to ensure we get the same splits every time we run this.
kf = KFold(titanic.shape[0], n_folds=3, random_state=1)

predictions = []
for train, test in kf:
    # The predictors we're using the train the algorithm.  Note how we only take the rows in the train folds.
    train_predictors = (titanic[predictors].iloc[train,:])
    # The target we're using to train the algorithm.
    train_target = titanic["Survived"].iloc[train]
    # Training the algorithm using the predictors and target.
    alg.fit(train_predictors, train_target)
    # We can now make predictions on the test fold
    test_predictions = alg.predict(titanic[predictors].iloc[test,:])
    predictions.append(test_predictions)

predictions = np.concatenate(predictions, axis=0)

# Map predictions to outcomes (only possible outcomes are 1 and 0)
predictions[predictions > .5] = 1
predictions[predictions <=.5] = 0
# accuracy = sum(predictions[predictions == titanic["Survived"]]) / len(predictions)


alg = RandomForestClassifier(random_state=1, n_estimators=100, min_samples_split=2, min_samples_leaf=2)
scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=3)
print(scores.mean())
print(predictions.size)

# #Any files you save will be available in the output tab below

open_file_object = csv.writer(open("myfirstforest.csv", "w"))
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, predictions))
#open_file_object.to_csv('copy_of_the_training_data.csv', index=False)