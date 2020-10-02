import pandas
from sklearn.cross_validation import *
from sklearn.linear_model import *
import numpy as np


titanic = pandas.read_csv("../input/train.csv")


# Replace all the missing ages in the data with the median age
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())

print (titanic.describe())
# Replace all male and female genders with '0's and '1's respectively
titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1

print (titanic.head())
print (titanic["Embarked"].unique())

# Replace all the empty port calls with S
titanic["Embarked"] = titanic["Embarked"].fillna("S")
print (titanic["Embarked"].unique())

# Replace the respective port calls with 0,1 and 2
titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2

# Using Scikit Learn to make linear-regression predictions on the target
# Using these features for prediction
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

# Initialize the algorithm
alg = LinearRegression()
kf = KFold(titanic.shape[0], n_folds=3, random_state=1)

predictions = []
for train, test in kf:
    train_predictors = (titanic[predictors].iloc[train, :])  # the features for training (x1, x2...xn)
    train_target = titanic["Survived"].iloc[train]  # the predictive target (y)
    alg.fit(train_predictors, train_target)  # finding the best fit for the target (using what? Gradient Descent?)
    test_predictions = alg.predict(titanic[predictors].iloc[test, :])  # predict based on the best fit produced by alg.fit
    predictions.append(test_predictions)

# Evaluating error - i.e. checking against the actual list of survived/died

predictions = np.concatenate(predictions)

print (predictions)

predictions[predictions > 0.5] = 1
predictions[predictions < 0.5] = 0

# Dividing the number of right predictions by the total count
accuracy = np.count_nonzero(titanic["Survived"] == predictions)/titanic["Survived"].count()
print (accuracy)

# Using logistic regression to make predictions

alg_logReg = LogisticRegression(random_state=1)

# Compute the accuracy score for all the cross validation folds: returns an array of the scores from the 3 folds
logReg_scores = cross_val_score(alg_logReg, titanic[predictors], titanic["Survived"], cv=3)

print("Scores from logistic regression: " + str(logReg_scores.mean()))

# Submitting the assignment with test.csv

titanic_test = pandas.read_csv("../input/test.csv")
print(titanic_test.describe())

# fill in the blank entries in the age with the median age
titanic_test["Age"] = titanic_test["Age"].fillna(titanic["Age"].median())

# replace genders with numbers
titanic_test.loc[titanic_test["Sex"] == "male", "Sex"] = 0
titanic_test.loc[titanic_test["Sex"] == "female", "Sex"] = 1

# fill in missing values in embarked with 'S'

titanic_test["Embarked"] = titanic_test["Embarked"].fillna("S")
print(titanic_test["Embarked"].unique())

# replace Embarked initials with letters

titanic_test.loc[titanic_test["Embarked"] == "S", "Embarked"] = 0
titanic_test.loc[titanic_test["Embarked"] == "C", "Embarked"] = 1
titanic_test.loc[titanic_test["Embarked"] == "Q", "Embarked"] = 2

# replace missing value in the Fare column

titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic_test["Fare"].median())

# use the training data to fit the logistic regression algo
# use the algo to apply on the test set
alg_logReg.fit(titanic[predictors], titanic["Survived"])
predictions = alg_logReg.predict(titanic_test[predictors])

# create a submission to kaggle

submission = pandas.DataFrame({
    "PassengerId": titanic_test["PassengerId"],
    "Survived": predictions
})

submission.to_csv("kaggle.csv", index=False)
