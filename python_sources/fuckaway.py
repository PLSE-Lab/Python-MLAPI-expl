import pandas
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
from sklearn import cross_validation

# Read the training set csv file.
titanic = pandas.read_csv("../input/train.csv")

# =====================Preprocessing the data=====================
# Fill the missing value in "Age".
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())

# Converting the Sex Column to numeric value
titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1

# Converting the Embarked Column
titanic["Embarked"] = titanic["Embarked"].fillna("S")
titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2

# =====================Record the prediction======================
# Making predictions with Linear Regression
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
alg = LinearRegression()
kf = KFold(titanic.shape[0], n_folds = 3, random_state = 1)
predictions = []

for train, test in kf:
	train_predictors = (titanic[predictors].iloc[train,:])
	train_target = titanic["Survived"].iloc[train]
	alg.fit(train_predictors, train_target)
	test_predictions = alg.predict(titanic[predictors].iloc[test,:])
	predictions.append(test_predictions)

# Evaluating error and accuracy
predictions = np.concatenate(predictions,axis = 0)
predictions[predictions > .5] = 1
predictions[predictions <= .5] = 0

accuracy = 1 - sum(abs(predictions - titanic["Survived"])) / len(predictions)

print ('Accuracy of Linear Regression on the training set is ' + str(accuracy))

# Logistic Regression
alg = LogisticRegression(random_state = 1)
scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv = 3)
print ('Accuracy of Logistic Regression using cross-validation on the training set is ' + str(scores.mean()))

# ===========================Test Set============================
# Read the test set csv file.
titanic_test = pandas.read_csv("../input/test.csv")

# ====================Preprocessing the data=====================
titanic_test["Age"] = titanic_test["Age"].fillna(titanic["Age"].median())

titanic_test.loc[titanic_test["Sex"] == "male", "Sex"] = 0
titanic_test.loc[titanic_test["Sex"] == "female", "Sex"] = 1

titanic_test["Embarked"] = titanic_test["Embarked"].fillna("S")
titanic_test.loc[titanic_test["Embarked"] == "S", "Embarked"] = 0
titanic_test.loc[titanic_test["Embarked"] == "C", "Embarked"] = 1
titanic_test.loc[titanic_test["Embarked"] == "Q", "Embarked"] = 2

titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic_test["Fare"].median())

alg = LogisticRegression(random_state = 1)
alg.fit(titanic[predictors], titanic["Survived"])

predictions_test = alg.predict(titanic_test[predictors])

print ("The predicted result on the test set is as follows: (1:Survived, 0:Deceased)")
print (predictions_test)

submission = pandas.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": predictions_test
    })