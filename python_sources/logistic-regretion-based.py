import numpy as np
import pandas 
import sklearn
# Import the linear regression class
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
# Sklearn also has a helper that makes it easy to do cross validation
from sklearn.cross_validation import KFold

#Print you can execute arbitrary python code
titanic_train = pandas.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
titanic_test = pandas.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

###############################################################################
###############################################################################
## Preprocess training set
titanic_train["Age"] = titanic_train["Age"].fillna(titanic_train["Age"].median())

titanic_train.loc[titanic_train["Sex"] == "male", "Sex"] = 0
titanic_train.loc[titanic_train["Sex"] == "female", "Sex"] = 1

titanic_train["Embarked"] = titanic_train["Embarked"].fillna('S')
titanic_train.loc[titanic_train["Embarked"] == "S", "Embarked"] = 0
titanic_train.loc[titanic_train["Embarked"] == "C", "Embarked"] = 1
titanic_train.loc[titanic_train["Embarked"] == "Q", "Embarked"] = 2

## Preprocess test set
titanic_test["Age"] = titanic_test["Age"].fillna(titanic_train["Age"].median())

titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic_test["Fare"].median())
titanic_test["Embarked"] = titanic_test["Embarked"].fillna('S')
titanic_test.loc[titanic_test["Embarked"] == "S", "Embarked"] = 0
titanic_test.loc[titanic_test["Embarked"] == "C", "Embarked"] = 1
titanic_test.loc[titanic_test["Embarked"] == "Q", "Embarked"] = 2

titanic_test.loc[titanic_test["Sex"] == "male", "Sex"] = 0
titanic_test.loc[titanic_test["Sex"] == "female", "Sex"] = 1


###############################################################################
###############################################################################
# The columns we'll use to predict the target
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

# Initialize our algorithm class
alg = LinearRegression()
# Generate cross validation folds for the titanic dataset.  It return the row indices corresponding to train and test.
# We set random_state to ensure we get the same splits every time we run this.
kf = KFold(titanic_train.shape[0], n_folds=3, random_state=1)

predictions = []
for train, test in kf:
    # The predictors we're using the train the algorithm.  Note how we only take the rows in the train folds.
    train_predictors = (titanic_train[predictors].iloc[train,:])
    # The target we're using to train the algorithm.
    train_target = titanic_train["Survived"].iloc[train]
    # Training the algorithm using the predictors and target.
    alg.fit(train_predictors, train_target)
    # We can now make predictions on the test fold
    test_predictions = alg.predict(titanic_train[predictors].iloc[test,:])
    predictions.append(test_predictions)
###############################################################################
###############################################################################
#predictions = np.concatenate(predictions, axis=0)

# Map predictions to outcomes (only possible outcomes are 1 and 0)
#print(np.asarray(predictions).shape)
predictionsArray=np.asarray(predictions)
predictionsArray[predictionsArray > .5] = 1
predictionsArray[predictionsArray <=.5] = 0
predictions=np.ndarray.tolist(predictionsArray)

#accuracy=(abs(titanic["Survived"]-predictions)/len(predictions)
#accuracy = sum(predictions[predictions == titanic_train["Survived"]]) / len(predictions)

# Initialize our algorithm
alg = LogisticRegression(random_state=1)
# Compute the accuracy score for all the cross validation folds.  (much simpler than what we did before!)
scores = cross_validation.cross_val_score(alg, titanic_train[predictors], titanic_train["Survived"], cv=3)
# Take the mean of the scores (because we have one for each fold)
print(scores.mean())










