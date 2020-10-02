import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test_sample = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#Print to standard output, and see the results in the "log" section below after running your script
print("\n\nTop of the training data:")
print(train.head())

print("\n\nSummary statistics of training data")
print(train.describe())

train["Age"]=train["Age"].fillna(train["Age"].median())

train.loc[train["Sex"] == "male", "Sex"] = 0
train.loc[train["Sex"] == "female", "Sex"] = 1

train["Embarked"]=train["Embarked"].fillna("S")

train.loc[train["Embarked"] == "S", "Embarked"] = 0
train.loc[train["Embarked"] == "C", "Embarked"] = 1
train.loc[train["Embarked"] == "Q", "Embarked"] = 2

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

alg = LinearRegression()

kf = KFold(train.shape[0], n_folds=3, random_state=1)

predictions = []
for training, test in kf:
    
    train_predictors = (train[predictors].iloc[training,:])

    train_target = train["Survived"].iloc[training]

    alg.fit(train_predictors, train_target)

    test_predictions = alg.predict(train[predictors].iloc[test,:])
    predictions.append(test_predictions)
    
predictions = np.concatenate(predictions, axis=0)

predictions[predictions > .5] = 1
predictions[predictions <=.5] = 0

accuracy = sum(predictions[predictions == train["Survived"]]) / len(predictions)

print(accuracy)

alg = LogisticRegression(random_state=1)

scores = cross_validation.cross_val_score(alg, train[predictors], train["Survived"], cv=3)

print(scores.mean())


test_sample["Age"]=test_sample["Age"].fillna(test_sample["Age"].median())
test_sample["Fare"]=test_sample["Fare"].fillna(test_sample["Fare"].median())

test_sample.loc[test_sample["Sex"] == "male", "Sex"] = 0
test_sample.loc[test_sample["Sex"] == "female", "Sex"] = 1

test_sample["Embarked"]=test_sample["Embarked"].fillna("S")

test_sample.loc[test_sample["Embarked"] == "S", "Embarked"] = 0
test_sample.loc[test_sample["Embarked"] == "C", "Embarked"] = 1
test_sample.loc[test_sample["Embarked"] == "Q", "Embarked"] = 2


alg = LogisticRegression(random_state=1)

alg.fit(train[predictors], train["Survived"])

predictions = alg.predict(test_sample[predictors])

algorithms = [
    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), predictors],
    [LogisticRegression(random_state=1), predictors]]

full_predictions = []
for alg, predictors in algorithms:
    # Fit the algorithm using the full training data.
    alg.fit(train[predictors], train["Survived"])
    # Predict using the test dataset.  We have to convert all the columns to floats to avoid an error.
    predictions = alg.predict_proba(test_sample[predictors].astype(float))[:,1]
    full_predictions.append(predictions)

# The gradient boosting classifier generates better predictions, so we weight it higher.
predictions3 = (full_predictions[0] * 3 + full_predictions[1]) / 4

predictions3[predictions3 > 0.5] = 1
predictions3[predictions3 <= 0.5] = 0

predictions3 = predictions3.astype(int)

submission = pd.DataFrame({
        "PassengerId": test_sample["PassengerId"],
        "Survived": predictions3
    })
    
submission.to_csv("kaggle.csv", index=False)