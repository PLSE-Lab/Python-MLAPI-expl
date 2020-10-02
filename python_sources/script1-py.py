import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import KFold

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#Print to standard output, and see the results in the "log" section below after running your script
print("\n\nTop of the training data:")
print(train.head())

print("\n\nSummary statistics of training data")
print(train.describe())

train["Age"] = train["Age"].fillna(train["Age"].median())
train["Fare"] = train["Fare"].fillna(train["Fare"].median())
train.loc[train["Sex"] == "male", "Sex"] = 0 
train.loc[train["Sex"] == "female", "Sex"] = 1
train["Embarked"] = train["Embarked"].fillna("S")

train.loc[train["Embarked"] == "S", "Embarked"] = 0
train.loc[train["Embarked"] == "C", "Embarked"] = 1
train.loc[train["Embarked"] == "Q", "Embarked"] = 2

test["Age"] = test["Age"].fillna(train["Age"].median())
test["Fare"] = test["Fare"].fillna(test["Fare"].median())
test.loc[test["Sex"] == "male", "Sex"] = 0 
test.loc[test["Sex"] == "female", "Sex"] = 1
test["Embarked"] = test["Embarked"].fillna("S")

test.loc[test["Embarked"] == "S", "Embarked"] = 0
test.loc[test["Embarked"] == "C", "Embarked"] = 1
test.loc[test["Embarked"] == "Q", "Embarked"] = 2



#Any files you save will be available in the output tab below
train.to_csv('copy_of_the_training_data.csv', index=False)

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

# Initialize our algorithm class
alg = LogisticRegression(random_state=1)
# Generate cross validation folds for the titanic dataset.  It return the row indices corresponding to train and test.
# We set random_state to ensure we get the same splits every time we run this.
kf = KFold(train.shape[0], n_folds=3, random_state=1)

predictions = []
for train1, test1 in kf:
    # The predictors we're using the train the algorithm.  Note how we only take the rows in the train folds.
    train_predictors = (train[predictors].iloc[train1,:])
    # The target we're using to train the algorithm.
    train_target = train["Survived"].iloc[train1]
    # Training the algorithm using the predictors and target.
    alg.fit(train_predictors, train_target)
    # We can now make predictions on the test fold
    predictions = alg.predict(test[predictors])

submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predictions
    })
    
submission.to_csv('op.csv', index=False)    