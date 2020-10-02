import numpy as np
import pandas as pd

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

#Print to standard output, and see the results in the "log" section below after running your script
print("\n\nTop of the training data:")
print(train.head())

print("\n\nSummary statistics of training data")
print(train.describe())

#Any files you save will be available in the output tab below
train.to_csv('copy_of_the_training_data.csv', index=False)

train["Age"] = train["Age"].fillna(train["Age"].median())

print(train["Sex"].unique())
train.loc[train["Sex"] == "male", "Sex"] = 0
train.loc[train["Sex"] == "female", "Sex"] = 1

print(train["Embarked"].unique())
train["Embarked"] = train["Embarked"].fillna("S")
train.loc[train["Embarked"] == "S", "Embarked"] = 0
train.loc[train["Embarked"] == "C", "Embarked"] = 1
train.loc[train["Embarked"] == "Q", "Embarked"] = 2
print(train["Embarked"].unique())

# Import the linear regression class
from sklearn.linear_model import LinearRegression
# Sklearn also has a helper that makes it easy to do cross validation
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression




# The columns we'll use to predict the target
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

# Initialize our algorithm class
alg = LinearRegression()
# Generate cross validation folds for the titanic dataset.  It return the row indices corresponding to train and test.
# We set random_state to ensure we get the same splits every time we run this.
kf = KFold(n_splits=3)
kf

predictions = []
print(kf.get_n_splits(train))
for train1, test1 in kf.split(train):
    # The predictors we're using the train the algorithm.  Note how we only take the rows in the train folds.
    train_predictors = (train[predictors].iloc[train1,:])
    # The target we're using to train the algorithm.
    train_target = train["Survived"].iloc[train1]
    # Training the algorithm using the predictors and target.
    alg.fit(train_predictors, train_target)
    # We can now make predictions on the test fold
    test_predictions = alg.predict(train[predictors].iloc[test1,:])
    predictions.append(test_predictions)

predictions = np.concatenate(predictions, axis=0)

# Map predictions to outcomes (only possible outcomes are 1 and 0)
predictions[predictions > .5] = 1
predictions[predictions <=.5] = 0

accuracy = sum(predictions[predictions == train["Survived"]]) / len(predictions)    
print(accuracy)

test["Age"] = test["Age"].fillna(test["Age"].median())

print(test["Sex"].unique())
test.loc[test["Sex"] == "male", "Sex"] = 0
test.loc[test["Sex"] == "female", "Sex"] = 1

print(test["Embarked"].unique())
test["Embarked"] = test["Embarked"].fillna("S")
test.loc[test["Embarked"] == "S", "Embarked"] = 0
test.loc[test["Embarked"] == "C", "Embarked"] = 1
test.loc[test["Embarked"] == "Q", "Embarked"] = 2
print(test["Embarked"].unique())
test["Fare"] = test["Fare"].fillna(train["Fare"].median())


# Import the linear regression class
#from sklearn.linear_model import LinearRegression
# Sklearn also has a helper that makes it easy to do cross validation
#from sklearn.model_selection import KFold



# The columns we'll use to predict the target

# Initialize our algorithm class
alg = LogisticRegression(random_state=1)
# Generate cross validation folds for the titanic dataset.  It return the row indices corresponding to train and test.
# We set random_state to ensure we get the same splits every time we run this.

alg.fit(train[predictors], train["Survived"])

# Make predictions using the test set.
predictions = alg.predict(test[predictors])
predictions = predictions.astype(int)


# Create a new dataframe with only the columns Kaggle wants from the dataset.
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predictions
    })
submission.to_csv("kaggle.csv", index=False)    