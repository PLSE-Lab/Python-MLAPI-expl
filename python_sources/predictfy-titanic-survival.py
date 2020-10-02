import numpy as np
import pandas as pd
# Import the linear regression class
from sklearn.linear_model import LinearRegression
# Import the logistic regression class
from sklearn.linear_model import LogisticRegression
# Sklearn also has a helper that makes it easy to do cross validation
from sklearn.cross_validation import KFold
# Sklearn cross validation
from sklearn import cross_validation

#Print you can execute arbitrary python code
t_train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
t_test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#Print to standard output, and see the results in the "log" section below after running your script

#Fixing column Averages FILL NA
t_train["Age"] = t_train["Age"].fillna(t_train["Age"].median())
t_train["Embarked"] = t_train["Embarked"].fillna("S")

print("\n\nSummary statistics of FIXED training data")
print(t_train.describe())

#Converting String columns into numeric columns
t_train.loc[t_train["Sex"] == "male", "Sex"] = 0
t_train.loc[t_train["Sex"] == "female", "Sex"] = 1
t_train.loc[t_train["Embarked"] == "S", "Embarked"] = 0
t_train.loc[t_train["Embarked"] == "C", "Embarked"] = 1
t_train.loc[t_train["Embarked"] == "Q", "Embarked"] = 2

print("\n\nTop of the training data:")
print(t_train.head())

# The columns we'll use to predict the target
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

# Initialize our algorithm class
alg = LinearRegression()
# Generate cross validation folds for the titanic dataset.  It return the row indices corresponding to train and test.
# We set random_state to ensure we get the same splits every time we run this.
kf = KFold(t_train.shape[0], n_folds=3, random_state=1)

predictions = []
for train, test in kf:
    # The predictors we're using the train the algorithm.  
    # Note how we only take the rows in the train folds.
    train_predictors = (t_train[predictors].iloc[train,:])
    # The target we're using to train the algorithm.
    train_target = t_train["Survived"].iloc[train]
    # Training the algorithm using the predictors and target.
    alg.fit(train_predictors, train_target)
    # We can now make predictions on the test fold
    test_predictions = alg.predict(t_train[predictors].iloc[test,:])
    predictions.append(test_predictions)

#Bring all kFolds back to an array
predictions = np.concatenate(predictions, axis=0)

#Convert values to 1 or 0
predictions[predictions > .5] = 1
predictions[predictions <= .5] = 0

#Calculate the accuracy
accuracy = sum(predictions[predictions == t_train["Survived"]]) / len(predictions)

print(accuracy)

# Initialize our algorithm
alg = LogisticRegression(random_state=1)
# Compute the accuracy score for all the cross validation folds.  (much simpler than what we did before!)
scores = cross_validation.cross_val_score(alg, t_train[predictors], t_train["Survived"], cv=3)
# Take the mean of the scores (because we have one for each fold)
print(scores.mean())


# Time to prodict under the test set
t_test["Age"] = t_test["Age"].fillna(t_test["Age"].median())
t_test["Fare"] = t_test["Fare"].fillna(t_test["Fare"].median())

t_test.loc[t_test["Sex"] == "male", "Sex"] = 0 
t_test.loc[t_test["Sex"] == "female", "Sex"] = 1

t_test["Embarked"] = t_test["Embarked"].fillna("S")
t_test.loc[t_test["Embarked"] == "S", "Embarked"] = 0
t_test.loc[t_test["Embarked"] == "C", "Embarked"] = 1
t_test.loc[t_test["Embarked"] == "Q", "Embarked"] = 2

# Initialize the algorithm class
alg = LogisticRegression(random_state=1)

# Train the algorithm using all the training data
alg.fit(t_train[predictors], t_train["Survived"])

# Make predictions using the test set.
predictions = alg.predict(t_test[predictors])

# Create a new dataframe with only the columns Kaggle wants from the dataset.
submission = pd.DataFrame({
        "PassengerId": t_test["PassengerId"],
        "Survived": predictions
    })

submission.to_csv("kaggle.csv", index=False)

##Store the Survived