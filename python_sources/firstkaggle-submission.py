import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation

# Initialize our algorithm
alg = LogisticRegression(random_state=1)

# Print you can execute arbitrary python code
#input the train and test datasets by using pd.read() to load the .csv files in to data frame
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

# Fix train data by filling the missing values in column age with the median of all the values in age
train["Age"] = train["Age"].fillna(train["Age"].median())

#Perform data transformation by transforming sex = male as 0 amd sex = female as 1.
train.loc[train["Sex"] == "male", "Sex"] = 0
train.loc[train["Sex"] == "female", "Sex"] = 1

#Perform Data cleaning on Embarked column by substituting all the missing values in the column with the most frequently occuring value "S"
train["Embarked"] = train["Embarked"].fillna("S")
#Transform Embarked data where Embarked ="S" is tranformed to 0 ,Embarked ="C" is tranformed to 1 and Embarked ="Q" is tranformed to 2
train.loc[train["Embarked"] == "S", "Embarked"] = 0
train.loc[train["Embarked"] == "C", "Embarked"] = 1
train.loc[train["Embarked"] == "Q", "Embarked"] = 2

#Perfom Data cleaning operations on the column fare by replacing all the missing values in the column with the median of all the values in fare.
train["Fare"] = train["Fare"].fillna(train["Fare"].median())

# Similarly perform the same operations on test data
test["Age"] = test["Age"].fillna(train["Age"].median())

test.loc[test["Sex"] == "male", "Sex"] = 0
test.loc[test["Sex"] == "female", "Sex"] = 1

test["Embarked"] = test["Embarked"].fillna("S")
test.loc[test["Embarked"] == "S", "Embarked"] = 0
test.loc[test["Embarked"] == "C", "Embarked"] = 1
test.loc[test["Embarked"] == "Q", "Embarked"] = 2

test["Fare"] = test["Fare"].fillna(train["Fare"].median())

# Predictors
predictors = ["Pclass", "Sex", "Age", "SibSp"]
print(predictors)
# Train the algorithm using all the training data

alg.fit(train[predictors], train["Survived"])
print(train["Survived"])
# Make predictions using the test set.
#output predictions on the test set.
#To predict class labels for data in test[predictors]
predictions = alg.predict(test[predictors])
print(predictions)  

# Create a new dataframe with only the columns Kaggle wants from the dataset.
#The PassengerID values are taken from the test set
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predictions
    })

# Any files you save will be available in the output tab below
submission.to_csv('submission.csv', index=False)