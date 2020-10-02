import numpy as np
import pandas as pd

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#Print to standard output, and see the results in the "log" section below after running your script
print("\n\nTop of the training data:")
print(train.head())

print("\n\nSummary statistics of training data")
print(train.describe())

#Any files you save will be available in the output tab below
train.to_csv('copy_of_the_training_data.csv', index=False)

#Script
#titanic_test = pd.read_csv("../input/titanic_test.csv")
titanic_test = test


titanic_test["Age"] = titanic_test["Age"].fillna(titanic_test["Age"].median())

titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic_test["Fare"].median())

titanic_test.loc[titanic_test["Sex"] == "male", "Sex"] = 0

titanic_test.loc[titanic_test["Sex"] == "female", "Sex"] = 1

titanic_test["Embarked"] = titanic_test["Embarked"].fillna("S")


titanic_test.loc[titanic_test["Embarked"] == "S", "Embarked"] = 0

titanic_test.loc[titanic_test["Embarked"] == "C", "Embarked"] = 1

titanic_test.loc[titanic_test["Embarked"] == "Q", "Embarked"] = 2
