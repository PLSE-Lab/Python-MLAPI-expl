import numpy as np
import pandas as pd
# Import the linear regression class
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
#from sklearn import cross_validation
# Sklearn also has a helper that makes it easy to do cross validation
#from sklearn.cross_validation import KFold

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#Print to standard output, and see the results in the "log" section below after running your script
'''
print("\n\nTop of the training data:")
print(train.head(5))  # first 5 rows

print("\n\nSummary statistics of training data")
print(train.describe()) 
'''


    
   
############################################### process data ###############################################
def process(dataset):
# replace missing age with median value
    dataset["Age"]  = dataset["Age"].fillna(dataset["Age"].median())
    dataset["Fare"] = dataset["Fare"].fillna(dataset["Fare"].median())
    # Find all the unique genders -- the column appears to contain only male and female.
    #print(train["Sex"].unique())
    # Replace all the occurences of male with the number 0.
    dataset.loc[dataset["Sex"] == "male", "Sex"] = 0
    dataset.loc[dataset["Sex"] == "female", "Sex"] = 1

    dataset["Embarked"] = dataset["Embarked"].fillna("S")
    dataset.loc[dataset["Embarked"] == "S", "Embarked"] = 0
    dataset.loc[dataset["Embarked"] == "C", "Embarked"] = 1
    dataset.loc[dataset["Embarked"] == "Q", "Embarked"] = 2

    dataset["class1"] = dataset["Pclass"]==1
    dataset["class2"] = dataset["Pclass"]==2
    dataset["class3"] = dataset["Pclass"]==3
 
    dataset["age2"] =  dataset["Age"]*dataset["Age"]/100
    return dataset

train_p = process(train)
test_p  = process(test)

############################################### model part ###############################################
# Initialize the algorithm class
alg = LogisticRegression(random_state=1)


# Train the algorithm using all the training data
#predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"] # 0.7512
predictors = ["Pclass", "Sex", "Age", "age2","SibSp", "Parch", "Fare", "Embarked"] # 0.77033
#predictors = ["Pclass", "Sex", "Age", "age2"] 
#predictors = ["class1","class2","class3", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
alg.fit(train[predictors], train_p["Survived"]) #fit(X, y, sample_weight)

############################################### submission ###############################################
# Make predictions using the test set.
predictions = alg.predict(test_p[predictors])

# Create a new dataframe with only the columns Kaggle wants from the dataset.
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predictions
    })

#print(submission.head(5))

submission.to_csv("fork_dataquest_v2.csv", index=False)