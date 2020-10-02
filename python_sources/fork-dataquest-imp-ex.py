'''
With random forests, we build hundreds of trees with slightly randomized input data, 
and slightly randomized split points. 
Each tree in a random forest gets a random subset of the overall training data. 
Each split point in each tree is performed on a random subset of the potential columns to split on. 
By averaging the predictions of all the trees, we get a stronger overall prediction and minimize overfitting.
'''

import numpy as np
import pandas as pd
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )


############################################
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
# Initialize our algorithm with the default paramters
#   n_estimators is the number of trees we want to make
#   min_samples_split is the minimum number of rows we need to make a split
#   min_samples_leaf  is the minimum number of samples we can have at the bottom points of the tree)
#   Note: increasing min_samples_split and min_samples_leaf can reduce overfitting

########################################### process data ###########################################
import re
import operator

# A function to get the title from a name.
def get_title(name):
    # Use a regular expression to search for a title.  Titles always consist of capital and lowercase letters, and end with a period.
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

# A function to get the id given a row (regulary experssion)
def get_family_id(row):
    family_id_mapping = {}
    # Find the last name by splitting on a comma
    last_name = row["Name"].split(",")[0]
    # Create the family id
    family_id = "{0}{1}".format(last_name, row["FamilySize"])
    # Look up the id in the mapping
    if family_id not in family_id_mapping:
        if len(family_id_mapping) == 0:
            current_id = 1
        else:
            # Get the maximum id from the mapping and add one to it if we don't have an id
            current_id = (max(family_id_mapping.items(), key=operator.itemgetter(1))[1] + 1)
        family_id_mapping[family_id] = current_id
    return family_id_mapping[family_id]


# A function to process train & test data
def process(dataset):
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
    
    # Generating a familysize column
    dataset["FamilySize"] = dataset["SibSp"] + dataset["Parch"]

    # The .apply method generates a new series
    dataset["NameLength"] = dataset["Name"].apply(lambda x: len(x))

    # Get all the titles and print how often each one occurs.
    titles = dataset["Name"].apply(get_title)
    # Map each title to an integer.  Some titles are very rare, and are compressed into the same codes as other titles.
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9,"Dona": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}
    for tilname,v in title_mapping.items():
        titles[titles == tilname] = v
    #print(pd.value_counts(titles)) # Verify that we converted everything.
    # Add in the title column.
    dataset["Title"] = titles
    
    ##--------- Family size ---------##
    # A dictionary mapping family name to id
    # Get the family ids with the apply method
    family_ids = dataset.apply(get_family_id, axis=1)
    # There are a lot of family ids, so we'll compress all of the families under 3 members into one code.
    family_ids[dataset["FamilySize"] < 3] = -1
    # Print the count of each unique id.
    #print(pd.value_counts(family_ids))
    dataset["FamilyId"] = family_ids

    return(dataset)

train_p = process(train)
test_p  = process(test) 




########################################### prediction
predictors = ["Pclass", "Sex", "Age", "age2", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]
predictors2= ["Pclass", "Sex", "Age", "age2", "Fare", "Embarked", "FamilySize", "Title"]

algorithms = [
    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), predictors],
    [LogisticRegression(random_state=1), predictors2]
]

full_predictions = []
for alg, predictors in algorithms:
    # Fit the algorithm using the full training data.
    alg.fit(train_p[predictors], train_p["Survived"])
    print(train_p["Pclass"].unique())
    # Predict using the test dataset.  We have to convert all the columns to floats to avoid an error.
    predictions = alg.predict_proba(test_p[predictors].astype(float))[:,1]
    full_predictions.append(predictions)

# The gradient boosting classifier generates better predictions, so we weight it higher.
predictions = (full_predictions[0] * 3 + full_predictions[1]) / 4
predictions[predictions <= .5] = 0
predictions[predictions > .5] = 1
predictions = predictions.astype(int)
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predictions
    })
    
submission.to_csv("kaggle.csv", index=False)

###########################################
###########################################