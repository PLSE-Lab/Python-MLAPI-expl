import numpy as np
import pandas as pd
import re
import operator

from sklearn import svm
from sklearn import linear_model
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

titanic = pd.read_csv('../input/train.csv')
titanic_test = pd.read_csv('../input/test.csv')

# Generating a familysize column
titanic["FamilySize"] = titanic["SibSp"] + titanic["Parch"] + 1
titanic_test["FamilySize"] = titanic_test["SibSp"] + titanic_test["Parch"] + 1

#titanic.loc[titanic["FamilySize"] == 1, "FamilySize"] = 0
#titanic.loc[(titanic["FamilySize"] > 1) & (titanic["FamilySize"] < 5), "FamilySize"] = 1
#titanic.loc[titanic["FamilySize"] > 4, "FamilySize"] = 2

# The .apply method generates a new series
titanic["NameLength"] = titanic["Name"].apply(lambda x: len(x))
titanic_test["NameLength"] = titanic_test["Name"].apply(lambda x: len(x))

# A function to get the title from a name.
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""
    
titles = titanic["Name"].apply(get_title)
titles_test = titanic_test["Name"].apply(get_title)
#print(pd.value_counts(titles))

# Map each title to an integer.  Some titles are very rare, and are compressed into the same codes as other titles.
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, 
                "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10,
                "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2, "Dona":10}

#title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 5, 
#                "Major": 5, "Col": 5, "Mlle": 2, "Mme": 3, "Don": 5, "Lady": 5,
#                "Countess": 5, "Jonkheer": 5, "Sir": 5, "Capt": 5, "Ms": 2, "Dona":5}


for k,v in title_mapping.items():
    titles[titles == k] = v
    titles_test[titles_test == k] = v
    
# Verify that we converted everything.
#print(pd.value_counts(titles))

# Add in the title column.
titanic["Title"] = titles
titanic_test["Title"] = titles_test
#=======================================
# A dictionary mapping family name to id
family_id_mapping = {}

# A function to get the id given a row
def get_family_id(row):
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

# Get the family ids with the apply method
family_ids = titanic.apply(get_family_id, axis=1)
family_ids_test = titanic_test.apply(get_family_id, axis=1)

# There are a lot of family ids, so we'll compress all of the families under 3 members into one code.
family_ids[titanic["FamilySize"] < 3] = -1
family_ids_test[titanic_test["FamilySize"] < 3] = -1

# Print the count of each unique id.
#print(pd.value_counts(family_ids))

titanic["FamilyId"] = family_ids
titanic_test["FamilyId"] = family_ids_test

child = titanic["Age"] < 18
child_test = titanic_test["Age"] < 18
titanic["Child"] = child.astype(int)
titanic_test["Child"] = child_test.astype(int)


Mother = ((titanic["Sex"] == 1) & (titanic["Parch"] > 0) & 
            (titanic["Age"] > 18) & (titanic["Title"] != "Miss"))
Mother_test = ((titanic_test["Sex"] == 1) & (titanic_test["Parch"] > 0) & 
            (titanic_test["Age"] > 18) & (titanic_test["Title"] != "Miss"))
titanic["Mother"] = Mother.astype(int)
titanic_test["Mother"] = Mother_test.astype(int)

######################################
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", 
              "FamilySize", "Title", "Child", "Mother"]
                          
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
titanic_test["Age"] = titanic_test["Age"].fillna(titanic["Age"].median())

titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic["Fare"].median())


# Find all the unique genders -- the column appears to contain only male and female.
#print(titanic["Sex"].unique())
# Replace all the occurences of male with the number 0.
titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1

titanic_test.loc[titanic_test["Sex"] == "male", "Sex"] = 0
titanic_test.loc[titanic_test["Sex"] == "female", "Sex"] = 1

#print(titanic["Embarked"].unique())
titanic["Embarked"] = titanic["Embarked"].fillna("C")
titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2

titanic_test["Embarked"] = titanic_test["Embarked"].fillna("C")
titanic_test.loc[titanic_test["Embarked"] == "S", "Embarked"] = 0
titanic_test.loc[titanic_test["Embarked"] == "C", "Embarked"] = 1
titanic_test.loc[titanic_test["Embarked"] == "Q", "Embarked"] = 2
###########################################################

# Perform feature selection
selector = SelectKBest(f_classif, k=5)
selector.fit(titanic[predictors], titanic["Survived"])

# Get the raw p-values for each feature, and transform from p-values into scores
scores = -np.log10(selector.pvalues_)

# Plot the scores.  See how "Pclass", "Sex", "Title", and "Fare" are the best?
#plt.bar(range(len(predictors)), scores)
#plt.xticks(range(len(predictors)), predictors, rotation='vertical')
#plt.show()

###########################################################
# update predictors
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", 
              "FamilySize", "Title", "Child", "Mother"]
              
# Initialize our algorithm
#alg = linear_model.LogisticRegression(random_state=1)
alg = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=8, min_samples_leaf=4)
#alg = GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3)
#alg = svm.SVC(kernel='linear', C=1)
#alg = svm.SVC(kernel='rbf', gamma=0.7, C=1)


# Compute the accuracy score for all the cross validation folds.  (much simpler than what we did before!)
scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=3)
# Take the mean of the scores (because we have one for each fold)
print("CV score is :", scores.mean())

alg.fit(titanic[predictors],titanic["Survived"])
predictions = alg.predict(titanic_test[predictors])


#############################################################
predictors = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]

algorithms = [
    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), predictors],
    [linear_model.LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]]
]

# Initialize the cross validation folds
kf = KFold(titanic.shape[0], n_folds=3, random_state=1)

predictions = []
for train, test in kf:
    train_target = titanic["Survived"].iloc[train]
    full_test_predictions = []
    # Make predictions for each algorithm on each fold
    for alg, predictors in algorithms:
        # Fit the algorithm on the training data.
        alg.fit(titanic[predictors].iloc[train,:], train_target)
        # Select and predict on the test fold.  
        # The .astype(float) is necessary to convert the dataframe to all floats and avoid an sklearn error.
        test_predictions = alg.predict_proba(titanic[predictors].iloc[test,:].astype(float))[:,1]
        full_test_predictions.append(test_predictions)
    # Use a simple ensembling scheme -- just average the predictions to get the final classification.
    test_predictions = (full_test_predictions[0]*3 + full_test_predictions[1]) / 4
    # Any value over .5 is assumed to be a 1 prediction, and below .5 is a 0 prediction.
    test_predictions[test_predictions <= .5] = 0
    test_predictions[test_predictions > .5] = 1
    predictions.append(test_predictions)

# Put all the predictions together into one array.
predictions = np.concatenate(predictions, axis=0)

# Compute accuracy by comparing to the training data.
accuracy = sum(predictions[predictions == titanic["Survived"]]) / len(predictions)
print("ensembling accuracy is: ",accuracy)

#####################################

full_predictions = []
for alg, predictors in algorithms:
    # Fit the algorithm using the full training data.
    alg.fit(titanic[predictors], titanic["Survived"])
    # Predict using the test dataset.  We have to convert all the columns to floats to avoid an error.
    predictions = alg.predict_proba(titanic_test[predictors].astype(float))[:,1]
    full_predictions.append(predictions)

# The gradient boosting classifier generates better predictions, so we weight it higher.
predictions = (full_predictions[0] * 3 + full_predictions[1]) / 4

predictions[predictions <= .5] = 0
predictions[predictions > .5] = 1
predictions = predictions.astype(int)

#############################################
# submission
submission = pd.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": predictions
    })

submission.to_csv('ttest.csv', index=False)