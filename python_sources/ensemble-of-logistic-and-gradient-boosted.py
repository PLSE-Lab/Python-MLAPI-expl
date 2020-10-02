import pandas as pd
import numpy as np
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
import re
import matplotlib.pyplot as plt

def fillMedian(df, fill_df, col_name):
	"""input dataframe with age na's
	return df with filled na's
	"""
	df[col_name] = df[col_name].fillna(fill_df[col_name].median())
	return df


def transformEmbarked(df):
	"""transformed Embarked column to be numeric, and fill na's
	"""
	df['Embarked'] = df['Embarked'].fillna('S')
	df.loc[df['Embarked'] == 'S', 'Embarked'] = 0
	df.loc[df['Embarked'] == 'C', 'Embarked'] = 1
	df.loc[df['Embarked'] == 'Q', 'Embarked'] = 2
	return df


def get_title(name):
    """ A function to get the title from a name.
    Use a regular expression to search for a title.  
    Titles always consist of capital and lowercase letters, and end with a 
    period.
    """
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""


def mapTitles(df):
	"""Extract Titles and convert them to integers
	"""
	#create title feature
	titles = df["Name"].apply(get_title)
	print(pd.value_counts(titles))
	title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, 
					 "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, 
					 "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, 
					 "Capt": 7, "Ms": 2, "Dona": 2}
	for k,v in title_mapping.items():
	    titles[titles == k] = v

	# Verify that we converted everything.
	print(pd.value_counts(titles))
	return titles
#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#Fill missing Age data
train = fillMedian(train, train, "Age")
test = fillMedian(test, train, "Age")

#transform Sex data
train.loc[train['Sex'] == "male", "Sex"] = 0
train.loc[train['Sex'] == "female", "Sex"] = 1
test.loc[test['Sex'] == "male", "Sex"] = 0
test.loc[test['Sex'] == "female", "Sex"] = 1

#transform Ebarked data
train = transformEmbarked(train)
test = transformEmbarked(test)

#fill nas for test Fare data
test = fillMedian(test, train, "Fare")

#create new features
train["FamilySize"] = train["SibSp"] + train["Parch"]
test["FamilySize"] = test["SibSp"] + test["Parch"]
train["NameLength"] = train["Name"].apply(lambda x: len(x))
test["NameLength"] = test["Name"].apply(lambda x: len(x))
train["Title"] = mapTitles(train)
test["Title"] = mapTitles(test)

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", 
			  "FamilySize", "NameLength", "Title"]

# Perform feature selection
selector = SelectKBest(f_classif, k=5)
selector.fit(train[predictors], train["Survived"])

# Get the raw p-values for each feature, and transform from p-values into scores
scores = -np.log10(selector.pvalues_)

# Plot the scores.  See how "Pclass", "Sex", "Title", and "Fare" are the best?
plt.bar(range(len(predictors)), scores)
plt.xticks(range(len(predictors)), predictors, rotation='vertical')
plt.show()

#use top 4
predictors = ["Pclass", "Sex", "Fare", "Title"]

#Ensemble Gradient Boosting and Logistic Regression
algorithms = [
	[GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3),
["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title"]],
	[LogisticRegression(random_state=1), 
["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]]]

kf = cross_validation.KFold(train.shape[0], n_folds=3, random_state=1)

pred = []
full_test_predictions = []
for alg, predictors in algorithms:
	alg.fit(train[predictors], train["Survived"])
	test_predictions = \
		alg.predict_proba(test[predictors].astype(float))[:,1]
	full_test_predictions.append(test_predictions)
test_predictions = (full_test_predictions[0] + full_test_predictions[1]) / 2

test_predictions[test_predictions <= .5] = 0
test_predictions[test_predictions > .5] = 1
pred.append(test_predictions)

pred = np.concatenate(pred, axis=0)

submission = pd.DataFrame({
		"PassengerId": test["PassengerId"], 
		"Survived": pred})

#Any files you save will be available in the output tab below
submission.to_csv('copy_of_the_training_data.csv', index=False)