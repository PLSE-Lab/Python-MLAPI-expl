import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

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

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#Print to standard output, and see the results in the "log" section below after running your script
print("\n\nTop of the training data:")
print(train.head())

print("\n\nSummary statistics of training data")
print(train.describe())

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

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

#initialize algo
alg = LogisticRegression(random_state=1)
#fit algo
alg.fit(train[predictors], train["Survived"])
#make predictions
pred = alg.predict(test[predictors])

submission = pd.DataFrame({
		"PassengerId": test["PassengerId"], 
		"Survived": pred})

#Any files you save will be available in the output tab below
submission.to_csv('copy_of_the_training_data.csv', index=False)