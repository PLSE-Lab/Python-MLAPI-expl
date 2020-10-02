import numpy as np
import pandas as pd
import random
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#Print to standard output, and see the results in the "log" section below after running your script
print("\n\nTop of the training data:")
print(train.head())

print("\n\nSummary statistics of training data")
print(train.describe())

# Split train set to X(features), Y(labels)
print("\n\nTrain set size: %s" % (train.shape,))
train_fixed = train.dropna()
X = train_fixed.loc[:, ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Cabin", "Embarked"]]
Y = train_fixed.loc[:,"Survived"]
print("\n\nTop of features:")
print(X.head())
print("\n\nTop of labels(1 Survived, 0 Not)")
print(Y.head())

# Data Clean
## Drop NaN data
X = X.drop("Cabin", 1)

# For Sex, convert male ->1, female -> 0
sex_map = {"male":1, "female":0}
X.loc[:, "Sex"] = X.loc[:, "Sex"].map(sex_map)

## Fix missing data
print("\n\nFind which columns contains NaN value:")
for i in range(X.columns.size):
    if X.iloc[:,i].isnull().sum() > 0 :
        print("%s :%d Nan" % (X.columns[i], X.iloc[:,i].isnull().sum()))
        
## For cabin and embarked use probability generate missing data
embarked_map = lambda e: ord(e)-ord("A")
X.loc[:,"Embarked"] = X.loc[:, "Embarked"].map(embarked_map)

# Look at features now
print("\n\nTop of features:")
print(X.head())

# Train classifier
clf = RandomForestClassifier(n_estimators=int(X.shape[1] / 2))
scores = cross_validation.cross_val_score(clf, X, Y, cv=5)
print("RandomForese Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
clf = clf.fit(X, Y)

# Features selecting
print("\n\nRandom forest feature importances: ")
for i in range(X.shape[1]):
    print("%s: %s" % (X.columns[i], clf.feature_importances_[i]))   

# Process test data
#Print to standard output, and see the results in the "log" section below after running your script
print("\n\nTop of the test data:")
print(test.head())

print("\n\nSummary statistics of test data")
print(test.describe())

to_predict = test.loc[:, ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
## Fix missing data
print("\n\nFind which columns in test contains NaN value:")
for i in range(to_predict.columns.size):
    if to_predict.iloc[:,i].isnull().sum() > 0 :
        print("%s :%d Nan" % (to_predict.columns[i], to_predict.iloc[:,i].isnull().sum()))
## For age use mean as missing ones
age_mean = to_predict["Age"].mean()
age_std = to_predict["Age"].std()
for index, flag in enumerate(to_predict["Age"].isnull()):
    if (flag):
        to_predict["Age"][index] = random.normalvariate(age_mean, age_std)
print("\nAfter fix age, missing number in age is %s" % (to_predict["Age"].isnull().sum()))
#to_predict.loc[:,"Age"] = to_predict.loc[:, "Age"].fillna(to_predict.loc[:, "Age"].mean())
to_predict.loc[:,"Fare"] = to_predict.loc[:, "Fare"].fillna(to_predict.loc[:, "Fare"].mean())
to_predict.loc[:,"Embarked"] = to_predict.loc[:, "Embarked"].map(embarked_map)
to_predict.loc[:, "Sex"] = to_predict.loc[:, "Sex"].map(sex_map)

# predict
result = clf.predict(to_predict)
prediction = pd.DataFrame({"PassengerId": test.loc[:,"PassengerId"], "Survived": result})
prediction.to_csv('result.csv', index=False)
print("\n\nCongratulations! Predict success")
