import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing

#Print you can execute arbitrary python code

train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

np.random.seed(12)

new_ages = np.where(train.Age.isnull(), 30, train.Age)

train.Age = new_ages

train["Q"] = np.where(train.Embarked == "Q", 1, -1)
train["C"] = np.where(train.Embarked == "C", 1, -1)
train["S"] = np.where(train.Embarked == "S", 1, -1)

new_sex = np.where(train.Sex == "male", 1, -1)
train.Sex = new_sex

features = ["scaled_id","Sex","Pclass","SibSp","Age","Fare","Q","C","S"]

scaled_fare = preprocessing.scale(train["Fare"])
train["Fare"] = scaled_fare

scaled_age = preprocessing.scale(train["Age"])
train["Age"] = scaled_age

train["scaled_id"] = preprocessing.scale(train["PassengerId"])


Y  = train.Survived
X = train[features]





#clf = RandomForestClassifier(n_estimators=1000,oob_score=True,min_samples_split = 4,max_features = 4)
clf = svm.SVC(kernel='rbf')
clf = clf.fit(X, Y)

new_ages =  np.where(test.Age.isnull(), 30, test.Age)
test.Age = new_ages

test["Q"] = np.where(test.Embarked == "Q", 1, 0)
test["C"] = np.where(test.Embarked == "C", 1, 0)
test["S"] = np.where(test.Embarked == "S", 1, 0)

new_sex = np.where(test.Sex == "male", 1, 0)
test.Sex = new_sex

new_fare = np.where(np.isnan(test.Fare), 31, test.Fare)
test.Fare=  new_fare

scaled_fare = preprocessing.scale(test["Fare"])
test["Fare"] = scaled_fare

scaled_age = preprocessing.scale(test["Age"])
test["Age"] = scaled_age

test["scaled_id"]  = preprocessing.scale(test["PassengerId"])


test_preds = clf.predict(X = test[features])
submission = pd.DataFrame({"PassengerId":test["PassengerId"],
                           "Survived":test_preds})

#Print to standard output, and see the results in the "log" section below after running your script

loc1 = 0
loc2 = 0
loc3 = 0
tot = 0
for a in range(len(train)):
    if train["Survived"][a] == 0:
        tot += 1
        if train["Embarked"][a] == "S":
            loc1 += 1
        elif train["Embarked"][a] == "C":
            loc2 += 1
        elif train["Embarked"][a] == "Q":
            loc3 += 1

print("S: " + str(loc1/ tot))
print("C: " + str(loc2/ tot))
print("Q: " + str(loc3/ tot))

print("\n\nTop of the training data:")
print(train.head())

print("\n\nSummary statistics of training data")
print(train.describe())



submission.to_csv("tutorial_randomForest_submission.csv", 
                  index=False) 

#Any files you save will be available in the output tab below
#train.to_csv('copy_of_the_training_data.csv', index=False)