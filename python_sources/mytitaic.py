import numpy as np
import pandas as pd

train = pd.read_csv("../input/train.csv", )
test  = pd.read_csv("../input/test.csv", )

def fix_data(titanic):
    titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
    titanic["Age"].median()

    titanic["Embarked"] = titanic["Embarked"].fillna("S")

    titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
    titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
    titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2

    titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
    titanic.loc[titanic["Sex"] == "female", "Sex"] = 1

    titanic.loc[titanic["Name"].str.find("Sir.") != -1, "NameClass"] = 1
    titanic.loc[titanic["Name"].str.find("Sir.") == -1, "NameClass"] = 0

    return titanic

train_data = fix_data(train)
test_data  = fix_data(test)
features = ["Pclass", "Sex", "Age", "NameClass"]

X =  train_data[features]
y = train_data.Survived

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)

from sklearn.linear_model import LogisticRegression
from sklearn import tree
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3,min_samples_leaf=5)
clf = clf.fit(X,y)
print("{:.2f}".format(clf.score(X_test,y_test)))

clf    = LogisticRegression(random_state=3)
clf.fit(X,y)
print("{:.2f}".format(clf.score(X_test,y_test)))
#0.83

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(
    random_state=1,
    n_estimators=150,
    min_samples_split=4,
    min_samples_leaf=2
)
clf = RandomForestClassifier(n_estimators=1000,random_state=33)
clf.fit(X,y)
print("{:.2f}".format(clf.score(X_test,y_test)))

predictions = clf.predict(test_data[features])
submission = pd.DataFrame({
        "PassengerId": test_data["PassengerId"],
        "Survived": predictions
    })

submission.to_csv("result.csv", index=False)