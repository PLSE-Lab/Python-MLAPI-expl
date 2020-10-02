# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

#Titanic practice


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#from sklearn.linear_model import LogisticRegression
#from sklearn.svm import SVC, LinearSVC
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.naive_bayes import GaussianNB
#from sklearn.linear_model import Perceptron
#from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier




pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

train = pd.read_csv("../input/titanic/train.csv")
test = pd.read_csv("../input/titanic/test.csv")


train["Age"] = train["Age"].fillna(train["Age"].median())
test["Age"] = test["Age"].fillna(test["Age"].median())
train["Embarked"] = train["Embarked"].fillna("S")
train = train.replace({'female': 0, 'male': 1})
test = test.replace({'female': 0, 'male': 1})
test["Fare"] = test["Fare"].fillna(test["Fare"].median())

train.to_csv("train.csv", sep=",")
test.to_csv("test.csv", sep=",")



X_train = train[["Pclass", "Sex", "Age", "Fare"]].values
Y_train = train["Survived"].values
X_test = test[["Pclass", "Sex", "Age", "Fare"]].values






#DecisionTreeClassifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
print("DecisionTreeClassifier = ", acc_decision_tree)

result = decision_tree.predict(X_test)
submission = pd.DataFrame({"PassengerId":test["PassengerId"], "Survived":result})
submission.to_csv("submission.csv", index=False)
