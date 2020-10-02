# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing,model_selection
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory are saved as output.

columns = ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]

dataframe = pd.read_csv("../input/train.csv")
survived = dataframe["Survived"].values
features = dataframe[columns]#.values
encoder = preprocessing.LabelEncoder()
features["Sex"] = encoder.fit_transform(features["Sex"].fillna('0'))
features["Embarked"] = encoder.fit_transform(features["Embarked"].fillna('0'))
features = features.fillna(0).values
X_train, X_test, y_train, y_test = model_selection.train_test_split(features,survived,test_size=0.6)

clf = GaussianNB()
clf.fit(X_train, y_train)
print (clf.score(X_train,y_train))
print (clf.score(X_test,y_test))

test_dataframe = pd.read_csv("../input/test.csv")
test_dataframe = test_dataframe[columns]
test_dataframe["Sex"] = encoder.fit_transform(test_dataframe["Sex"])
test_dataframe["Embarked"] = encoder.fit_transform(test_dataframe["Embarked"].fillna('0'))
test_dataframe = test_dataframe.fillna(0).values

out = clf.predict(test_dataframe)
print (out)
print (clf.score(test_dataframe,out))