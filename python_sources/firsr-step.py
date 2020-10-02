"""
Author: Sean Walker
Date: 5 October 2016
Target: Read csv file in python
"""

import pandas as pd

from sklearn import ensemble
from sklearn import svm
from sklearn.neural_network import MLPClassifier

# always use header=0 to read header of csv files
train = pd.read_csv('../input/train.csv', header=0)
test = pd.read_csv('../input/test.csv', header=0)

"""after using train.describe(), 
we can known some details about the data,
so i decided to make a func to clean the data 
"""
def data_clean(data):
    data['Age'] = data['Age'].fillna(data['Age'].median())
    data['Gender'] = data['Sex'].map({'female':0, 'male':1}).astype(int)
    data['Family'] = data['Parch'] + data['SibSp']
    data['Fare'] = data['Fare'].fillna(data['Fare'].mean())
    data = data.drop(['SibSp','Parch','Sex','Name','Cabin','Embarked','Ticket'],axis=1)
    return data

train_data = data_clean(train)
test_data = data_clean(test)
X = train_data.drop(['PassengerId','Survived'], axis=1)
y = train_data['Survived']
X_test = test_data.drop(["PassengerId"],axis=1)
	
def pred_with_randomforest():
	random_forest = ensemble.RandomForestClassifier(n_estimators=100)
	random_forest.fit(X,y)
	return random_forest.score(X,y)

def pred_with_svm():
	kernel_svm = svm.SVC(gamma=.1)
	linear_svm = svm.LinearSVC()
	
	kernel_svm.fit(X,y)

	return 	kernel_svm.score(X,y)

score_random_forest = pred_with_randomforest()
score_kernel_svm = pred_with_svm()

print "score_random_forest: ", score_random_forest
print "score_kernel_svm: ", score_kernel_svm

