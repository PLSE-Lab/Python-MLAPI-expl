import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neural_network import MLPClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# loading and cleaning training data
data = pd.read_csv('/kaggle/input/train.csv')
Y = data['Survived']
X = data
X.drop(['PassengerId','Survived','Name','Ticket','Embarked','Cabin'], axis = 1, inplace = True)
X['Sex'].replace(to_replace=['male'], value=float(1), inplace=True)
X['Sex'].replace(to_replace=['female'], value=float(2), inplace=True)
X['Age'].fillna(value = int(X['Age'].mean()), inplace=True)

# loading and cleaning test data
test = pd.read_csv('/kaggle/input/test.csv')
test_copy = test.copy()
test.drop(['PassengerId','Name','Ticket','Embarked','Cabin'], axis = 1, inplace = True)
test['Sex'].replace(to_replace=['male'], value=float(1), inplace=True)
test['Sex'].replace(to_replace=['female'], value=float(2), inplace=True)
test['Age'].fillna(value = int(test['Age'].mean()), inplace=True)
test['Fare'].fillna(value = int(test['Fare'].mean()), inplace=True)

# Logistic Regression 
# lr_model = LogisticRegression()
# lr_model.fit(X, Y)
# print('Logistic regression: ',lr_model.score(X, Y))
# lr_result = lr_model.predict(test)

# # SVM
# svm_model = svm.SVC()
# svm_model.fit(X, Y)
# print('SVM: ',svm_model.score(X, Y))
# svm_result = svm_model.predict(test)

# # NN - time intensive
# nn_model = MLPClassifier()
# nn_model.fit(X, Y)
# print("Neural Networks: ",nn_model.score(X, Y))
# nn_result = nn_model.predict(test)

# # Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X, Y)
# print('Random Forest: ',rf_model.score(X, Y))
rf_result = rf_model.predict(test)

# Decision Trees
# dt_model = DecisionTreeClassifier()
# dt_model.fit(X, Y)
# # print('Decision Tree: ',dt_model.score(X, Y))
# dt_result = dt_model.predict(test)

# creating submission.csv
submission = pd.DataFrame({'PassengerId': test_copy.PassengerId, 'Survived': rf_result})
submission.to_csv('submission.csv', index=False)