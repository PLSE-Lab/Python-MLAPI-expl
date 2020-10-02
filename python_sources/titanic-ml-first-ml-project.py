# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from math import modf
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix  
import os
print(os.listdir("../input/titanic"))
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic-test/test.csv')
print(train.shape)
print(test.shape)
print(train.columns)

test_X = test[['Sex','Age','Pclass','Embarked','Ticket','SibSp','Parch']]
test_X.Sex[test_X.Sex =='female'] = 1
test_X.Sex[test_X.Sex == 'male'] = 0
test_X['Age'] = test_X['Age'].fillna(modf(test_X['Age'][test_X['Age'].notna()].mean())[1])
test_X.Embarked = test_X.Embarked.fillna(test_X.Embarked.dropna().mode()[0])
test_X.Embarked[test_X.Embarked == 'S'] = 0
test_X.Embarked[test_X.Embarked == 'C'] = 1
test_X.Embarked[test_X.Embarked == 'Q'] = 2
test_X['Ticket_len'] = test_X.Ticket.apply(len)
test_X['Family'] = test_X.SibSp + test_X.Parch
test_X['Family'] = test_X['Family'].replace([0],'Alone')
test_X['Family'] = test_X['Family'].replace([1,2,3],'Little Family')
test_X['Family'] = test_X['Family'].replace([4,5,6,7,8,9,10],'Big Family')
test_X['Family'][test_X['Family'] == 'Alone'] = 0
test_X['Family'][test_X['Family'] == 'Little Family'] = 1
test_X['Family'][test_X['Family'] == 'Big Family'] = 2
#test_X['Ticket_letter'] = test_X.Ticket.str[0]
test_X = test_X.drop('Ticket',1)


X = train[['Sex','Age','Pclass','Embarked','Ticket','SibSp','Parch']]
y = train[['Survived']]
age_groups = {0: (0,15) , 1: (15,25), 2: (25,40), 3: (40,65), 4: (65,81)}
print(X.shape)
print(y.shape)
X['Age'] = X['Age'].fillna(modf(X['Age'][X['Age'].notna()].mean())[1])
#print(X[-10:])
X.Sex[X.Sex =='female'] = 1
X.Sex[X.Sex == 'male'] = 0
print(X)
#print(X[X.columns].mean())
#X = X.fillna(X[X.columns].mean())
#print(X.info())
#plt.figure()
#pd.plotting.scatter_matrix(X,figsize=(15,15), grid=False);
#plt.show()
print(X.Embarked.unique())
X.Embarked = X.Embarked.fillna(X.Embarked.dropna().mode()[0])
print(X.Embarked)
X.Embarked[X.Embarked == 'S'] = 0
X.Embarked[X.Embarked == 'C'] = 1
X.Embarked[X.Embarked == 'Q'] = 2
X['Ticket_len'] = X.Ticket.apply(len)
print(X.Ticket_len.value_counts())
X['Family'] = X.SibSp + X.Parch
X['Family'] = X['Family'].replace([0],'Alone')
X['Family'] = X['Family'].replace([1,2,3],'Little Family')
X['Family'] = X['Family'].replace([4,5,6,7,8,9,10],'Big Family')
X['Family'][X['Family'] == 'Alone'] = 0
X['Family'][X['Family'] == 'Little Family'] = 1
X['Family'][X['Family'] == 'Big Family'] = 2
print(X.Family.value_counts())
#X['Ticket_letter'] = X.Ticket.str[0]
#print(X.Ticket_letter.value_counts())
X = X.drop('Ticket', 1)	
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 0)
#print(X_train.shape)
#print(X_test.shape)
#print(y_train.shape)
#print(y_test.shape)
#print(train.describe(include=['O']))


#Logistic Regression
logis = LogisticRegression(penalty='l1', C = 5)
logis.fit(X_train,y_train)
y_pred = logis.predict(X_test)
print(y_pred)
print('Accuracy of logistic regression classifier on training set: {:.2f}'.format(logis.score(X_train, y_train)))
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logis.score(X_test, y_test)))
print('Confusion Matrix: ',confusion_matrix(y_test,y_pred)) 
testy_pred = logis.predict(test_X)

submission = test.copy()
submission['Survived'] = testy_pred
print(submission[['PassengerId', 'Survived']].head(15))

plt.figure()
sns.regplot(X_test.Age, y_test.Survived)
sns.regplot(X_test.Age, y_pred)
plt.show()

# Support Vector Machine
svm = SVC(kernel = 'rbf', gamma = 'scale', C=15)
svm.fit(X_train,y_train)
y_svm_pred = svm.predict(X_test)
#print(y_svm_pred)
print('Accuracy of SVM on training set: {:.2f}'.format(svm.score(X_train, y_train)))
print('Accuracy of SVM on test set: {:.2f}'.format(svm.score(X_test, y_test)))
print('Confusion Matrix: ',confusion_matrix(y_test,y_pred)) 
plt.figure()
sns.regplot(X_test.Age, y_test.Survived)
sns.regplot(X_test.Age, y_pred)
plt.show()

testy_pred = svm.predict(test_X)
submission_svm = test.copy()
submission_svm['Survived'] = testy_pred
print(submission_svm[['PassengerId', 'Survived']].head(15))