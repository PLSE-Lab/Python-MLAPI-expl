# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
train = pd.read_csv("/kaggle/input/titanic/train.csv", index_col= "PassengerId")
test = pd.read_csv("/kaggle/input/titanic/test.csv", index_col= "PassengerId")

# Any results you write to the current directory are saved as output.

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score



train["Survived"] = train["Survived"].astype(str)
train["Pclass"] = train["Pclass"].astype(str)


#replace missing data
train['Age'].fillna(train['Age'].mean(),inplace=True)
train['Embarked'].fillna(train['Embarked'].value_counts().index[0],inplace= True)
train = train.drop(columns='Cabin')

#convert string to float
sexdummy = pd.get_dummies(train['Sex'],prefix='Sex', prefix_sep='_',columns=1)
trainwithdummy = pd.concat([train,sexdummy],axis=1,sort=True)

features = ['Pclass','SibSp','Parch', 'Fare', 'Age']
target = ['Survived']
X = train[features]
y = train[target]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.469, random_state = 40)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

pred_test = model.predict(X_test)
pred_train = model.predict(X_train)
print(pred_test)

# evaluate result 
print("Accuracy from 40% test data:",accuracy_score(y_test, pred_test, normalize=True, sample_weight=None))
print("Confusion Matrix:", "\n", confusion_matrix(y_test, pred_test))

print("Accuracy from 60% train data:",accuracy_score(y_train, pred_train, normalize=True, sample_weight=None))
print("Confusion Matrix:", "\n", confusion_matrix(y_train, pred_train))

#cross-validation
score_cv = cross_val_score(model, X, y, cv=10)
print(score_cv.mean())

#output

test2 = pd.read_csv("/kaggle/input/titanic/test.csv")
output = pd.DataFrame({'PassengerId': test2['PassengerId'], 'Survived': pred_test})
output.to_csv('MySubmission.csv', index=False)