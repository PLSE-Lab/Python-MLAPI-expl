# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
DC=DecisionTreeClassifier()
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 10,criterion = 'entropy',random_state=0)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
train=pd.read_csv('../input/titanic/train.csv')
test=pd.read_csv('../input/titanic/test.csv')
gend_sub=pd.read_csv('../input/titanic/gender_submission.csv')
print(train.shape)
print(test.shape)
print(gend_sub.shape)
from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
train['Sex']=LE.fit_transform(train['Sex'])
sns.countplot(train['Survived'],hue=train['Sex'])
train.groupby(train['Sex'])[['Survived']].mean()
sns.countplot(train['Survived'],hue=train['Pclass'])
train.drop(['PassengerId','Name','Cabin','Embarked','Fare'],axis=1,inplace=True)
train.drop(['Ticket'],axis=1,inplace=True)
train.drop(['Age'],axis=1,inplace=True)
sns.countplot(train['Parch'],hue=train['Survived'])
train.dtypes
from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
test['Sex']=LE.fit_transform(test['Sex'])
lr=LogisticRegression()

X =train[['Pclass', 'Sex', 'SibSp', 'Parch']]
Y =train['Survived']
test_features=test[['Pclass', 'Sex', 'SibSp', 'Parch']]
to_predict=gend_sub['Survived']



#Logistic Regression
lr.fit(X,Y)

pred=lr.predict(test_features)
print('Prediction made in Logistic Regression:',pred)

acc_score= accuracy_score(to_predict,pred,normalize=True)

print('Accuracy sore of Logistic Regression:',acc_score*100)

