# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import seaborn as sns
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
import math# data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")
titanic_data = train_data.append(test_data)
titanic_data.head()
sns.countplot(x = 'Survived' , data = titanic_data)
sns.countplot(x = 'Survived',hue = 'Pclass' , data = titanic_data)
sns.countplot(x = 'Age',data = titanic_data)
sns.countplot(x = 'Survived',hue = 'Sex', data = titanic_data)
titanic_data.isnull().sum()
titanic_data.head(10)
sns.boxplot(x='Pclass', y = 'Age',data = titanic_data)
titanic_data.drop("Cabin",axis = 1,inplace = True)
titanic_data.head()
titanic_data.dropna(inplace = True)
titanic_data.isnull().sum()
sex = pd.get_dummies(titanic_data['Sex'],drop_first = True)
embark = pd.get_dummies(titanic_data['Embarked'],drop_first = True)
Pc1 = pd.get_dummies(titanic_data['Pclass'],drop_first =True)
Pc1.head(5)
titanic_data = pd.concat([titanic_data,sex,embark,Pc1],axis =1)
titanic_data.head(5)
titanic_data.drop(['Name','Sex','Embarked','Pclass','PassengerId','Ticket'],axis = 1,inplace = True)
titanic_data.head()
X = titanic_data.drop('Survived',axis = 1)
y = titanic_data['Survived']
from sklearn.model_selection import train_test_split
train_X,test_X,train_y,test_y = train_test_split(X,y,test_size = 0.3,random_state = 1) 
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(train_X,train_y)
predictions = model.predict(test_X)
from sklearn.metrics import classification_report
classification_report(test_y,predictions)
from sklearn.metrics import accuracy_score
accuracy_score(test_y,predictions)
submission = pd.DataFrame({
        #"PassengerId": titanic_data["PassengerId"]
        "Survived": predictions
    })
submission.head()






