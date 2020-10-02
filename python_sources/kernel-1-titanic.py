#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


train_data = pd.read_csv("/kaggle/input/titanic/train.csv", dtype={'Age': np.float64})
test_data = pd.read_csv("/kaggle/input/titanic/test.csv", dtype={'Age': np.float64})
gender_submission = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")


# In[ ]:


def gen2numConverter(item):
    if(item == "male"):
        return 1
    else:
        return 0


# In[ ]:


train_data["Sex_Num"] = train_data["Sex"].apply(gen2numConverter)
test_data["Sex_Num"]= test_data["Sex"].apply(gen2numConverter)


# In[ ]:


train_data.drop(['Sex','Embarked','Name','Ticket','Cabin','Embarked','SibSp','Parch'],axis=1,inplace=True)
test_data.drop(['Sex','Embarked','Name','Ticket','Cabin','Embarked','SibSp','Parch'],axis=1,inplace=True)


# In[ ]:


def isNAN2Num(item):
    if np.isnan(item):
        return 0
    else:
        return item


# In[ ]:


train_data["PassengerId"] = train_data["PassengerId"].apply(isNAN2Num)
train_data["Pclass"] = train_data["Pclass"].apply(isNAN2Num)
train_data["Age"] = train_data["Age"].apply(isNAN2Num)
train_data["Fare"] = train_data["Fare"].apply(isNAN2Num)
train_data["Sex_Num"] = train_data["Sex_Num"].apply(isNAN2Num)
train_data["Survived"] = train_data["Survived"].apply(isNAN2Num)


# In[ ]:


test_data["PassengerId"] = test_data["PassengerId"].apply(isNAN2Num)
test_data["Pclass"] = test_data["Pclass"].apply(isNAN2Num)
test_data["Age"] = test_data["Age"].apply(isNAN2Num)
test_data["Fare"] = test_data["Fare"].apply(isNAN2Num)
test_data["Sex_Num"] = test_data["Sex_Num"].apply(isNAN2Num)


# In[ ]:


test_data.head()


# In[ ]:


X_train = train_data.drop('Survived',axis=1)
Y_train = train_data['Survived']
X_test = test_data
Y_test = gender_submission['Survived']


# In[ ]:


from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,Y_train)


# In[ ]:


test_data["Survived"] = logmodel.predict(X_test)


# In[ ]:


test_data.head()


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(Y_test,test_data["Survived"]))


# In[ ]:


from sklearn.metrics import confusion_matrix
print(confusion_matrix(Y_test,test_data["Survived"]))

