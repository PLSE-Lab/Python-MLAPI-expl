#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


import pandas as pd
gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")
test = pd.read_csv("../input/titanic/test.csv")
train = pd.read_csv("../input/titanic/train.csv")


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


train.head()


# In[ ]:


train['Age'] = train['Age'].fillna(train['Age'].mean())
train['Cabin'] = train['Cabin'].fillna(train['Cabin'].mode()[0])
train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])


# In[ ]:


train_y = train['Survived']
train_X = train.drop(['Survived','PassengerId','Name','Ticket','Fare','Embarked'], axis=1)
# del train


# In[ ]:


from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
train_X['Sex'] = lb_make.fit_transform(train_X['Sex'])
train_X[["Sex"]].head(11)#,["Cabin", "Cabin_code"]


# In[ ]:


from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
train_X['Cabin'] = lb_make.fit_transform(train_X['Cabin'])
train_X[["Cabin"]].head(11)#,["Cabin", "Cabin_code"]


# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(train_X, train_y, test_size=0.3, random_state=41)


# In[ ]:


# del train, train_y, train_X


# In[ ]:


import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
# from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression


# In[ ]:


clf = LogisticRegression(random_state=41).fit(X_train, Y_train)
clf.predict(X_test)


# In[ ]:


accuracy_score(Y_test, pred_list)


# In[ ]:


fil=open('gender_submission.csv','w')
fil.write('PassengerId,Survived\n')
i=1
for each in pred_list:
   fil.write('%d,%d\n'%(i+891,each))
   i=i+1
fil.close()


# In[ ]:


test.isna().sum()


# In[ ]:


test['Age'] = test['Age'].fillna(test['Age'].mean())
test['Cabin'] = test['Cabin'].fillna(test['Cabin'].mode()[0])


# In[ ]:


test = test.drop(['PassengerId','Name','Ticket','Fare','Embarked'], axis=1)


# In[ ]:


test.isna().sum()


# In[ ]:


test.head()


# In[ ]:


from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
test['Sex'] = lb_make.fit_transform(test['Sex'])
test[["Sex"]].head(11)#,["Cabin", "Cabin_code"]


# In[ ]:


from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
test['Cabin'] = lb_make.fit_transform(test['Cabin'])
test[["Cabin"]].head(11)#,["Cabin", "Cabin_code"]


# In[ ]:


test.head()


# In[ ]:


pred_list = clf.predict((test))


# In[ ]:




