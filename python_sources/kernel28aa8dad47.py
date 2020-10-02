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


train=pd.read_csv('/kaggle/input/titanic/train.csv')


# In[ ]:


train


# In[ ]:


train.isnull().sum()


# In[ ]:


train.corr()


# In[ ]:


import seaborn as snp


# In[ ]:


snp.countplot(train['Pclass'])


# In[ ]:


train['Age'].value_counts()


# In[ ]:


snp.distplot(a=train['Age'])


# In[ ]:


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
mypipeline=Pipeline([
    
    ('imputer',SimpleImputer(strategy="median"))
])


# In[ ]:


train.describe(include="all")


# In[ ]:


train.dtypes


# In[ ]:


train=train.drop(['Name','Ticket'],axis=1)


# In[ ]:


train['Sex'].unique


# In[ ]:


train['Embarked'].unique


# In[ ]:


train['Sex']=train['Sex'].map({'male':1,'female':2})
train['Embarked']=train['Embarked'].map({'C':0,'S':1,'Q':2})


# In[ ]:


train


# In[ ]:


train=train.drop('Cabin',axis=1)


# In[ ]:


headers=train.columns.values
headers=list(headers)


# In[ ]:


train=mypipeline.fit_transform(train)


# In[ ]:


train


# In[ ]:


cleaned_train=pd.DataFrame(train)


# In[ ]:


cleaned_train.columns=headers
cleaned_train


# In[ ]:


cleaned_train.isnull().sum()


# In[ ]:


snp.distplot(cleaned_train['Age'],kde=True)


# In[ ]:


test=pd.read_csv('/kaggle/input/titanic/test.csv')


# In[ ]:


test


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


test.isnull().sum()


# In[ ]:


test=test.drop(['Name','Ticket','Cabin'],axis=1)


# In[ ]:


test['Sex']=test['Sex'].map({'male':1,'female':2})
test['Embarked']=test['Embarked'].map({'C':0,'S':1,'Q':2})


# In[ ]:


test=mypipeline.fit_transform(test)


# In[ ]:


cleaned_test=pd.DataFrame(test)
headers.remove('Survived')
cleaned_test.columns=headers
cleaned_test


# In[ ]:


from sklearn.model_selection import train_test_split

# Here is out local validation scheme!
X_train, X_test, y_train, y_test = train_test_split(cleaned_train.drop(['Survived'], axis = 1), 
                                                    cleaned_train['Survived'], test_size = 0.2, 
                                                    random_state = 42)


# In[ ]:


model=RandomForestClassifier(random_state=42,n_estimators=500)


# In[ ]:


model.fit(X_train,y_train)


# In[ ]:


model.score(X_train,y_train)


# In[ ]:


pred=model.predict(X_test)
pred


# In[ ]:





# In[ ]:


from sklearn.metrics import accuracy_score
print(accuracy_score(pred,y_test))


# In[ ]:


cleaned_test['Survived']=model.predict(cleaned_test)


# In[ ]:


cleaned_test['PassengerId']=cleaned_test['PassengerId'].astype(int)


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, pred))


# In[ ]:


cleaned_test[['PassengerId','Survived']]


# In[ ]:


cleaned_test[['PassengerId','Survived']].to_csv('kaggle_submission.csv',index=False)


# In[ ]:




