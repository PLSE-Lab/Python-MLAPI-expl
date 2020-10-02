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


train.head()


# In[ ]:


train.isnull().sum()


# In[ ]:


import seaborn as sns
sns.countplot(train['Sex'])


# In[ ]:


sns.factorplot('Pclass',data=train,kind='count',hue='Sex')


# In[ ]:


def child(x):
    if x<12:
        return 'Child'
    else:
        return 'Elder'

train['Person']=train['Age'].apply(child)
train.head(10)


# In[ ]:


sns.factorplot('Pclass',data=train,kind='count',hue='Person')


# In[ ]:


fig=sns.FacetGrid(train,hue='Sex',aspect=4)
fig.map(sns.kdeplot,'Age',shade=True)
a=train['Age'].max()
fig.set(xlim=(0,a))
fig.add_legend()


# In[ ]:


fig=sns.FacetGrid(train,hue='Pclass',aspect=4)
fig.map(sns.kdeplot,'Age',shade=True)
a=train['Age'].max()
fig.set(xlim=(0,a))
fig.add_legend()


# In[ ]:


sns.factorplot('Embarked',data=train,kind='count',hue='Pclass')


# In[ ]:


def fam(x):
    sb,p=x
    if (sb==0) & (p==0):
        return 'alone'
    else:
        return 'family'
    
train['Family']=train[['SibSp','Parch']].apply(fam,axis=1)
train.head()


# In[ ]:


sns.factorplot('Family',data=train,kind='count',hue='Pclass')


# In[ ]:


sns.countplot(train['Survived'])


# In[ ]:


p_null=(len(train)-train.count())*100/len(train)
p_null


# In[ ]:


df1=train.drop(['Cabin','Name','PassengerId','Ticket'],axis=1)
df1.head()


# In[ ]:


df1['Embarked'].fillna('S',inplace=True)
df1.isnull().any()


# In[ ]:


df1['Age'].interpolate(inplace=True)
df1.isnull().any()


# In[ ]:


df1 = pd.get_dummies(df1, columns=["Sex","Embarked","Person","Family"])
df1.head()


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
a=MinMaxScaler()
scaled=a.fit_transform(df1[['Age','Fare']])
df1[['Age','Fare']]=pd.DataFrame(scaled)
df1.head()


# In[ ]:


df1.corr()


# In[ ]:


from sklearn.model_selection import train_test_split
X=df1.drop('Survived',axis=1)
y=df1['Survived']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)


# In[ ]:


import xgboost as xgb
model1 = xgb.XGBClassifier()
model2 = xgb.XGBClassifier(n_estimators=100, max_depth=8, learning_rate=0.1, subsample=0.5)

train_model1 = model1.fit(X_train, y_train)
train_model2 = model2.fit(X_train, y_train)


# In[ ]:


from sklearn.metrics import accuracy_score
pred1 = train_model1.predict(X_test)
pred2 = train_model2.predict(X_test)
print("Accuracy for model 1: %.2f" % (accuracy_score(y_test, pred1) * 100))
print("Accuracy for model 2: %.2f" % (accuracy_score(y_test, pred2) * 100))


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()
rfc_model = rfc.fit(X_train, y_train)
pred8 = rfc_model.predict(X_test)
print("Accuracy for Random Forest Model: %.2f" % (accuracy_score(y_test, pred8) * 100))


# In[ ]:


# Test set


# In[ ]:


def fam(x):
    sb,p=x
    if (sb==0) & (p==0):
        return 'alone'
    else:
        return 'family'
    
test['Family']=test[['SibSp','Parch']].apply(fam,axis=1)
test.head()


# In[ ]:


def child(x):
    if x<12:
        return 'Child'
    else:
        return 'Elder'

test['Person']=test['Age'].apply(child)
test.head(10)


# In[ ]:


df2=test.drop(['Cabin','Name','PassengerId','Ticket'],axis=1)
df2.head()


# In[ ]:


df2 = pd.get_dummies(df2, columns=["Sex","Embarked","Person","Family"])
df2.head()


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
a=MinMaxScaler()
scaled=a.fit_transform(df2[['Age','Fare']])
df2[['Age','Fare']]=pd.DataFrame(scaled)
df2.head()


# In[ ]:


pred4 = train_model2.predict(df2)


# In[ ]:


pred4


# In[ ]:


pred=pd.DataFrame(pred4)
df = pd.read_csv("../input/titanic/gender_submission.csv")
data=pd.concat([df['PassengerId'],pred],axis=1)
data.columns=['PassengerId','Survived']
data.to_csv('sample_submission.csv',index=False)


# In[ ]:




