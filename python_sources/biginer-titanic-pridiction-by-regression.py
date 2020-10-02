#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:



import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train.drop(['Name'],axis=1, inplace=True)
test.drop(['Name'],axis=1,inplace=True)


# In[ ]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:


train.isnull().sum()


# In[ ]:


test.isnull().sum()


# In[ ]:


train.drop(['Cabin'],axis=1,inplace=True)
test.drop(['Cabin'],axis=1,inplace=True)
train.drop(['Ticket'],axis=1,inplace=True)
test.drop(['Ticket'],axis=1,inplace=True)
train.drop(['PassengerId'],axis=1,inplace=True)
test.drop(['PassengerId'],axis=1,inplace=True)


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='Survived', data=train)


# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex', data=train)


# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass', data=train)


# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='Survived', hue='Parch', data=train)


# In[ ]:


sns.distplot(train['Age'].dropna(),kde=False, color='red', bins=30)


# In[ ]:


sns.countplot(x='SibSp', data=train)


# In[ ]:


train['Fare'].hist(color='blue',bins=40,figsize=(8,4))


# In[ ]:


import cufflinks as cf
cf.go_offline()


# In[ ]:


train['Fare'].iplot(kind='hist',bins=30,color='green')


# In[ ]:


plt.figure(figsize=(12,7))
sns.boxplot(x='Pclass', y='Age', data=train)


# In[ ]:


#we can see the wealthier passengers in the higher classes tend to be older, which makes sense. we will use these average age values to impute based on Pclass for Age.


# In[ ]:


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
         
            if Pclass == 1:
                return 37
            
            elif Pclass == 2:
                return 29
            
            else:
                return 24
            
            
    else:
        return Age


# In[ ]:


train['Age'] = train[['Age', 'Pclass']].apply(impute_age, axis=1)


# In[ ]:


test['Age']=test[['Age','Pclass']].apply(impute_age, axis=1)


# In[ ]:


test.isnull().sum()


# In[ ]:


test['Fare'].fillna((test['Fare'].mean()), inplace=True)


# In[ ]:


train.dropna(inplace=True)


# In[ ]:


Pclass = pd.get_dummies(train['Pclass'],drop_first = True)
Pclass1 = pd.get_dummies(test['Pclass'],drop_first = True)
Sex = pd.get_dummies(train['Sex'],drop_first = True)
Sex1 = pd.get_dummies(test['Sex'],drop_first = True)
Embarked = pd.get_dummies(train['Embarked'],drop_first = True)
Embarked1 = pd.get_dummies(test['Embarked'],drop_first = True)


# In[ ]:


train = pd.concat([train,Pclass,Sex,Embarked],axis=1)
test = pd.concat([test,Pclass1,Sex1,Embarked1],axis=1)


# In[ ]:


train.drop(['Pclass','Sex','Embarked'], axis=1, inplace=True)
test.drop(['Pclass','Sex','Embarked'], axis=1, inplace=True)


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


from sklearn.model_selection import train_test_split 
Y= train['Survived']
X= train.drop('Survived',axis=1)


# In[ ]:


X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.3, random_state=101)


# In[ ]:


from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train, Y_train)


# In[ ]:


predictions = logmodel.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix


# In[ ]:


print(classification_report(Y_test,predictions))


# In[ ]:


print(classification_report(Y_test,predictions))


# In[ ]:


print(confusion_matrix(Y_test,predictions))


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
gbk = GradientBoostingClassifier()
gbk.fit(X_train, Y_train)
Y_pred = gbk.predict(X_test)
acc_gbk = round(accuracy_score(Y_pred, Y_test) * 100, 2)
print(acc_gbk)


# In[ ]:


sample_sub=pd.read_csv("../input/gender_submission.csv")


# In[ ]:


sample_sub.head()


# In[ ]:


test.index


# In[ ]:


sample_sub.index


# In[ ]:


predictions1 = gbk.predict(test)
sample_sub['Survived']= predictions1
sample_sub.to_csv("submit.csv", index=False)
sample_sub.head()


# In[ ]:




