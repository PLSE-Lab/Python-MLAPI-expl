#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train= pd.read_csv('/kaggle/input/titanic/train.csv')
train.head()


# In[ ]:


train.shape


# In[ ]:


test=pd.read_csv('/kaggle/input/titanic/test.csv')
test1=pd.read_csv('/kaggle/input/titanic/test.csv')

test.head()


# In[ ]:


test.shape


# In[ ]:


train.head()
#setting 'PassengerId' as Index
train.set_index(['PassengerId'],inplace=True)
test.set_index(['PassengerId'],inplace=True)
test.head()


# In[ ]:


train.isnull().sum()


# In[ ]:


test.isnull().sum()


# In[ ]:


train.dtypes


# In[ ]:


#missing value treatment

#first lets get a visual on these
import missingno as mn
mn.matrix(train)


# In[ ]:


mn.matrix(test)


# In[ ]:


#lets start imputing 'Age'
from sklearn.preprocessing import Imputer
imp=Imputer(missing_values='NaN',strategy='median',axis=1)
Age2=imp.fit_transform(train['Age'].values.reshape(1,-1))
Age2=Age2.T
train['Age2']=Age2
train.head()


# In[ ]:


Age_test=imp.fit_transform(test['Age'].values.reshape(1,-1))
Age_test=Age_test.T
test['Age_test']=Age_test
test.head()


# In[ ]:


train.isnull().sum()


# In[ ]:


test.isnull().sum()


# In[ ]:


#using fillna to fill 'Embarked' section values
train.Embarked.value_counts()


# In[ ]:


#S is the dominant column
train.Embarked.fillna('S',inplace=True)
train.isnull().sum()


# In[ ]:


#filling 'Fare' with mean
test.Fare.fillna(test.Fare.mean(),inplace=True)
train.isnull().sum()


# In[ ]:


#'cabin' contains more than 80% missing values, so dropping that as well as 'age' from before.
train.drop(['Age','Cabin'],axis=1,inplace=True)
test.drop(['Age','Cabin'],axis=1,inplace=True)
train.isnull().sum()


# In[ ]:


test.isnull().sum()


# In[ ]:


#no more missing values, now lets handle categorical data.
#transforming 'Sex' from object to int
train['Sex']=train.Sex.apply(lambda x:0 if x=='female' else 1)
test['Sex']=test.Sex.apply(lambda x:0 if x=='female' else 1)
test.Sex.head()


# In[ ]:


#removing outliers from 'Fare'
sns.boxplot('Survived','Fare',data=train)


# In[ ]:


train['Fare']=train[train['Fare']<=400]
test['Fare']=test[test['Fare']<=400]


# In[ ]:


#feature_engineering
train['family_size']=train['SibSp']+train['Parch']+1 #+1 if alone
test['family_size']=test['SibSp']+test['Parch']+1 #+1 if alone

train.head()


# In[ ]:


test.head()


# In[ ]:


#creating categories acc. to family_size
def family_group(size):
    a=''
    if(size<=1):
        a='alone'
    elif(size<=4):
        a='small'
    else:
        a='large'
    return a
train['family_group']=train.family_size.map(family_group)
test['family_group']=test.family_size.map(family_group)

train.head()


# In[ ]:


#creating categories acc. to age
def age_group(age):
    a=''
    if(age<=1):
        a='infant'
    elif(age<=4):
        a='small'
    elif(age<=14):
        a='child'
    elif(age<=25):
        a='young'
    elif(age<=40):
        a='adult'
    elif(age<=55):
        a='mid-age'
    else:
        a='old'
    return a
train['age_group']=train.Age2.map(age_group)
test['age_group']=test.Age_test.map(age_group)
train.age_group.value_counts()


# In[ ]:


#creating categories acc. to fare: fare per person
train['fare_per_person']=train['Fare']/train['family_size']
test['fare_per_person']=test['Fare']/test['family_size']

def fare_group(fare):
    a=''
    if(fare<=4):
        a='very-low'
    elif(fare<=10):
        a='low'
    elif(fare<=20):
        a='mid'
    elif(fare<=45):
        a='high'
    else:
        a='very-high'
    return a
train['fare_group']=train.fare_per_person.map(fare_group)
test['fare_group']=test.fare_per_person.map(fare_group)

test.fare_group.value_counts()


# In[ ]:


#creating dummy variables
train=pd.get_dummies(train,columns=['Embarked','family_group','age_group','fare_group'],drop_first=True)
test=pd.get_dummies(test,columns=['Embarked','family_group','age_group','fare_group'],drop_first=True)

#will do onehotencoding


# In[ ]:


train.shape
test.shape


# In[ ]:


#dropping unnecessary columns
train.drop(['Name','Ticket','Fare','Age2','fare_per_person','family_size'],axis=1,inplace=True)# Fare and fare-per_person are replaced by fare_group, age by age_group, family by family_group
test.drop(['Name','Ticket','Fare','Age_test','fare_per_person','family_size'],axis=1,inplace=True)
test.head()


# In[ ]:


X=train.drop('Survived',1)
y=train['Survived']


from xgboost import XGBClassifier
xgb=XGBClassifier()

score = cross_val_score(xgb, X, y, n_jobs=1, scoring= 'accuracy')
print(score)
round(np.mean(score)*100, 2)


# In[ ]:


test.head()


# In[ ]:


xgb=XGBClassifier()
xgb.fit(X, y)


# In[ ]:


#test_data = test.drop(['PassengerId'], axis=1).copy()
prediction = xgb.predict(test)
print(prediction)
print(len(prediction))


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test1['PassengerId'],
        "Survived": prediction
    })

submission.to_csv('submission.csv', index=False)

submission = pd.read_csv('submission.csv')


# > > <a href="./submission.csv"> Download File </a>
