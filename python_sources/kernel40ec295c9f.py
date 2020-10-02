#!/usr/bin/env python
# coding: utf-8

# In[170]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[171]:


#import train
train=pd.read_csv('../input/train.csv')

#combining data
X_test=pd.read_csv('../input/test.csv')
X_test_passenger=X_test
titanic_df=pd.concat([train,X_test])


# In[172]:


#DataExploration
import matplotlib.pyplot as plt

titanic_df.describe()
titanic_df.head()
titanic_df.count()


# In[173]:


#overview age/sex/survival
sexclass=titanic_df.groupby(['Sex','Pclass']).mean()
sexclass


# In[174]:


#plot of Survivalrates Sex/Class
sexclass['Survived'].plot.bar()


# In[175]:


#Sex/Age Survivalrates
bin_age=pd.cut(titanic_df['Age'],10)
titanic_df.groupby(bin_age)['Survived'].mean().plot.bar()


# In[176]:


#violin plot for correlation
import seaborn as sns
sns.violinplot(x="Sex",y="Age", hue="Survived",data=titanic_df, split=True)


# In[177]:


titanic_df.groupby('Embarked')['Survived'].mean()
titanic_df.groupby('Embarked').sum()


# #Feature Engineering Names& Titles

# In[178]:


titanic_df['Title']=titanic_df.Name.apply(lambda x: x.split('.')[0].split(',')[1].strip())
titlegroup={
    "Capt":"Officer",
    "Col": "Officer",
    "Major":"Officer",
    "Jonkheer":   "HighClass",
    "Don":        "HighClass",
    "Sir" :       "HighClass",
    "Lady" :      "HighClass",
    "Dr":         "Officer",
    "Rev":        "Officer",
    "the Countess":"HighClass",
    "Dona":       "HighClass",
    "Mme":        "Mrs",
    "Mlle":       "Miss",
    "Ms":         "Mrs",
    "Mr" :        "Mr",
    "Mrs" :       "Mrs",
    "Miss" :      "Miss",
    "Master" :    "Master"}

titanic_df['Title']=titanic_df['Title'].map(titlegroup)


# fill NANs Age/Fare/Family Size
# #Code blow is a quick fix for missing age values, a better estimation can be done bei using the available data 
# #titanic_df['Age']=titanic_df['Age'].fillna(titanic_df.Age.median())

# In[179]:


print(titanic_df.groupby(['Title','Sex','Pclass']).Age.mean())
agegroups=titanic_df.groupby(['Title','Sex','Pclass']).Age.mean()


# In[180]:


def missingage(features):
    if pd.isnull(features['Age']):
        return agegroups[features['Title'],features['Sex'],features['Pclass']].mean()
    else:
        return features['Age']

titanic_df['Age'] =titanic_df.apply(missingage, axis=1)


# In[181]:


#fill embarked with 'S' since it is the most common port
titanic_df['Embarked']=titanic_df['Embarked'].fillna('S')
titanic_df['Fare']=titanic_df['Fare'].fillna(titanic_df.Fare.median())
titanic_df['FamilySize']=titanic_df['Parch']+titanic_df['SibSp']


# In[182]:


titanic_df.head()
titanic_df.Title.unique()


# In[183]:


#labelencoding of data
from sklearn.preprocessing import LabelEncoder
label=LabelEncoder()
titanic_df.Sex=label.fit_transform(titanic_df['Sex'])
titanic_df.Embarked=label.fit_transform(titanic_df['Embarked'])
titanic_df.Title=label.fit_transform(titanic_df['Title'])


# In[184]:


#data prep/subset of usefull data
y_train=titanic_df.loc[:,'Survived']
subset=['Pclass','Sex','Age','FamilySize','Embarked','Fare','Title']
titanic_df=titanic_df.loc[:,subset]
titanic_df.head()


# In[185]:


""""#train Algorithm
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=50)
clf.fit(titanic_df.iloc[:891],y_train[:891])"""


# In[186]:


"""#predict
prediction=clf.predict(titanic_df.iloc[891:])
submit=pd.DataFrame({"PassengerId":X_test["PassengerId"], "Survived":prediction.astype(int)})
submit.to_csv('titanic.csv',index=False)
print('COMPLETED')"""


# In[187]:


#XGboost
from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(titanic_df.iloc[:891],y_train[:891])
#predict
prediction=xgb.predict(titanic_df.iloc[891:])
submit=pd.DataFrame({"PassengerId":X_test["PassengerId"], "Survived":prediction.astype(int)})
submit.to_csv('titanic.csv',index=False)
print('COMPLETED')

