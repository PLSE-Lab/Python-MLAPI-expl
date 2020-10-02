#!/usr/bin/env python
# coding: utf-8

# This is a simple implementation of Logistic Regression model

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


import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


# In[ ]:


train_df = pd.read_csv('../input/titanic/train.csv')
test_df = pd.read_csv('../input/titanic/test.csv')


# In[ ]:


train_df.head()


# In[ ]:


x = train_df.Survived
y = train_df.Embarked
print(len(x))


# In[ ]:


train_df.describe()


# In[ ]:


train_df.info()


# In[ ]:


plt.hist(train_df['Fare'],bins=10)
plt.show()


# In[ ]:


male=0
female=0
for gender in train_df['Sex']:
    if gender == 'male':
        male=male+1
    if gender == 'female':
        female=female+1
        
print(male,female,len(train_df['Sex']))


# In[ ]:


train_df.iloc[0]


# In[ ]:


train_df.head()


# In[ ]:


mensur=0
femsur=0

for i in train_df.index:
    if(train_df.iloc[i][2]==1 and train_df.iloc[i][4]=='male'):
        mensur=mensur+1
    if(train_df.iloc[i][2]==1 and train_df.iloc[i][4]=='female'):
        femsur=femsur+1
        
            
        
            
            


# In[ ]:


print(mensur,femsur,(mensur+femsur)/len(train_df.index))


# In[ ]:


(train_df.iloc[2][4])


# In[ ]:


sns.countplot(x='Survived',data=train_df)


# In[ ]:


sns.countplot(x=('Survived'),hue='Pclass',data=train_df)


# In[ ]:


train_df['Survived']


# In[ ]:


age_df = pd.DataFrame({
    'Age' : train_df['Age'],
    'Survived' : train_df['Survived']
})
#age_df = train_df['Age','Survived']
age_df.head()


# In[ ]:


survival_age=[]
for i in age_df.index:
    if age_df.iloc[i][1]==1:
        survival_age.append(age_df.iloc[i][0])


# In[ ]:


plt.hist(x=survival_age,bins=10)


# Data Wrangling

# In[ ]:


train_df.isnull().sum()


# In[ ]:


sns.heatmap(train_df.isnull() ,yticklabels=False)


# In[ ]:


train_df.drop('Cabin',axis=1,inplace=True)


# In[ ]:


train_df.dropna(inplace=True)


# In[ ]:


sns.heatmap(train_df.isnull() ,yticklabels=False)


# Changing variables into catagorical form

# In[ ]:


train_df.head()


# In[ ]:


Pclass = pd.get_dummies(train_df['Pclass'],drop_first=True)
Pclass.head()


# In[ ]:


sex = pd.get_dummies(train_df['Sex'],drop_first=True)
sex.head()


# In[ ]:


Embarked = pd.get_dummies(train_df['Embarked'],drop_first=True)
Embarked.head()


# In[ ]:


train_df=pd.concat([train_df,Embarked,sex,Pclass],axis=1)
train_df.head()


# In[ ]:


train_df.drop(['PassengerId','Pclass','Name','Sex','Ticket','Embarked'],axis=1,inplace=True)
train_df.head()


# Training Model

# In[ ]:


logmodel=LogisticRegression()
X = train_df.drop("Survived",axis=1)
y= train_df["Survived"]


# In[ ]:


logmodel.fit(X,y)


# In[ ]:


test_df.head()


# In[ ]:




