#!/usr/bin/env python
# coding: utf-8

# # linear regression with titanic dataset

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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train = pd.read_csv('../input/titanic_train.csv')


# In[ ]:


train.head()


# In[ ]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis') #To get the columns which has null values


# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train)


# In[ ]:


sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r') #survival based on sex


# In[ ]:


sns.countplot(x='Survived',hue='Pclass',data=train) #survival based on passenger class


# In[ ]:


sns.distplot(train['Age'].dropna(),kde=False,bins=30)


# In[ ]:


sns.countplot(x='SibSp',data=train) # count based on sibling or spouse


# In[ ]:


train['Fare'].hist(bins=40,figsize=(10,4))


# In[ ]:


import cufflinks as cf
cf.go_offline()


# In[ ]:


train['Fare'].iplot(kind='hist',bins=40)


# In[ ]:


sns.countplot(x='Parch',data=train)


# In[ ]:


plt.figure(figsize=(10,7))
sns.boxplot(x='Pclass',y='Age',data=train)


# In[ ]:


def inpute_age(cols):
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


train['Age'] = train[['Age','Pclass']].apply(inpute_age,axis=1)


# In[ ]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False)


# In[ ]:


train.drop('Cabin',axis=1,inplace=True)


# In[ ]:


train.dropna(inplace=True)


# In[ ]:


sex = pd.get_dummies(train['Sex'],drop_first=True)


# In[ ]:


sex.head()


# In[ ]:


embark = pd.get_dummies(train['Embarked'],drop_first=True)


# In[ ]:


embark.head()


# In[ ]:


train = pd.concat([train,sex,embark],axis=1)


# In[ ]:


train.head()


# In[ ]:


train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)


# In[ ]:


train.head()


# In[ ]:


train.tail()


# In[ ]:


train.drop('PassengerId',axis=1,inplace=True)


# In[ ]:


train.head()


# In[ ]:


pclass = pd.get_dummies(train['Pclass'])


# In[ ]:


pclass.head()


# In[ ]:


X= train.drop('Survived',axis=1)
y= train['Survived'] #trying to predict


# In[ ]:


from sklearn.cross_validation import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


logmodel = LogisticRegression()


# In[ ]:


logmodel.fit(X_train,y_train)


# In[ ]:


predictions = logmodel.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


print(classification_report(y_test,predictions))


# In[ ]:


from sklearn.metrics import confusion_matrix


# In[ ]:


confusion_matrix(y_test,predictions)


# In[ ]:




