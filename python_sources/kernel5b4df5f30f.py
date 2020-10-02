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


#This titanic dataset using DecisionTreeClassifier algorithm 


# In[ ]:


#now uploading pandas library  and titanic dataset
import pandas as pd
df = pd.read_csv("../input/titanicdataset-traincsv/train.csv")
df.head()


# In[ ]:


#drop uneccesary feature columns  or clecaning dataset
df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns',inplace=True)
df.head()


# In[ ]:


# now drop the Survived columns 
inputs = df.drop('Survived',axis='columns')
target = df.Survived


# In[ ]:


#now this is converting string to interger
inputs.Sex = inputs.Sex.map({'male': 1, 'female': 2})


# In[ ]:


inputs.Age[:10]


# In[ ]:


#filling nan 
inputs.Age = inputs.Age.fillna(inputs.Age.mean())


# In[ ]:


inputs.head()


# In[ ]:


#splitin data
from sklearn.model_selection import train_test_split


# In[ ]:


#spliting data
X_train, X_test, y_train, y_test = train_test_split(inputs,target,test_size=0.2)


# In[ ]:


len(X_train)


# In[ ]:


len(X_train)


# In[ ]:


len(X_test)


# In[ ]:


len(X_train)


# In[ ]:


from sklearn import tree
model = tree.DecisionTreeClassifier()


# In[ ]:


model.fit(X_train,y_train)


# In[ ]:


model.score(X_test,y_test)


# In[ ]:


model.score(X_train,y_train)


# In[ ]:




