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


import pandas as pd


# In[ ]:


train = pd.read_csv('/kaggle/input/titanic/train.csv')


# In[ ]:


train.head()


# In[ ]:


train.describe()


# In[ ]:


# help(train)


# In[ ]:


train.dtypes


# In[ ]:


train.iloc[0:100,0:5]


# In[ ]:


train.loc[0:100,'Survived']


# loc gets rows (or columns) with particular labels from the index.
# iloc gets rows (or columns) at particular positions in the index (so it only takes integers)

# In[ ]:


train


# In[ ]:


train.dropna()


# In[ ]:


print(len(train['Cabin']) - train['Cabin'].count())
print(len(train['Embarked']) - train['Embarked'].count())


# In[ ]:


train  =  train.drop(['Cabin'],axis = 1)


# In[ ]:


train  =  train.drop(['PassengerId'],axis = 1)


# In[ ]:


train  =  train.drop(['Name'],axis = 1)


# In[ ]:


train  =  train.drop(['Ticket'],axis = 1)


# In[ ]:


train


# In[ ]:


print('Survived : ' + str( len(train['Survived']) - train['Survived'].count()) )
print('Pclass : ' + str( len(train['Pclass']) - train['Pclass'].count()))
print('Sex : ' + str( len(train['Sex']) - train['Sex'].count()))
print('Age : ' + str( len(train['Age']) - train['Age'].count()))
print('SibSp : ' + str( len(train['SibSp']) - train['SibSp'].count()))
print('Parch : ' + str( len(train['Parch']) - train['Parch'].count()))
print('Ticket : ' + str( len(train['Ticket']) - train['Ticket'].count()))
print('Fare : ' + str( len(train['Fare']) - train['Fare'].count()))
print('Embarked : ' + str( len(train['Embarked']) - train['Embarked'].count()))


# In[ ]:


print(train['Age'].mean())


# In[ ]:


train['Age'] = train['Age'].fillna(train['Age'].mean())


# In[ ]:


print('Age : ' + str( len(train['Age']) - train['Age'].count()))


# In[ ]:


train = train.dropna()


# In[ ]:


train.dtypes


# In[ ]:


set(train['Sex'])


# In[ ]:


Sex_data_dic = {'female':0,'male':1}


# In[ ]:


train['Sex'] = train['Sex'].map(Sex_data_dic)


# In[ ]:


train.dtypes


# In[ ]:


set(train['Sex'])


# In[ ]:


set(train['Embarked'])


# In[ ]:


Embarked_data_dic = {'C':0, 'Q':1, 'S':2}


# In[ ]:


train['Embarked'] = train['Embarked'].map(Embarked_data_dic)


# In[ ]:


set(train['Embarked'])


# In[ ]:


train.describe()


# In[ ]:


train.head(10)


# In[ ]:




