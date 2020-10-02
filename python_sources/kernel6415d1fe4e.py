#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


train = pd.read_csv('../input/titanic/train.csv')
train.head()


# In[ ]:


train.drop(['Name','Ticket'],axis = 1,inplace=True)
train.head()


# In[ ]:


sex = pd.get_dummies(train['Sex'],drop_first=True)


# In[ ]:


embark = pd.get_dummies(train['Embarked'],drop_first=True)


# In[ ]:


train.drop(['Sex','Embarked'],axis=1,inplace=True)


# In[ ]:


df = pd.concat([train,sex,embark],axis=1)


# In[ ]:


sns.heatmap(df.isnull(),cmap='plasma')


# In[ ]:


df.drop('Cabin',inplace=True,axis=1)


# In[ ]:


sns.heatmap(df.isnull(),cmap='plasma')


# In[ ]:


sns.boxplot(x='Pclass',y='Age',data=df)


# In[ ]:


def age_find(cols):
    age = cols[0]
    pclass = cols[1]
    
    if pd.isnull(age):
        if pclass == 1:
            return 37
        elif pclass == 2:
            return 29
        else:
            return 23
    
    else:
        return age


# In[ ]:


df['Age'] = df[['Age','Pclass']].apply(age_find,axis=1)


# In[ ]:


sns.heatmap(df.isnull(),cmap='plasma')


# In[ ]:


df.head()


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


logmodel = LogisticRegression()


# In[ ]:


logmodel.fit(df.drop('Survived',axis=1),df['Survived'])


# In[ ]:


test = pd.read_csv('../input/titanic/test.csv')
test.head()


# In[ ]:


test.drop(['Name','Ticket','Cabin'],axis=1,inplace=True)


# In[ ]:


test.head()


# In[ ]:


test_sex = pd.get_dummies(test['Sex'],drop_first=True)


# In[ ]:


test_embark = pd.get_dummies(test['Embarked'],drop_first=True)


# In[ ]:


df2 = pd.concat([test,test_sex,test_embark],axis=1)


# In[ ]:


df2.head()


# In[ ]:


df2.drop(['Sex','Embarked'],inplace=True,axis=1)


# In[ ]:


df2.head()


# In[ ]:


sns.heatmap(df2.isnull(),cmap='plasma')


# In[ ]:


sns.boxplot(x='Pclass',y='Age',data=df2)


# In[ ]:


def age_set(cols):
    age = cols[0]
    pclass = cols[1]
    
    if pd.isnull(age):
        if pclass == 1:
            return 41
        elif pclass == 2:
            return 27
        else:
            return 23
    else:
        return age


# In[ ]:


df2['Age'] = df2[['Age','Pclass']].apply(age_set,axis=1)


# In[ ]:


sns.heatmap(df2.isnull(),cmap='plasma')


# In[ ]:


df2.dropna(inplace=True)


# In[ ]:


sns.heatmap(df2.isnull(),cmap='plasma')


# In[ ]:


pred = logmodel.predict(df2)


# In[ ]:


sns.countplot(pred)


# In[ ]:




