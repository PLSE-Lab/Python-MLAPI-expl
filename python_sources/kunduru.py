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


df=pd.read_csv('/kaggle/input/titanic/train.csv')


# In[ ]:


df.head() 


# In[ ]:


df.info()


# In[ ]:


df['Pclass'].value_counts().plot(kind='bar')


# In[ ]:


df['Survived'].value_counts().plot(kind='bar')


# In[ ]:


df.isna().sum()


# In[ ]:


df['Age'].hist()


# In[ ]:


df[df['Age'].notnull()].shape


# In[ ]:


df['Cabin_pre']=df['Cabin'].str[0]
df['Cabin_pre'][339]='O'


# In[ ]:


df['Cabin_pre']=df['Cabin_pre'].fillna(value='O')


# In[ ]:


df['Cabin_pre']=df['Cabin_pre'].replace(df.groupby('Cabin_pre')['Survived'].mean())
df.groupby('Cabin_pre')['Survived'].mean()


# In[ ]:


df['Embarked']=df['Embarked'].replace(df.groupby('Embarked')['Survived'].mean())
df.groupby('Embarked')['Survived'].mean()


# In[ ]:


df['Fare'].hist()


# In[ ]:


from sklearn.preprocessing import PowerTransformer
pt=PowerTransformer()


# In[ ]:


df['Fare_log']=pt.fit_transform(df['Fare'].values.reshape(-1,1))


# In[ ]:


df['Fare_log'].hist()


# In[ ]:


df['Ticket_s']=df['Ticket'].str[0]
df['Ticket_s']=df['Ticket_s'].apply(lambda x:1 if x.isnumeric() else 0)
df['Ticket_s'].value_counts()


# In[ ]:


df['Ticket_l']=df['Ticket'].str.len()


# In[ ]:


df=df.drop(['Ticket','Fare'],axis=1)


# In[ ]:


df['Sex']=df['Sex'].replace({'male':1,'female':0})


# In[ ]:


df=df.drop(['PassengerId','Cabin'],axis=1)


# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor
gbr=GradientBoostingRegressor()


# In[ ]:


y=df[df['Age'].notnull()]['Age']
x=df[df['Age'].notnull()]


# In[ ]:


test=df[df['Age'].isnull()].drop(['Age','Name'],axis=1)


# In[ ]:


x=x.fillna(value=x['Embarked'].max())
x.shape


# In[ ]:


gbr.fit(x,y.values)


# In[ ]:


df['Embarked']=df['Embarked'].fillna(value=x['Embarked'].max())


# In[ ]:


df.loc[df[df['Age'].isnull()].index,'Age']=gbr.predict(test)


# In[ ]:


df.info()


# In[ ]:


name=df.pop('Name')


# In[ ]:


from sklearn.svm import SVC
model=SVC()


# In[ ]:


Y=df['Survived'].values
X=df.drop(['Survived'],axis=1)


# In[ ]:


from sklearn.model_selection import cross_val_score


# In[ ]:


cross_val_score(model,X,Y,cv=5)


# In[ ]:




