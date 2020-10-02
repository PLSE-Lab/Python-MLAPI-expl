#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # Data Cleaning and Dealing with missing values

# In[ ]:


df=pd.read_csv('../input/train.csv')
df.head()


# In[ ]:


df_test=pd.read_csv('../input/test.csv')
df_test.head()


# In[ ]:


mat=df.corr()
mat


# In[ ]:


sns.heatmap(mat)


# In[ ]:


df.count()


# In[ ]:


df_test.isnull().sum()


# In[ ]:


df['Cabin'].apply(lambda i:str(i)[0]).unique()


# In[ ]:


df_test['Cabin'].apply(lambda i:str(i)[0]).unique()


# In[ ]:


df['Cabin'].fillna(value=0,inplace=True)


# In[ ]:


df_test['Cabin'].fillna(value=0,inplace=True)


# In[ ]:


df['Cabin'].head()


# In[ ]:


df_test['Cabin'].head()


# In[ ]:


df['Cabin']=df['Cabin'].apply(lambda i:str(i)[0]).replace({'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7,'T':8})
df['Cabin'].head()


# In[ ]:


df_test['Cabin']=df_test['Cabin'].apply(lambda i:str(i)[0]).replace({'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7,'T':8})
df_test['Cabin'].head()


# In[ ]:


df['Cabin']=df['Cabin'].astype(dtype='int32')


# In[ ]:


df_test['Cabin']=df_test['Cabin'].astype(dtype='int32')


# In[ ]:


df['Sex']=df['Sex'].replace({'male':1,'female':2})


# In[ ]:


df_test['Sex']=df_test['Sex'].replace({'male':1,'female':2})


# In[ ]:


df['Age']=df['Age'].fillna(value=df['Age'].mean())   #or value='bfill'||'ffill'


# In[ ]:


df_test['Age']=df_test['Age'].fillna(value=df['Age'].mean())   #or value='bfill'||'ffill'


# In[ ]:


df['Embarked'].unique()


# In[ ]:


df_test['Embarked'].unique()


# In[ ]:


df['Embarked']=df['Embarked'].fillna(value=0)
df['Embarked']=df['Embarked'].replace({'S':1,'C':2,'Q':3})
df['Embarked'].head()


# In[ ]:


df_test['Embarked']=df_test['Embarked'].fillna(value=0)
df_test['Embarked']=df_test['Embarked'].replace({'S':1,'C':2,'Q':3})
df_test['Embarked'].head()


# In[ ]:


df_test['Fare']=df_test['Fare'].fillna(value=df_test['Fare'].mean())


# In[ ]:


df.isnull().sum()


# In[ ]:


df_test.isnull().sum()


# # Correlation Matrix and heatmap

# In[ ]:


mat=df.corr()
mat


# In[ ]:


sns.heatmap(mat)


# In[ ]:


df.drop(columns=['Name','Ticket'],inplace=True)
df.head()


# In[ ]:


df_test.drop(columns=['Name','Ticket'],inplace=True)
df_test.head()


# # Splitting the train data into train and test

# In[ ]:


df_pred=df.drop(columns=['Survived'])
df_out=df['Survived']


# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(df_pred,df_out,test_size=0.2,random_state=42)


# # Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)


# In[ ]:


model.score(x_train,y_train)*100


# In[ ]:


model.score(x_test,y_test)*100


# # Checking for test dataset

# In[ ]:


temp=pd.read_csv('../input/gender_submission.csv')
temp.head()


# # Decision Tree

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
model_decision_tree=DecisionTreeClassifier(min_impurity_decrease=0.005,min_samples_split=10)
model_decision_tree.fit(x_train,y_train)


# In[ ]:


model_decision_tree.score(x_train,y_train)*100


# In[ ]:


model_decision_tree.score(x_test,y_test)*100


# In[ ]:


k=model_decision_tree.predict(df_test)


# In[ ]:


test_Survived = pd.Series(k, name="Survived")
Submission = pd.concat([df_test.PassengerId,test_Survived],axis=1)
Submission.to_csv('submission_Shivam.csv', index=False)


# # Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
model_random_forest=RandomForestClassifier(n_estimators=10,min_impurity_decrease=0.05,min_samples_split=20)
model_random_forest.fit(x_train,y_train)


# In[ ]:


model_random_forest.score(x_test,y_test)*100


# # SVC

# In[ ]:


from sklearn.svm import SVC
model=SVC(kernel='rbf')
model.fit(x_train,y_train)


# In[ ]:


model.score(x_test,y_test)*100


# # NEURAL NETWORK

# In[ ]:


from sklearn.neural_network import MLPClassifier
model_nn=MLPClassifier()
model_nn.fit(x_train,y_train)


# In[ ]:


model_nn.score(x_test,y_test)*100


# In[ ]:




