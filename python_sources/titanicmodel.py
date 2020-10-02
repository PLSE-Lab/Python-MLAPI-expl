#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# In[ ]:


# loading the data 
df=pd.read_csv('../input/train.csv')
df.head(10)


# In[ ]:


## Data Analysis
df.hist('Survived');


# In[ ]:


## summary of the data 
df.info(verbose= False)


# In[ ]:


df["Sex"].value_counts()
## we can notice than the number of male is greater than female


# In[ ]:


sns.countplot(x = "Sex", hue ="Survived",data = df, palette = "Blues");


# In[ ]:


sns.countplot(x = "Pclass", hue ="Survived",data = df, palette = "Blues");


# In[ ]:


sns.countplot(x = "Pclass", hue ="Sex",data = df, palette = "Greens");


# In[ ]:


sns.barplot(x = "Embarked", y = "Survived", data = df);


# In[ ]:


labels = ["Pclass","Sex","Age","Fare","Parch","SibSp","Embarked"]
y=df['Survived']
df=df[labels]


# In[ ]:


df.isnull().sum()


# In[ ]:


## cleaning the data by drawing the histogram of the indepedant variable (Age)
p=df.hist('Age')


# In[ ]:


df['Embarked'].value_counts()


# In[ ]:


df['Age'].fillna(df['Age'].median(),inplace=True)


# In[ ]:


df['Embarked'].fillna(df['Embarked'].value_counts().index[0], inplace=True)


# In[ ]:


p = {1:'1st',2:'2nd',3:'3rd'}


# In[ ]:


df['Pclass'] = df['Pclass'].map(p)


# In[ ]:


categorical_df = df[['Pclass','Sex','Embarked']]
one_hot_encode = pd.get_dummies(categorical_df,drop_first=True) 
df = df.drop(['Pclass','Sex','Embarked'],axis=1)
df = pd.concat([df,one_hot_encode],axis=1)


# In[ ]:


X = df
train_X, test_X, train_y, test_y = train_test_split(X,y,random_state = 0)


# In[ ]:


clf = DecisionTreeClassifier()
clf.fit(train_X,train_y)


# In[ ]:


pred = clf.predict(test_X)


# In[ ]:


from sklearn import metrics
k=metrics.accuracy_score(test_y, pred)
print("the score is:",k)


# In[ ]:




