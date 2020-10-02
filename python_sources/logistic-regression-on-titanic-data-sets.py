#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import math
get_ipython().run_line_magic('matplotlib', 'inline')
import os

data= pd.read_csv('../input/titanic123/ChanDarren_RaiTaran_Lab2a.csv')
data


# Analyzing Data

# In[ ]:


sns.countplot(x='Survived' , data=data)


# In[ ]:


sns.countplot(x='Survived' , hue='Sex' , data=data)


# In[ ]:


sns.countplot(x='Survived' , hue='Pclass' , data=data)


# In[ ]:


data['Age'].plot.hist()


# In[ ]:


data['Fare'].plot.hist()


# In[ ]:


data['Fare'].plot.hist(bins= 10 , figsize=(10,5))


# In[ ]:


data.info()


# In[ ]:


sns.countplot(x='SibSp',data=data)


# Data Wrangling

# In[ ]:


data.isnull().sum()


# In[ ]:


sns.heatmap(data.isnull(),yticklabels=False)


# In[ ]:


sns.heatmap(data.isnull(),yticklabels=False, cmap='viridis')


# In[ ]:


sns.boxplot(x='Pclass',y='Age',data=data)


# In[ ]:


data.drop('Cabin',axis=1,inplace=True)


# In[ ]:


data.head(5)


# In[ ]:


data.dropna(inplace=True)


# In[ ]:


sns.heatmap(data.isnull(),yticklabels=False,cbar=False)


# In[ ]:


data.isnull().sum()


# In[ ]:


data.head(2)


# In[ ]:


sex=pd.get_dummies(data['Sex'], drop_first=True)
sex.head(2)


# In[ ]:


embark=pd.get_dummies(data['Embarked'],drop_first=True)
embark.head(5)


# In[ ]:


pcl=pd.get_dummies(data['Pclass'],drop_first=True)
pcl.head(5)


# In[ ]:


data=pd.concat([data,sex,embark,pcl],axis=1)
data.head(5)


# In[ ]:


data.drop(['Sex','Embarked','PassengerId','Name','Ticket','Pclass'],axis=1,inplace=True)


# In[ ]:


data.head(5)


# Data Spliting
# 

# In[ ]:


x=data.drop('Survived',axis=1)
y=data['Survived']


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


logmodel=LogisticRegression()


# In[ ]:


logmodel.fit(X_train,y_train)


# In[ ]:


Predictions = logmodel.predict(X_test)
Predictions


# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


classification_report(y_test,Predictions)


# In[ ]:


from sklearn.metrics import confusion_matrix


# In[ ]:


confusion_matrix(y_test , Predictions)


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


accuracy_score(y_test , Predictions)


# In[ ]:




