#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split


# In[ ]:


df2 = pd.read_csv('../input/titanic/train_and_test2.csv')


# In[ ]:


col = df2.columns
col
df2 = df2[['Passengerid', 'Age', 'Fare', 'Sex', 'sibsp', 'Parch','Pclass', 'Embarked', '2urvived']]
df2.columns = ['PassengerId', 'Age', 'Fare', 'Sex', 'SibSp', 'Parch','Pclass', 'Embarked', 'Survived']


# In[ ]:


df = df2


# In[ ]:


df.head() 


# In[ ]:


df.isna().sum()


# In[ ]:


df.drop(['PassengerId','Embarked','Parch'],axis=1,inplace=True)


# In[ ]:


df


# In[ ]:


sex = pd.get_dummies(df['Sex'],drop_first=True).copy()
pclass = pd.get_dummies(df['Pclass'],drop_first=True).copy()
sibsp = pd.get_dummies(df['SibSp'],drop_first=True).copy()
age = df["Age"].copy()
age = age.interpolate()
fare = np.round(df['Fare'],2)
df.drop(['Pclass','Sex','Age','Fare'],inplace=True,axis=1)


# In[ ]:


for x in range(len(age)):
    if 0<=age[x] and age[x]<18:
        age[x] = 0
    elif 18<=age[x] and age[x]<25:
        age[x] = 1
    elif 25<=age[x] and age[x]<45:
        age[x] = 2
    elif 45<=age[x] and age[x]<55:
        age[x] = 3
    else:
        age[x] = 4

for x in range(len(fare)):
    if fare[x]<50:
        fare[x]=0
    elif fare[x]<100:
        fare[x]=0.5
    else:
        fare[x]=1


# In[ ]:


#sibsp,age,fare,sex,pclass
pclass.columns = ["2nd_class","3rd_class"]
sibsp.drop(8,inplace=True,axis=1)
sibsp.columns = ['siblings_1','siblings_2','siblings_3','siblings_4','siblings_5']

df = pd.concat([df,sex,pclass,sibsp,age,fare],axis=1)


# In[ ]:


df


# In[ ]:


df.shape


# In[ ]:


y = df["Survived"]
X = df.drop('Survived',axis=1)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=6)


# In[ ]:


logmodel = LogisticRegression()


# In[ ]:


logmodel.fit(X_train,y_train)


# In[ ]:


y_predictions = logmodel.predict(X_test)


# In[ ]:


result = classification_report(y_test,y_predictions)


# In[ ]:


result


# In[ ]:


confusion_matrix(y_test,y_predictions)


# In[ ]:


accuracy_score(y_test,y_predictions)


# In[ ]:




