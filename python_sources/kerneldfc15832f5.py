#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# Load data in data frame in pandas

# In[ ]:


df_train = pd.read_csv('/kaggle/input/titanic/train.csv')
df_test = pd.read_csv('/kaggle/input/titanic/test.csv')
#preview data
print("-"*20)
#df.info()
# check null value
print("Null values in train data\n",df_train.isnull().sum())
print("Null values in test data\n",df_test.isnull().sum())


# **Data Cleaning**
# Survied

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(18,8))
print(df_train['Survived'].value_counts())
df_train['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Survived')
ax[0].set_ylabel('')
sns.countplot('Survived',data=df_train,ax=ax[1])
ax[1].set_title('Survived')
plt.show()


# out of 891, only 342 were survived i.e. 38.4% of total of train data set

# **Sex**

# In[ ]:


#S,sx=plt.subplots(1,2,figsize=(18,8))
sx=sns.countplot('Sex',hue='Survived',data=df_train)
sx.set_title('Sex vs Survived')


# Number of males survival is less as compared to female number. (women and children first policy)

# In[ ]:


df_train.head(10)


# Class vs Survived

# In[ ]:


sns.countplot('Survived',hue='Pclass',data=df_train)


# Class 3 has less survival rate as compared to Class 1 and Class 2

# 1. filling null values in Age column and Droping Cabin i.e. null values Column

# In[ ]:


plt.figure(figsize=(12,7))
sns.boxplot(x='Pclass',y='Age',data=df_train)


# In[ ]:


def imput_age(cols):
    age=cols[0]
    pclass=cols[1]
    if pd.isnull(age):
        if pclass==1:
            return 38
        if pclass==2:
            return 29
        if pclass==3:
            return 25
    else:
        return age

            
        


# In[ ]:


df_train['Age']=df_train[['Age','Pclass']].apply(imput_age,axis=1)
df_train['Age'].isnull().sum()
df_train.drop(['Cabin'],axis=1,inplace=True)


# In[ ]:


df_train.head()


# **Feature Engineering **

# In[ ]:


sex_dummy=df_train['Sex'].str.get_dummies()
sex_dummy.head()


# In[ ]:


Emb_dummy=df_train['Embarked'].str.get_dummies()
Emb_dummy.head()
df_train.drop(['Sex','Pclass','Name','Ticket','Embarked'],axis=1,inplace=True)


# In[ ]:



df_train=pd.concat([df_train,sex_dummy,Emb_dummy],axis=1)
df_train.head()
                


# **Building Model******

# In[ ]:


df_train.drop('Survived',axis=1)


# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(df_train.drop('Survived',axis=1),df_train['Survived'],test_size=0.30,
                                               random_state=101)


# **Using Logistic Regression**

# In[ ]:


from sklearn.linear_model import LogisticRegression # for logistic regression
from sklearn.ensemble import RandomForestClassifier  # for Random Forest
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
logmodel=LogisticRegression()
logmodel=RandomForestClassifier(n_estimators=100)
#logmodel= KNeighborsClassifier()
#logmodel= svm.SVC(kernel='rbf',C=1,gamma=0.1)
logmodel.fit(x_train,y_train)


# In[ ]:


predication=logmodel.predict(x_test)


# In[ ]:


from sklearn.metrics import confusion_matrix


# In[ ]:


accuracy=confusion_matrix(y_test,predication)


# In[ ]:


accuracy


# In[ ]:


from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,predication)
accuracy


# In[ ]:


predication


# In[ ]:


df_test1=df_test.copy()
#df_test1.head()
#df_test1.info()
df_test1['Age']=df_test1['Age'].fillna(df_test['Age'].mean())
df_test1['Fare']=df_test1['Fare'].fillna(df_test['Fare'].mean())
df_test1.drop(columns=['Name','Ticket','PassengerId','Cabin'],inplace=True)


# In[ ]:



df_test1=pd.get_dummies(data=df_test1,columns=['Sex','Embarked'])


# In[ ]:


df_test1.head()
y_pred=logmodel.predict(df_test1)


# In[ ]:


submission=pd.DataFrame({'PassengerId':df_test['PassengerId'],'Survived':y_pred})


# In[ ]:


submission.to_csv('submission.csv',index=False)

