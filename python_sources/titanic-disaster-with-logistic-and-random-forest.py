#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Importing libraries for Data Manipulation and Visualization

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Modelling Algorithms
from sklearn.ensemble import RandomForestClassifier 

# Modelling Helpers
#from sklearn.preprocessing import  Normalizer , scale
from sklearn.model_selection import train_test_split , StratifiedKFold
from sklearn.feature_selection import RFECV

#Modelling Metrics
#from sklearn.metrics import r2_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# In[ ]:


#Reading the CSV
df=pd.read_csv("/kaggle/input/titanic/train.csv")


# In[ ]:


#Checking the first few rows
df.head()


# In[ ]:


#Checking null values
df.isnull().sum()


# In[ ]:


df.describe()


# In[ ]:


df.pivot_table('Survived', index='Age').plot()


# In[ ]:


plt.figure(figsize=(8, 6))
sns.countplot(x='Survived',data=df)
plt.title("Survived",fontsize=30)
plt.ylabel("Count",fontsize=20)


# In[ ]:


df['Cabin'].head()


# In[ ]:


#Dropping Cabin Column
df.drop('Cabin',axis=1,inplace=True)


# In[ ]:


df.head()


# In[ ]:


#Creating a new variable while concatenating other 2 columns
df['Family']=df['SibSp']+df['Parch']


# In[ ]:


#Checking the correlation
sns.heatmap(df.corr(),annot=True)


# In[ ]:


#Dropping few columns
df.drop(['Name','Ticket','Fare','Parch','SibSp'],axis=1,inplace=True)


# In[ ]:


df.plot(by='Age')


# In[ ]:


df.info()


# In[ ]:


df.isnull().sum()


# In[ ]:


df.shape


# In[ ]:


df.drop(df[df['Embarked'].isnull()].index,inplace=True)


# In[ ]:


df.isnull().sum()


# In[ ]:


#Getting the mean age to replace null values in Age column
age_mean=df['Age'].mean()


# In[ ]:


print(age_mean)


# In[ ]:


df['Age'].fillna(age_mean,inplace=True)


# In[ ]:


df.info()


# In[ ]:


df.isnull().sum()


# In[ ]:


Sex=pd.get_dummies(df['Sex'],drop_first=True)


# In[ ]:


Emb=pd.get_dummies(df['Embarked'],drop_first=True)


# In[ ]:


Emb.head()


# In[ ]:


df1=df


# In[ ]:


df1=pd.concat([df,Sex,Emb],axis=1)


# In[ ]:


df1.head()


# In[ ]:


df1.drop(['Sex','Embarked'],axis=1,inplace=True)


# In[ ]:


X=df1[df1.loc[:,df1.columns!='Survived'].columns]
y=df1['Survived']


# In[ ]:


print (X.dtypes )


# In[ ]:


#Splitting the data into Training and Test 
X_train, X_test, y_train, y_test=train_test_split(X ,y, test_size=.2, random_state=10)


# In[ ]:


modelRF=RandomForestClassifier(n_estimators=100)


# In[ ]:


modelRF.fit(X_train,y_train )


# In[ ]:


print (modelRF.score(X_train,y_train), modelRF.score(X_test,y_test))


# In[ ]:


rfecvRF = RFECV( estimator = modelRF , step = 1 , cv = 5, scoring = 'accuracy', )
rfecvRF.fit( X_train,y_train)


# In[ ]:


y_predRF=rfecvRF.predict(X_test)


# In[ ]:


#Getting the accuracy Score
accuracy_score(y_test,y_predRF)


# In[ ]:


confusion_matrix(y_test,y_predRF)


# In[ ]:


#Getting the Test Data for Titanic

testdf=df=pd.read_csv("/kaggle/input/titanic/test.csv")


# In[ ]:


#Manipulating Data to match the Features
testdf['Family']=testdf['SibSp']+testdf['Parch']


# In[ ]:


Sex=pd.get_dummies(testdf['Sex'],drop_first=True)


# In[ ]:


Emb=pd.get_dummies(testdf['Embarked'],drop_first=True)


# In[ ]:


testdf1=testdf


# In[ ]:


testdf1=pd.concat([testdf,Sex,Emb],axis=1)


# In[ ]:


testdf1.head()


# In[ ]:


testdf1.drop(['SibSp','Parch','Sex','Embarked','Cabin','Name','Ticket','Fare'],axis=1,inplace=True)


# In[ ]:


testdf1.head()


# In[ ]:


testdf1.info()


# In[ ]:


age_mean=testdf1['Age'].mean()


# In[ ]:


testdf1['Age'].fillna(age_mean,inplace=True)


# In[ ]:


X_train.head()


# In[ ]:


X_Testnew=testdf1


# In[ ]:


y_Testnew=rfecvRF.predict(X_Testnew)


# In[ ]:


y_Testnew.shape


# In[ ]:


submissionnew = pd.DataFrame({'PassengerId':X_Testnew['PassengerId'],'Survived':y_Testnew})


# In[ ]:


filename = 'Titanic_Predictionsnew.csv'

submissionnew.to_csv(filename,index=False)

print('Saved file: ' + filename)

