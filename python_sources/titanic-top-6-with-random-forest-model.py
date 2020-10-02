#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


train=pd.read_csv("../input/titanic/train.csv")


# In[ ]:


test=pd.read_csv('../input/titanic/test.csv')


# In[ ]:


corr=train.corr()
top_co=corr.index
plt.figure(figsize=(20,20))
g=sns.heatmap(train[top_co].corr(),annot=True,cmap="RdYlGn")


# In[ ]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:


train['Age']=train['Age'].median()


# In[ ]:


train.drop(['PassengerId', 'Ticket'], axis=1, inplace=True)
test.drop(['PassengerId', 'Ticket'], axis=1, inplace=True)


# In[ ]:


train.isna().sum()


# In[ ]:


train['Embarked'].mode()


# In[ ]:





# In[ ]:


test.isna().sum()


# In[ ]:


test['Age']=test['Age'].median()


# In[ ]:


train.drop(['Cabin'], axis=1, inplace=True)
test.drop(['Cabin'], axis=1, inplace=True)


# In[ ]:


# Get Title from Name for train
train_title = [i.split(",")[1].split(".")[0].strip() for i in train["Name"]]
train["Title"] = pd.Series(train_title)
train["Title"].head()


# In[ ]:


g = sns.countplot(x="Title",data=train)
g = plt.setp(g.get_xticklabels(), rotation=45) 


# In[ ]:


#Convert to categorical values Title 
train["Title"] = train["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
train["Title"] = train["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})
train["Title"] = train["Title"].astype(int)


# In[ ]:


# Get Title from Name for train
test_title = [i.split(",")[1].split(".")[0].strip() for i in test["Name"]]
test["Title"] = pd.Series(test_title)
test["Title"].head()


# In[ ]:


g = sns.countplot(x="Title",data=test)
g = plt.setp(g.get_xticklabels(), rotation=45) 


# In[ ]:


#Convert to categorical values Title 
test["Title"] = test["Title"].replace(['Col', 'Dr', 'Rev', 'Dona'], 'Rare')
test["Title"] = test["Title"].map({"Master":0, "Miss":1, "Ms" : 1  ,"Mrs":1, "Mr":2, "Rare":3})
test["Title"] = test["Title"].astype(int)


# In[ ]:


train.isna().sum()


# In[ ]:


test.isna().sum()


# In[ ]:


train['Embarked'].fillna('S',inplace=True)


# In[ ]:


test.Fare = test['Fare'].fillna(train['Fare'].median())


# In[ ]:


train=train.drop('Name',axis=1)


# In[ ]:


test=test.drop('Name',axis=1)


# Creating Family Size for train and test data

# In[ ]:


train['FamilySize'] = train['Parch'] + train['SibSp'] + 1 


# In[ ]:


test['FamilySize'] = test['Parch'] + test['SibSp'] + 1 


# In[ ]:


# drop the variable 'SibSp' as we have already created a similar variable FamilySize
train = train.drop(['SibSp'], axis = 1)
test  = test.drop(['SibSp'], axis = 1)


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train['Sex']=train["Sex"].replace({'male':0,'female':1})
test['Sex']=test["Sex"].replace({'male':0,'female':1})


# In[ ]:


train['Embarked']=train["Embarked"].replace({'C':0,'S':1,'Q':3})


# In[ ]:


test['Embarked']=test["Embarked"].replace({'C':0,'S':1,'Q':3})


# In[ ]:


# seperate the feature set and the target set
Y_train = train["Survived"]
X_train = train.drop(labels = ["Survived"],axis = 1)
X_test = test


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# #model Creation 

# In[ ]:


model = RandomForestClassifier(n_estimators=250, min_samples_leaf=4, n_jobs=-1)
model.fit(X_train, Y_train)
model.score(X_train, Y_train)


# In[ ]:


X_test=test
y_predi=model.predict(X_test)


# In[ ]:


gn=pd.read_csv('../input/titanic/gender_submission.csv')


# In[ ]:


Y_test=gn['Survived'].values


# In[ ]:


accuracy_score(Y_test,y_predi)


# In[ ]:


confusion_matrix(Y_test,y_predi)


# In[ ]:


test1=pd.read_csv('../input/titanic/test.csv')


# In[ ]:


data_to_submit = pd.DataFrame({
    'PassengerId':test1['PassengerId'],
    'Survived':y_predi
})
data_to_submit.to_csv('finalsub4.csv', index = False)


# In[ ]:


data_to_submit.head()


# In[ ]:


data_to_submit.count()


# In[ ]:




