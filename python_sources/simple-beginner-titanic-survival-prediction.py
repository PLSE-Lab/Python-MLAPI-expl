#!/usr/bin/env python
# coding: utf-8

# ## Simple Beginner kernel using logistic regression

# ### Imports

# In[ ]:


import pandas as pd 
import numpy as np


# In[ ]:


train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")


# In[ ]:


train.head()


# In[ ]:


train.drop(['Name'],axis=1,inplace=True)


# In[ ]:


train.head()


# In[ ]:


train.columns


# In[ ]:


train.index


# In[ ]:


sample_sub=pd.read_csv("../input/gender_submission.csv")


# In[ ]:


sample_sub.head()


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


test.drop(['Name'],axis=1,inplace=True)


# In[ ]:


test.head()


# ### Checking for null values

# In[ ]:


train.isnull().sum()


# In[ ]:


train.index


# In[ ]:


test.isnull().sum()


# ### Dropping columns with too much null values and non required data

# In[ ]:


train.drop(['Cabin'],axis=1,inplace=True)
test.drop(['Cabin'],axis=1,inplace=True)
test.drop(['Ticket'],axis=1,inplace=True)
train.drop(['Ticket'],axis=1,inplace=True)
train.drop(['PassengerId'],axis=1,inplace=True)
test.drop(['PassengerId'],axis=1,inplace=True)


# In[ ]:


test.head()


# In[ ]:


train.head()


# ### Visualization

# In[ ]:


import seaborn as sns
sns.countplot(x='Survived',hue='Sex',data=train)


# In[ ]:


sns.countplot(x='Survived',hue='Parch',data=train)


# In[ ]:


sns.countplot(x='Survived',hue='SibSp',data=train)


# In[ ]:


sns.countplot(x='Survived',hue='Pclass',data=train)


# ### transformation

# In[ ]:


train.isnull().sum()


# In[ ]:


test.isnull().sum()


# In[ ]:


train['Age'].mean()


# ### Replacing null values by mean of the vaues of column data

# In[ ]:


train['Age'].fillna((train['Age'].mean()), inplace=True)


# In[ ]:


test['Age'].fillna((test['Age'].mean()), inplace=True)


# In[ ]:


test['Fare'].fillna((test['Fare'].mean()), inplace=True)


# #### Dropping null values from train

# In[ ]:


train.dropna()


# In[ ]:


test.isnull().sum()


# In[ ]:


train.columns


# In[ ]:


train.head()


# In[ ]:


test.head()


# ### Transforming instead of skewing

# In[ ]:


Pclass=pd.get_dummies(train['Pclass'],drop_first=True)
Pclass1=pd.get_dummies(test['Pclass'],drop_first=True)
Sex=pd.get_dummies(train['Sex'],drop_first=True)
Sex1=pd.get_dummies(test['Sex'],drop_first=True)
Embarked=pd.get_dummies(train['Embarked'],drop_first=True)
Embarked1=pd.get_dummies(test['Embarked'],drop_first=True)


# ### Joining newly created dummy data columns 

# In[ ]:


train=pd.concat([train,Pclass,Sex,Embarked],axis=1)
test=pd.concat([test,Pclass1,Sex1,Embarked1],axis=1)


# ### Dropping old columns

# In[ ]:


train.drop(['Sex','Embarked','Pclass'],axis=1,inplace=True)
test.drop(['Sex','Embarked','Pclass'],axis=1,inplace=True)


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


sample_sub.head()


# ## Modelling

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


y=train['Survived']


# In[ ]:


y.head()


# In[ ]:


X=train.drop('Survived',axis=1)


# In[ ]:


X.head()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)


# ### Fitting the data into model

# ##### Model 1

# In[ ]:


from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[ ]:


predictions = logmodel.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


classification_report(y_test,predictions)


# In[ ]:


from sklearn.metrics import confusion_matrix


# In[ ]:


confusion_matrix(y_test,predictions)


# In[ ]:


(145+67)/(41+67+15+145)


# ##### Model 2

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
gbk = GradientBoostingClassifier()
gbk.fit(X_train, y_train)
y_pred = gbk.predict(X_test)
acc_gbk = round(accuracy_score(y_pred, y_test) * 100, 2)
print(acc_gbk)


# approx 79% accuracy

# ## Predicting 

# In[ ]:


test.head()


# In[ ]:


test.isnull().sum()


# In[ ]:


predictions1 = gbk.predict(test)
sample_sub['Survived']= predictions1
sample_sub.to_csv("submit.csv", index=False)
sample_sub.head()


# ### Please comment how to improve the accuracy i am new 
