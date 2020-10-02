#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing , for importing csv files
import matplotlib.pyplot as plt  # data visualization
import seaborn as sns   # data visualization

import os
print(os.listdir("../input"))


# In[ ]:


# importing the dataset
train_data=pd.read_csv("../input/train.csv")


# In[ ]:


train_data.head()


# In[ ]:


# descriptive Statistics
train_data.describe()


# # Data Analysis

# In[ ]:


train_data[['Pclass','Survived']].groupby(['Pclass']).mean()


# In[ ]:


train_data[['Sex','Survived']].groupby(['Sex']).mean()


# In[ ]:


# checking survival with Sex
sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train_data,hue='Sex' )


# In[ ]:


# checking survival with our Pclass
sns.countplot(x='Survived',data=train_data,hue='Pclass')


# # data Cleaning

# In[ ]:


train_data['Cabin'].isnull().value_counts()


# In[ ]:


# droping name, ticket and Cabin column from table
train_data.drop(['Name','Ticket','Cabin'],inplace=True,axis=1)


# In[ ]:


# checking number of passengers with unknown Age
train_data['Age'].isnull().value_counts()


# In[ ]:


# checking for null values in data set
sns.heatmap(train_data.isnull(),yticklabels=False,cbar=False)


# In[ ]:


sns.barplot(x='Pclass',y='Age',data=train_data)


# In[ ]:


# function to fill unknown ages according to class of passenger
def age_fill(cols):
    Age=cols[0]
    Pclass=cols[1]
    
    if pd.isnull(Age):
        
        if Pclass==1:
            return 37
        elif Pclass==2:
            return 29
        else:
            return 24
    else:
        return Age


# In[ ]:


# applying age_fill() function to data set
train_data['Age']=train_data[['Age','Pclass']].apply(age_fill,axis=1)


# In[ ]:


# checking data set
train_data.head(10)


# In[ ]:


# converting Categorial variable Sex to binary variable.
train_data['Sex']=train_data['Sex'].apply(lambda x : 1 if x=='male' else 0 )


# In[ ]:


# converting Categorial variable Embarked to two binary variable.
# pd.get_dummies returns data table with number of columns according to number of categories.

Embarked=pd.get_dummies(train_data['Embarked'],drop_first=True)


# In[ ]:


# removing Embarked column from data set
train_data.drop('Embarked',axis=1,inplace=True)


# In[ ]:


# adding binary colums from Embarked cloumn to our data set
train_data=pd.concat([train_data,Embarked],axis=1)


# In[ ]:


# checking the head of data set
train_data.head()


# # Model training and Predictions

# In[ ]:


# used logistic regression 
from sklearn.linear_model import LogisticRegression


# In[ ]:


model=LogisticRegression() 


# In[ ]:


# here X is the input
# y is output corresponding to input in X.
X=data.drop('Survived',axis=1)
y=data['Survived']


# In[ ]:


# fitting data to our model
model.fit(X,y)


# In[ ]:


# importing testing data
test_data=pd.read_csv('../input/test.csv')


# In[ ]:


test_data.head()


# In[ ]:


# cleaning test data as we have done for training data
test_data.drop(['Name','Ticket','Cabin'],axis=1,inplace=True)


# In[ ]:


test_data['Sex']=test_data['Sex'].apply(lambda x : 1 if x=='male' else 0 )


# In[ ]:


sns.heatmap(test_data.isnull(),cbar=False)


# In[ ]:


test_data['Age']=test_data[['Age','Pclass']].apply(age_fill,axis=1)


# In[ ]:


test_data.head()


# In[ ]:


Embarked=pd.get_dummies(test_data['Embarked'],drop_first=True)  


# In[ ]:


test_data.drop('Embarked',axis=1,inplace=True)


# In[ ]:


test_data=pd.concat([test_data,Embarked],axis=1)


# In[ ]:


test_data["Fare"]=test_data["Fare"].fillna(value=test_data['Fare'].mean()) 


# In[ ]:


# calculating the predictions
predictions=model.predict(test_data)


# In[ ]:


# creating table for result
passen_id=test_data['PassengerId']


# In[ ]:


predict_=pd.Series(data=predictions)


# In[ ]:


result=pd.concat([passen_id,predict_],axis=1)


# In[ ]:


result.to_csv('my_submission')


# In[ ]:


result.head(10)

