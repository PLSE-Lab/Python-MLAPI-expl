#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# linear algebra
import numpy as np
# data processing
import pandas as pd
# data visualization
import matplotlib.pyplot as plt
import seaborn as sns
# Algorithms
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier


# In[ ]:


#loading dataset
train = pd.read_csv("/kaggle/input/titanic/train.csv")


# In[ ]:


#printing 5 rows of dataset train
train.head()
#print(train)


# In[ ]:


#total no of passengers
train.count()
#total 891 rows, 12 cols, 891 passengers and some values null


# In[ ]:


#male passengers
train[train['Sex'].str.match("male")].count()
#female passengers
train[train["Sex"].str.match("female")].count()


# In[ ]:


#no of people survived based on sex
sex_grouping = train.groupby('Sex').mean()
sex_grouping['Survived'].plot.bar()


# In[ ]:


#no of people survived based on sex and class
class_sex_grouping = train.groupby(['Sex','Pclass']).mean()
class_sex_grouping['Survived'].plot.pie()


# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='Survived', data=train, palette='RdBu_r')


# In[ ]:


sns.countplot(x='Survived', hue='Sex', data= train,palette='RdBu_r')


# In[ ]:


sns.countplot(x='Survived', hue='Pclass', data= train, palette='rainbow')


# In[ ]:


sns.distplot(train['Age'].dropna(),color='darkred',bins=30)


# In[ ]:


plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')


# In[ ]:


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age
    
train['Age'] = train[['Age', 'Pclass']].apply(impute_age, axis = 1)


# In[ ]:


def impute_cabin(col):
    Cabin = col[0]
    if type(Cabin) == str:
        return 1
    else:
        return 0
    
train['Cabin'] = train[['Cabin']].apply(impute_cabin, axis = 1)


# In[ ]:


#Let's work on a copy of our present dataset for further operations
dataset = train
sex = pd.get_dummies(dataset['Sex'],drop_first=True)
embark = pd.get_dummies(dataset['Embarked'],drop_first=True)
dataset.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
dataset = pd.concat([dataset,sex,embark],axis=1)


# In[ ]:


dataset.head()


# In[ ]:


#Train Test Split
X_train, X_test, y_train, y_test = train_test_split(dataset.drop('Survived',axis=1),dataset['Survived'], test_size=0.25,random_state=101)


# In[ ]:


regressor = LogisticRegression()
regressor.fit(X_train, y_train)
pred = regressor.predict(X_test)
print(accuracy_score(y_test, pred))

