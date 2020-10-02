#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import all libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as mp
import seaborn as sns


# In[ ]:


# Read the train and test datasets
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')


# In[ ]:


# Quick summary of the train dataset
train['group']='train'
train.describe(include='all')


# In[ ]:


# Quick summary of the test dataset
test['group']='test'
test.describe(include='all')


# In[ ]:


# convert some data types
train.info()
train[['PassengerId','Survived','Pclass']] = train[['PassengerId','Survived','Pclass']].astype('object')
test[['PassengerId','Pclass']] = test[['PassengerId','Pclass']].astype('object')


# In[ ]:


# Merge the train and test datasets together
combined = pd.concat([train,test])
combined.columns = combined.columns.str.lower()


# In[ ]:


# Handling missing values
combined.isnull().sum()


# In[ ]:


# Before we start to handle missing data we need to have a feel of the dataset
# Exploratory data analysis
sns.pairplot(combined[['age','sibsp','parch','fare']])


# In[ ]:


sns.distplot(train[train.Age.notnull()].Age)


# In[ ]:


# Survived versus age
g = sns.FacetGrid(combined,col= 'survived')
g.map(sns.distplot,'age')


# In[ ]:


age_data = combined[combined.age.notnull()]
g = sns.FacetGrid(age_data,col='survived')
g.map(sns.distplot,'age')


# In[ ]:


sns.countplot(x='Sex',data=train)


# In[ ]:


sns.countplot(x='Sex',hue = 'Survived',data=train)


# In[ ]:


sns.violinplot(x='Age',y='Sex',hue='Survived', data=train)


# In[ ]:


sns.boxplot(x='Age',y='Sex',hue='Survived', data=train)


# In[ ]:


sns.countplot(x='SibSp',hue='Survived',data=train)


# In[ ]:


sns.catplot(x="Sex", y="Survived", hue="SibSp", kind="bar", data=train);


# In[ ]:


sns.countplot(x='Parch',hue='Survived',data=train)


# In[ ]:


sns.factorplot(x="Pclass", y="Fare", hue="Survived", data=train)


# In[ ]:


#Feature engineering
train['Title'] = train.Name.str.extract(' ([A-Za-z]+)\.')
sns.countplot(x='Title',hue='Survived',data=train)

