#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression Project
# ## Tracy(11/14/2019)_Titanic
# 
# This is my project about Logistic Regression. I will predict a classification- survival or deceased.

# ## Import Libraries and Data

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train = pd.read_csv('/kaggle/input/titanic-machine-learning-from-disaster/train.csv')


# In[ ]:


train.head()


# ## Exploratory Data Analysis

# In[ ]:


## Use seaborn to create a simple heatmap to see where we are missing data.
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')


# Roughly 20 percent of the Age data is missing. The proportion of Age missing is likely small enough for reasonable replacement with some form of imputation. Looking at the Cabin column, it looks like we are just missing too much of that data to do something useful with at a basic level. We'll probably drop this later, or change it to another feature like "Cabin Known: 1 or 0"

# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='Survived', data=train, palette='RdBu')


# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='Survived', hue='Sex', data=train, palette='RdBu')


# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='Survived', hue='Pclass', data=train, palette='rainbow')


# In[ ]:


sns.distplot(train['Age'].dropna(), kde=False, color='darkred', bins=30)


# In[ ]:


train['Age'].hist(bins=30, color='darkred', alpha=0.7)


# In[ ]:


sns.countplot(x='SibSp', data=train)


# In[ ]:


train['Fare'].hist(color='green', bins=40, figsize=(8,4))


# ## Data Cleaning
# I want to fill in missing age data instead of just dropping the missing age data rows. One way to do this is by filling in the mean age of all the passengers (imputation). However we can be smarter about this and check the average age by passenger class.

# In[ ]:


plt.figure(figsize=(12,7))
sns.boxplot(x='Pclass', y='Age', data=train, palette='winter')


# We can see the wealthier passengers in the higher classes tend to be older, which makes sense. We'll use these average age values to impute based on Pclass for Age.

# In[ ]:


def impute_age(cols):
    Age=cols[0]
    Pclass=cols[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age


# In[ ]:


train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)


# In[ ]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:


## Drop the Cabin column and the row in Embarked that is NaN.
train.drop('Cabin', axis=1, inplace=True)


# In[ ]:


train.head()


# In[ ]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# ## Convert Categorical Features
# I will convert categorical features to dummy variables using pandas.
# Otherwise machine learning algorithims won't be able to directly take in those features as inputs.

# In[ ]:


train.info()


# In[ ]:


sex = pd.get_dummies(train['Sex'], drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)


# In[ ]:


train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)


# In[ ]:


train = pd.concat([train,sex,embark],axis=1)


# In[ ]:


train.head()


# ## Build a Logistic Regression Model

# In[ ]:


## Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1),train['Survived'], test_size=0.25, random_state=101)


# In[ ]:


## Training and Predicting
from sklearn.linear_model import LogisticRegression
lm = LogisticRegression()
lm.fit(X_train, y_train)


# In[ ]:


predictions = lm.predict(X_test)


# ## Evaluation

# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))

