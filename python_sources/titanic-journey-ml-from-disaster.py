#!/usr/bin/env python
# coding: utf-8

# # Data Exploration

# In[ ]:


#data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd


# # Acquire Data
# 
# We need to start by acquiring the training and testing dataset.

# In[ ]:


#Reading train dataset
train_df   = pd.read_csv('/kaggle/input/titanic/train.csv')
test_df    = pd.read_csv('/kaggle/input/titanic/test.csv')

#previewing the dataset
train_df.head()
test_df.head()


# In[ ]:


#Knowing about the data
train_df.info()
print('----------------------')
test_df.info()


# # Quite a lot information
# 
# * Survival = Survival
# * Pclass = Ticket class
# * Sex = Sex
# * Age = Age in years
# * Sibsp = # of siblings / spouses aboard the Titanic
# * Parch = # of parents / children aboard the Titanic
# * Ticket = Ticket number
# * Fare = Passenger fare
# * Cabin = Cabin number
# * Embarked = Port of Embarkation
# 

# In[ ]:


#Checking for missing values in dataset
train_df.isnull().sum()


# ** 'Age', 'Cabin' and 'Embarked'** contain missing values, so we cannot use these variables for prediction without dealing with the missing variables.
# 

# In[ ]:


#Analyzing the dataset
train_df.describe()


# In[ ]:


# Load our plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Countplot for 'Survived' variable
sns.countplot(train_df['Survived'])


# In[ ]:


sns.countplot(x = 'Survived', hue = 'Sex', data = train_df)


#  **'Sex'** looks like a strong explanatory variable, and we can use it for our single feature Logistic Regression model!
# 

# # **First Simple Model**
# 
# 

# In[ ]:


# Import LogisticRegression
from sklearn.linear_model import LogisticRegression

# Fit a logistic regression model to our train data, by converting 'Sex' to a dummy variable, to feed it into the model.
logisticRegression = LogisticRegression()
logisticRegression.fit(X = pd.get_dummies(train_df['Sex']), y = train_df['Survived'])


# In[ ]:


# Predict!
test_df['Survived'] = logisticRegression.predict(pd.get_dummies(test_df['Sex']))


# # First Submission

# In[ ]:


# Write test predictions for final submission
test_df[['PassengerId', 'Survived']].to_csv('kaggle_submission.csv', index = False)


# In[ ]:




