#!/usr/bin/env python
# coding: utf-8

# ## Exercise notebook for the fourth session
# 
# This is the exercise notebook for the fourth session of the [Machine Learning workshop series at Harvey Mudd College](http://www.aashitak.com/ML-Workshops/). Please feel free to ask for help from the instructor and/or TAs.

# First we import python modules:

# In[1]:


import pandas as pd
import numpy as np
import re

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.simplefilter('ignore')


# In today's exercise, we will work with the [Titanic dataset from Kaggle](https://www.kaggle.com/c/titanic). The objective of this Kaggle competition is to predict whether a passenger survives or not given a number of features related to passengers' information such as gender, age, ticket class, etc. We are going to build a few classification models to predict whether a passenger survives. The `train.csv` file contains features along with the information about the survival of the passenger, so we will use it to train and validate our models. The `test.csv` file contains only features and we will use one of our trained models to predict the survival for these passengers and [submit our predictions to the competitions leaderboard](https://www.kaggle.com/c/titanic/submit).

# For your convenience, the data preprocessing and feature engineering that we did in the previous sessions is summarized below.

# In[2]:


path = '../input/'
df = pd.read_csv(path + 'train.csv')
train = pd.read_csv(path + 'train.csv')
target = train.Survived.astype('category', ordered=False)
train.drop('Survived', axis=1)

test = pd.read_csv(path + 'test.csv')
PassengerId = test.PassengerId

def get_Titles(df):
    df.Name = df.Name.apply(lambda name: re.findall("\s\S+[.]\s", name)[0].strip())
    df = df.rename(columns = {'Name': 'Title'})
    df.Title.replace({'Ms.': 'Miss.', 'Mlle.': 'Miss.', 'Dr.': 'Rare', 'Mme.': 'Mr.', 'Major.': 'Rare', 'Lady.': 'Rare', 'Sir.': 'Rare', 'Col.': 'Rare', 'Capt.': 'Rare', 'Countess.': 'Rare', 'Jonkheer.': 'Rare', 'Dona.': 'Rare', 'Don.': 'Rare', 'Rev.': 'Rare'}, inplace=True)
    return df

def fill_Age(df):
    df.Age = df.Age.fillna(df.groupby("Title").Age.transform("median"))
    return df

def get_Group_size(df):
    Ticket_counts = df.Ticket.value_counts()
    df['Ticket_counts'] = df.Ticket.apply(lambda x: Ticket_counts[x])
    df['Family_size'] = df['SibSp'] + df['Parch'] + 1
    df['Group_size'] = df[['Family_size', 'Ticket_counts']].max(axis=1)
    return df

def process_features(df):
    df.Sex = df.Sex.astype('category', ordered=False).cat.codes
    features_to_keep = ['Age', 'Fare', 'Group_size', 'Pclass', 'Sex']
    df = df[features_to_keep]
    return df

def process_data(df):
    df = df.copy()
    df = get_Titles(df)
    df = fill_Age(df)
    df = get_Group_size(df)
    df = process_features(df)
    medianFare = df['Fare'].median()
    df['Fare'] = df['Fare'].fillna(medianFare)
    return df

X_train, X_test = process_data(train), process_data(test)


# Please feel free to refer to the classification algorithms notebook for the code below.

# First, split the data into training and validation set using `train_test_split` and name the variables as `X_train, X_valid, y_train, y_valid `.

# In[ ]:


X_train, X_valid, y_train, y_valid = train_test_split(X_train, target, random_state=0)


# In[ ]:


X_train.head()


# In[ ]:


y_train.head()


# Train a logistic regression classifier on `X_train, y_train` and test its accuracy on both `X_train, y_train` and `X_valid, y_valid`.

# In[ ]:





# [The evaluation metric for this competition is accuracy](https://www.kaggle.com/c/titanic/overview/evaluation).

# Try training  a few more classifiers and compare the accuracy. Try tuning the hyperparameters too. You can also try more feature engineering by editing the code above.

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# Once you have explored a different classifiers and decided on one trained model (or a voting classifer ensemble as seen before), let us use it to make predictions using the features from `X_test` and save the results into `y_test`.

# In[ ]:





# We create a dataframe for submission using the predictions from `y_test` and save it to a csv file. It is important that our submission file is in correct format to be graded without errors.

# In[ ]:


submission = pd.DataFrame({'PassengerId': PassengerId, 'Survived': y_test})
submission.to_csv('submission.csv', index=False)


# In[ ]:




