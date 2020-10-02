#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


# Loda Data

# In[ ]:


train_df = pd.read_csv('../input/titanic/train.csv')
test_df = pd.read_csv('../input/titanic/test.csv')


# In[ ]:


train_df.head()


# It's quite obvious that features like Name, Ticket, Cabin and PassengerId doensn't contribute to predictions, hence let's drop these features

# In[ ]:


cols_to_drop = ['Name', 'Ticket', 'Cabin']
train_df = train_df.drop(cols_to_drop, axis=1)
test_df = test_df.drop(cols_to_drop, axis=1)


# We also don't need PassengerId in training data, hence we drop it. We need Passenger Id for submissions so we don't drop it from test data.

# In[ ]:


train_df = train_df.drop(['PassengerId'], axis=1)
train_df.head()


# Let's see if there are missing values in our dataset

# In[ ]:


train_df.describe()


# feature Age has some missing values which is 891-714 = 177, let's impute these missing values by grouping them by gender and Pclass first and then filling them with relevant mean. Feature Embarked also contains missing values let's drop these rows since it's only few rows.

# In[ ]:


train_df.groupby(['Pclass', 'Sex']).describe()


# In[ ]:


train_df['Age'] = train_df['Age'].fillna(train_df.groupby(['Sex', 'Pclass'])['Age'].transform('mean'))


# In[ ]:


train_df = train_df.dropna(axis=0)


# In[ ]:


test_df.describe()


# in test dataset,feature Age contains missing values, let's fill it using same method as we used in train data. 

# In[ ]:


test_df['Age'] = test_df['Age'].fillna(test_df.groupby(['Sex', 'Pclass'])['Age'].transform('mean'))


# In[ ]:


train_df.count()


# In[ ]:


test_df.count()


# one row of feature Fare contains missing value let's impute that by mean.

# In[ ]:


test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].mean())


# Let's one hot encode features Sex and Embarked

# In[ ]:


train_df['Sex'] = train_df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
test_df['Sex'] = test_df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)


# In[ ]:


train_df.head()


# In[ ]:


train_df['Embarked'] = train_df['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
test_df['Embarked'] = test_df['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)


# **Now we're ready to train a suitable Machine Learning Model on our training data and then make predicitons on test data.**

# In[ ]:


X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape


# **Logistic Regression for Binary Classification**

# In[ ]:


model = LogisticRegression()
model.fit(X_train, Y_train)
Y_preds = model.predict(X_test)
model.score(X_train, Y_train)


# In[ ]:


submission = pd.DataFrame({"PassengerId": test_df["PassengerId"], "Survived": Y_preds})


# In[ ]:


submission.to_csv('preds.csv', index=False)


# In[ ]:




