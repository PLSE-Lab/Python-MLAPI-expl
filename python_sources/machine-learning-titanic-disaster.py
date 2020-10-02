#!/usr/bin/env python
# coding: utf-8

# ##  Welcome to Titanic Disaster - we will explore that a bit

# ### I would be pleased if you would give me some feedback :) 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


# Loading Training Dataset and Test Dataset
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")


# #### Part 1 - Exploring the data, visualizations, choosing right model

# In this part I am going to explore the data, make some graphs, see some correlations among independent variables, choosing right model for Classification and so on.
# 
# Features:
# - PassengerId - ID of passenger
# - Survived - 0 dead, 1 alive - (dependent variable)
# - Pclass - class in which were passengers - 1st best, 3rd worst
# - Name - clear enough
# - Sex - clear enough
# - SibSp - Amount of siblings or spouses
# - Parch - Amount of parents or children
# - Ticket - Name/ID of passenger's ticket
# - Fare - Ticket price
# - Cabin - Name of cabin
# - Embarked - City of embarkation - (C = Cherbourg, Q = Queenstown, S = Southampton)

# In[ ]:


train_df.head()


# In[ ]:


# Some NaN values in Age column, Cabin nearly useless (too many NaNs)
train_df.info()


# In[ ]:


train_df.describe()


# ## Graph section

# In[ ]:


# Number of Males and Females on the board
sns.countplot(x="Sex", data=train_df, palette="pastel")


# In[ ]:


# Did they survived? With distinction between males and females
sns.countplot(x="Survived", hue="Sex", data=train_df, palette="pastel")


# In[ ]:


# Distributions of Ages - dropped NaNs
sns.distplot(train_df["Age"].dropna(), kde=False, bins=20, color="red")


# In[ ]:


# Most people Embarked in?
sns.countplot(x="Embarked", data=train_df, palette="pastel")


# In[ ]:


# Distribution between classes
sns.countplot(x="Pclass", data=train_df, palette="pastel")


# #### Part 2 - Choosing prediction target, choosing predictors, handling missing values, dummy variables, etc.

# - Prediction target - pretty obvious - Survived column
# - Predictors - Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
# - Handling missing values - Inject mean of Age column to every NaN, Cabin column will be dropped
# - Dummy variables - Sex, Pclass, Embarked

# In[ ]:


train_df.head()


# In[ ]:


# Preparing Training dataset - Dropping ID, Name, Ticket, Cabin
train_df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1, inplace=True)


# In[ ]:


train_df.head()


# In[ ]:


# Dummy Pclass
pclass_dummy = pd.get_dummies(train_df["Pclass"], drop_first=True)
pclass_dummy.head()


# In[ ]:


# Dummy Sex
sex_dummy = pd.get_dummies(train_df["Sex"], drop_first=True)
sex_dummy.head()


# In[ ]:


# Dummy Embark
embarked_dummy = pd.get_dummies(train_df["Embarked"], drop_first=True)
embarked_dummy.head()


# In[ ]:


train_df.head()


# In[ ]:


# Dropping original Pclass, Sex, Embarked columns
train_df.drop(["Pclass", "Sex", "Embarked"],axis=1, inplace=True)


# In[ ]:


# Concatenating rest of dataset with newly created dummy variables
train_df = pd.concat([train_df, pclass_dummy, sex_dummy, embarked_dummy], axis=1)


# In[ ]:


train_df.head()


# In[ ]:


# Replacing NaNs with mean of whole Age column
def age_nan(age_number):
    if np.isnan(age_number):
        return round(np.mean(train_df["Age"]))
    else:
        return age_number

age_none_nan = train_df["Age"].apply(age_nan)


# In[ ]:


# Rewriting old Age column with Age Series with no NaN values
train_df["Age"] = age_none_nan


# In[ ]:


# Splitting data into Dependent Variables and Independent Variable
y_train = train_df["Survived"]
X_train = train_df[train_df.columns[1:]]


# In[ ]:


# Fitting LogisticRegression on Training dataset
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()
classifier.fit(X_train, y_train)


# In[ ]:


test_df.head()


# In[ ]:


# Dropping columns - PassengerId, Name, Ticket, Cabin
test_df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1, inplace=True)
test_df.head()


# In[ ]:


# Dummy variables - Pclass, Sex, Embarked
dummy_pclass = pd.get_dummies(test_df["Pclass"], drop_first=True)
dummy_sex = pd.get_dummies(test_df["Sex"], drop_first=True)
dummy_embarked = pd.get_dummies(test_df["Embarked"], drop_first=True)


# In[ ]:


# Dropping old columns
test_df.drop(["Pclass", "Sex", "Embarked"], axis=1, inplace=True)


# In[ ]:


# Concatenating rest of dataset with newly created dummy variables
test_df = pd.concat([test_df, dummy_embarked, dummy_pclass, dummy_sex], axis=1)


# In[ ]:


test_df.head()


# In[ ]:


# Removing NaN and replacing with mean of Age column
def age_nan(age_number):
    if np.isnan(age_number):
        return round(np.mean(test_df["Age"]))
    else:
        return age_number

age_none_nan = test_df["Age"].apply(age_nan)
test_df["Age"] = age_none_nan


# In[ ]:


# Removing NaN and replacing with mean of Fare column
def fare_nan(fare_number):
    if np.isnan(fare_number):
        return round(np.mean(test_df["Fare"]))
    else:
        return fare_number

fare_none_nan = test_df["Fare"].apply(fare_nan)
test_df["Fare"] = fare_none_nan


# In[ ]:


# Test for other NaN values in Dataframe
test_df.isna().any()


# In[ ]:


# Make predictions from our model
y_pred = classifier.predict(test_df)


# In[ ]:


# Graphic results, if passengers from test_df would be dead or not
sns.countplot(y_pred)


# In[ ]:





# In[ ]:




