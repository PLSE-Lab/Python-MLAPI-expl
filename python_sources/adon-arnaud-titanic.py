#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import re # use for regex

# Input data files are available in the "../input/" directory.


# **Dataset import**

# In[ ]:


# import dataset train and test
input_path = "../input/titanic/"

df_train = pd.read_csv(input_path + 'train.csv')
df_test = pd.read_csv(input_path + 'test.csv')
sub = pd.read_csv(input_path + "gender_submission.csv")


# In[ ]:


# Check start line dataset
df_train.head()


# In[ ]:


# Check end line dataset
df_train.tail()


# In[ ]:


# Verification of data typing for df_train
df_train.info()


# In[ ]:


# # Verification of data typing for df_test
df_test.info()


# In[ ]:


# Verification of data completeness
df_train.isna().sum()


# In[ ]:


# check if there are any valuations that are not zero
df_test.isna().sum()


# In[ ]:


# Printing df_train basic descriptive statistics
df_train.describe()


# In[ ]:


# # Printing df_test basic descriptive statistics
df_test.describe()


# **Imputation of missing values**

# In[ ]:


# Let's calculate the average age with df_train
average_age = df_train['Age'].loc[~df_train['Age'].isna()].mean()
average_age


# In[ ]:


# Replace Age None Attribuate Values by Average age
df_train['Age'] = df_train['Age'].fillna(average_age)
df_test['Age'] = df_test['Age'].fillna(average_age)
df_train


# In[ ]:


# count peaple boarding
df_train['Embarked'].value_counts()


# A majority of the people boarded in the s port, so we will consider that the last embarked in the s port 

# In[ ]:


df_train['Embarked'].fillna('S', inplace = True)
df_test['Embarked'].fillna('S', inplace = True)


# In[ ]:


# create columns who said who in (0) or not in cabin (1)
df_train['Cabin'] = np.where(df_train['Cabin'].isnull() , 0, 1)
df_test['Cabin'] = np.where(df_test['Cabin'].isnull() , 0, 1)
df_train


# In[ ]:


# # check if there are any valuations that are not zero on test dataframe
df_test.isna().sum()


# In[ ]:


df_test.loc[df_test['Fare'].isnull()]


# In[ ]:


# calculate the Fare average on test dataframe
average_fare = df_test['Fare'].loc[~df_test['Fare'].isna()].mean()
average_fare


# In[ ]:


# give him the Fare average
df_test['Fare'] = df_train['Fare'].fillna(average_fare)


# **Creation of the first model**

# We map the categorical variables

# In[ ]:


df_train['Embarked'] = df_train['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
df_test['Embarked'] = df_test['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

df_train['Sex'] = df_train['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
df_test['Sex'] = df_test['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

df_train['Sex'] = df_train['Sex'].astype('category').cat.codes
df_train['Embarked'] = df_train['Embarked'].astype('category').cat.codes

df_train


# Analyse by Graphs

# In[ ]:


sns.countplot(x="Sex", data=df_train)


# In[ ]:


sns.catplot(x="Sex", y="Survived", data=df_train, kind="bar")


# There are more men but more women survived to the accident

# In[ ]:


sns.catplot(x="Pclass", y="Survived", data=df_train, kind="bar")


# In[ ]:


sns.countplot(x="Pclass", data=df_train)


# * There are more people to lower class but people to higher class have a better chance of survival

# In[ ]:


sns.catplot(x="SibSp", y="Survived", data=df_train, kind="bar")


# In[ ]:


sns.countplot(x="SibSp", data=df_train)


# There are more people who have no spouse, sister or brother but people who have between one or two (sisters, brothers, spouse) are more likely to survive

# In[ ]:


sns.catplot(x="Parch", y="Survived", data=df_train, kind="bar")


# In[ ]:


sns.countplot(x="Parch", data=df_train)


# there more people who don't have family relations but people who have between zero and three family relations are more likely to survive

# In[ ]:


sns.catplot(x="Embarked", y="Survived", data=df_train, kind="bar")


# In[ ]:


sns.countplot(x="Embarked", data=df_train)

There are more people  who boarded at port S but people who boarded at port C are the least likely to have escaped the disaster
# In[ ]:


sns.catplot(x="Cabin", y="Survived", data=df_train, kind="bar")


# People who have a cabin have better chance to have escaped the disaster

# In[ ]:


sns.boxplot(x='Survived', y="Fare", data=df_train.loc[df_train['Fare'] <500]);


# More people are survived with a expensive ticket

# **Improve the model / Cleaning**

# In[ ]:


df_train['FamilyCount'] = df_train['SibSp'] + df_train['Parch']
df_test['FamilyCount'] = df_test['SibSp'] + df_test['Parch']


# Calculation of the total family composition of survivors

# In[ ]:


df_train


# In[ ]:


def ageGroup(df):
    if df['Age'] <= 0 :
        group='Unknown'
    elif df['Age'] <= 14 :
        group='Child'
    elif df['Age'] <=24 :
        group='Teenager'
    elif df['Age'] <=64 :
        group='Adult'
    else :
        group='Senior'
    return group

df_train['Age'] = df_train.apply(ageGroup, axis=1)
df_test['Age'] = df_test.apply(ageGroup, axis=1)

df_train['Age'] = df_train['Age'].map( {'Unknown': 0, 'Child': 1, 'Teenager': 2, 'Adult': 3,'Senior': 4 } ).astype(int)
df_test['Age'] = df_test['Age'].map( {'Unknown': 0, 'Child': 1, 'Teenager': 2, 'Adult': 3,'Senior': 4 } ).astype(int)
df_train


# In[ ]:


sns.catplot(x="Age", y="Survived", data=df_train, kind="bar")


# Those who have survived most are the children

# In[ ]:


def civility(df):
    if re.search('Mme.',df['Name']) != None:
        civility = 'Mrs'
    elif re.search('Ms.',df['Name']) != None:
        civility = 'Mrs'
    elif re.search('Major.',df['Name']) != None:
        civility = 'Major'
    elif re.search('Capt.',df['Name']) != None:
        civility = 'Captain'
    elif re.search('Jonkheer.',df['Name']) != None:
        civility = 'Jonkheer'
    elif re.search('Mlle.',df['Name']) != None:
        civility = 'Miss'
    elif re.search('the Countess.',df['Name']) != None:
        civility = 'the Countess'
    elif re.search('Mlle.',df['Name']) != None:
        civility = 'colonel'
    elif re.search('Col.',df['Name']) != None:
        civility = 'Colonel'
    elif re.search('Don.',df['Name']) != None:
        civility = 'Don'
    elif re.search('Dr.',df['Name']) != None:
        civility = 'Doctor'
    elif re.search('Master.',df['Name']) != None:
        civility = 'Master'
    elif re.search('Mrs.',df['Name']) != None:
        civility = 'Mrs'
    elif re.search('Miss.',df['Name']) != None:
        civility = 'Miss'
    elif re.search('Rev.',df['Name']) != None:
        civility = 'Reverand'
    elif re.search('Mr.',df['Name']) != None:
        civility = 'Mr'
    else :
        civility = 'Unknown'
    return civility

df_train['Civility'] = df_train.apply(civility, axis=1)
df_test['Civility'] = df_test.apply(civility, axis=1)

df_train['Civility'] = df_train['Civility'].map( {'Unknown': 0, 'Mr': 1, 'Mrs': 2, 'Miss': 3,'Reverand': 4, 'Master': 5, 'Doctor': 6, 'Don': 7, 'Colonel': 8, 'the Countess': 9, 'Jonkheer': 10, 'Captain': 11, 'Major': 12 } ).astype(int)
df_test['Civility'] = df_test['Civility'].map( {'Unknown': 0, 'Mr': 1, 'Mrs': 2, 'Miss': 3,'Reverand': 4, 'Master': 5, 'Doctor': 6, 'Don': 7, 'Colonel': 8, 'the Countess': 9, 'Jonkheer': 10, 'Captain': 11, 'Major': 12 } ).astype(int)

df_train


# In[ ]:


# For find the rest of civility to map
df_train.query("Civility == '0'")


# In[ ]:


sns.catplot(x="Civility", y="Survived", data=df_train, kind="bar")


# The most people who survive are the Countess

# In[ ]:


df_train


# In[ ]:


def fareSection(df):
    df['Fare']
    if df['Fare'] < 10:
        fare = 'less expensive'
    elif df['Fare'] < 30:
        fare = 'less expensive than 30'
    elif df['Fare'] < 70:
        fare = 'less expensive than 70'
    elif df['Fare'] < 100:
        fare = 'less expensive than 100'
    else : 
        fare = 'expensive price'
    return fare

df_train['Fare'] = df_train.apply(fareSection, axis=1)
df_test['Fare'] = df_test.apply(fareSection, axis=1)

df_train['Fare'] = df_train['Fare'].map( { 'less expensive': 0, 'less expensive than 30': 1,'less expensive than 70': 2, 'less expensive than 100': 3, 'expensive price': 4  } ).astype(int)
df_test['Fare'] = df_test['Fare'].map( { 'less expensive': 0, 'less expensive than 30': 1,'less expensive than 70': 2, 'less expensive than 100': 3, 'expensive price': 4   } ).astype(int)

df_train


# In[ ]:


my_cols = ['Age', 'Sex', 'Pclass', 'FamilyCount', 'Fare', 'SibSp', 'Parch', 'Cabin', 'Civility']


# In[ ]:


y_train = df_train['Survived']


# In[ ]:


X_train = df_train.loc[:,my_cols]


# In[ ]:


X_test = df_test.loc[:, my_cols]

sub.columns
# **Test and simulation**

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
rf = RandomForestClassifier(n_estimators=100)


# In[ ]:


def train_model(model, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    return {
        'train accuracy': train_acc,
        'test accuracy': test_acc
    }

print(train_model(rf, X_train, y_train))


# In[ ]:


rf.fit(X_train, y_train)
train_acc = accuracy_score(y_train, rf.predict(X_train))
print(train_acc)


# In[ ]:


rf.predict(X_test)


# In[ ]:


# cell to store the result of the model by calling its rf model and data set
# generate a dataframe with PassengerId and survived

submission = pd.DataFrame({'PassengerId': df_test['PassengerId'], 'survived': rf.predict(X_test)})
submission.to_csv('submission.csv', index=False)

