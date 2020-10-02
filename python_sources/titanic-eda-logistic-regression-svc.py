#!/usr/bin/env python
# coding: utf-8

# # The process

# Here's How we will be going with the process flow :
# 
# * Import the data
# * Understand and Clean the data
# * Analyse any patterns in the data
# * Model/ Create a solution
# * Submit your predictions

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # plotting graphs
import seaborn as sns # Seaborn for plotting and styling
import math
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# #  Importing files and having a first look at the data

# In[ ]:


train = pd.read_csv("/kaggle/input/titanic/train.csv")
test = pd.read_csv("/kaggle/input/titanic/test.csv")
combine = [train, test]


# In[ ]:


train.head()


# The basic logic that we will be following in this notebook is to find a pattern.
# 
# In order to find a pattern in  a problem , the most important thing is to know the problem statement and its data fields inside out.
# Hence we will try to to do an EDA in the following cells.

# In[ ]:


print(train.columns.values)


# ## Idenntify different data fields, e.g Numerical/textual/Categorical etc

# Looking at train.head()
# 
# Categorical data : 
# * Survived
# * Pclass
# * Embarked
# * Sex
# 
# Numerical:
# 
# * Age
# * Fare
# * SibSp
# * Parch
# 
# Mixed : 
# * Ticket
# * Cabin

# ### Chances of Error in data capturing
# 
# * Typos :  Name column
# * Missing values :  possible in all columns
# * Out of place values : eg survived cannot contain an alphabet
# 

# In[ ]:


train.columns


# # Univariate analysis

# In[ ]:


#def count_analysis()
##def create_subplot()
##def draw_graphs()

cols = ['Pclass', 'Sex', 'SibSp','Survived',
       'Parch', 'Embarked']

fig,ax = plt.subplots(math.ceil(len(cols)/3),3,figsize=(20, 12))
ax = ax.flatten()
for a,s in zip(ax,cols):
    sns.countplot(x =s,data = train,palette = "bright",ax =a)


# In[ ]:


#def count_analysis()
##def create_subplot()
##def draw_graphs()

cols = ['Pclass', 'Sex', 'SibSp',
       'Parch', 'Embarked']

fig,ax = plt.subplots(math.ceil(len(cols)/3),3,figsize=(20, 12))
ax = ax.flatten()
for a,s in zip(ax,cols):
    sns.barplot(x =s,y="Survived",data = train,palette = "bright",ax =a)


# # Bivariate analysis

# In[ ]:


cols = ['Pclass', 'Sex', 'SibSp',
       'Parch', 'Embarked']

fig,ax = plt.subplots(math.ceil(len(cols)/3),3,figsize=(20, 12))
ax = ax.flatten()
for a,s in zip(ax,cols):
    sns.countplot(x=s, hue="Survived", data=train,palette = "bright",ax =a)


# ## Insight
# - Check for correlation of bucketed values from EDA
# 
# e.g Pclass 1 is more likely to survive and females are more likely to survive

# In[ ]:


x = pd.Series(train["Age"])


# In[ ]:


fig,ax = plt.subplots(figsize=(20, 10)) 
sns.distplot(x, color="y",ax = ax)


# In[ ]:


train.corr()


# In[ ]:


cols = ['Pclass', 'SibSp','Survived','Fare',
       'Parch']
colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train[cols].astype(float).corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)


# ### Insights
# 
# * Fare and survived have a direct correlation
# * Parch and survived have a direct correlation

# # Clean the data

# ### Look for missing values

# In[ ]:


print("TRAIN DATA\n")
train.info()
print("\n===================================================\n")
print("TEST DATA\n")
test.info()


# In[ ]:


train.describe()


# Age, embarked,Cabin have missing values
# Cabin has very less values hence we will discard it
# Hence we need to treat Age and embarked
# 
# * For treating Age we will replace the missing values with median value
# * For treating embarked we will replace missing values with mode

# In[ ]:


# check for na values
for col in train.columns:
    print("No. of Na values in " + str(col) + " " + str(len(train[train[col].isnull()])) +'\n')


# In[ ]:


imputed_age = float(train['Age'].dropna().median())
imputed_embarked = train['Embarked'].dropna().mode()[0]
print("imputed_age : ",imputed_age)
print("imputed_embarked : ",imputed_embarked)


# In[ ]:


for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(imputed_embarked)
    dataset['Age'] = dataset['Age'].fillna(imputed_age)


# In[ ]:


train.describe(include=['O'])


# In[ ]:


#########
# QC

# check for na values
for col in train.columns:
    print("No. of Na values in " + str(col) + " " + str(len(train[train[col].isnull()])) +'\n')


# In[ ]:


combine=[train,test]


# In[ ]:


for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)


# In[ ]:


for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']
    dataset['Age'] = dataset['Age'].astype(int)
    


# In[ ]:


train.head()


# In[ ]:


train = train.drop(['Ticket', 'Cabin','Name'], axis=1)
test = test.drop(['Ticket', 'Cabin','Name'], axis=1)


# In[ ]:


test['Fare'].fillna(test['Fare'].dropna().mean(), inplace=True)
# test.head()


# In[ ]:


train['FareBand'] = pd.qcut(train['Fare'], 5)
train[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)


# In[ ]:


combine = [train,test]


# In[ ]:


for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.854, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 10.5), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 10.5) & (dataset['Fare'] <=  21.679), 'Fare']   = 2
    dataset.loc[(dataset['Fare'] > 21.679) & (dataset['Fare'] <=  39.688), 'Fare']   = 3
    dataset.loc[ dataset['Fare'] > 39.688, 'Fare'] = 4
    dataset['Fare'] = dataset['Fare'].astype(int)

train = train.drop(['FareBand'], axis=1)


# In[ ]:


combine = [train,test]


# In[ ]:


for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)


# In[ ]:


train.head()


# In[ ]:





# In[ ]:


train_df = train
test_df = test
X_train = train_df.drop("Survived", axis=1)
X_train = X_train.drop("PassengerId", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape


# In[ ]:


X_train.head()


# In[ ]:


X_test.columns


# In[ ]:


X_train.info()


# In[ ]:


logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log


# In[ ]:


svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc


# In[ ]:


knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn


# In[ ]:


random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('submission_v2.csv', index=False)


# In[ ]:




