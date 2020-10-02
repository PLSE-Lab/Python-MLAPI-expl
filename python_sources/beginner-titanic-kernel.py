#!/usr/bin/env python
# coding: utf-8

# ## 1. Data preparation
# We import the data and investigate some basic properties of the data.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Additional packages for data preprocessing, model building, data visualisation
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Import data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

combined = pd.concat([train,test],axis = 0)


# In[ ]:


# Basic properties of data
train.info()
train.describe()


# In[ ]:


test.info()
test.describe()


# In[ ]:


# Investigating NA values in each column
#train.loc[train['Fare'] == 0]
#train.loc[train['Age'].isna()]
#train.loc[train['Cabin'].notna()]
train.loc[train['Embarked'].isna()]


# We can observe a few things here.
# 1. Some people has a recorded fare of 0, distributed among all class. It is possible that they were awarded the ticket, or there was an error with the data record.
# 2. Many people with a record of Cabin is also a 1st class passenger

# ## 2. Dealing with NA values
# 
# There are a few choices to deal with the missing values in the Age, Cabin and Embarked columns.
# 1. Filling in the missing values with some proxy value (e.g. mean, median)
# 2. Dropping the entries with missing values entirely
# 
# We will try out the second method for some exploratory data analysis with Age

# In[ ]:


# Removing NA values from data
train_pre = train.dropna(subset = ["Age"],inplace = False)
train_pre.info()


# In[ ]:


sns.swarmplot(x = "Pclass",y="Age",hue = "Survived",data = train_pre)


# From the swarmplot (made with the data visualization package Seaborn), we can determine a few properites who survived the Titanic:
# 1. Most survivors seem to come from 1st class
# 2. Survivors have their age evenly distributed

# In[ ]:


sns.swarmplot(x = "Survived",y="Age",hue = "Sex",data = train_pre)


# It looks as if being a female also greatly increases your chances of surviving the Titanic!

# In[ ]:


sns.heatmap(data = train_pre.corr(),cmap = "Spectral",center = 0)


# It appears that most factors have a weak correlation with each other, with the exception of fare and parch. However, this weak correlation can also be because we did not normalise the data before plotting the heatmap. 

# In[ ]:


train_pre.dropna(subset = ['Cabin','Embarked'],inplace = True)
train_pre.info()
train_pre.describe()


# It appears that many of the labelled cabins are ones from the 1st class cabins. We will not use the Cabin column in further analysis.

# We will now fill the NA values in the Age, Cabin and Embarked columns. The following will be performed:
# 1. Fill NA values in Age with the mean age.
# 2. Fill NA values in Cabin with the proxy value "Unknown"
# 3. Fill NA values in Embarked with the mode port of embarkation.

# In[ ]:


# Filling NA values for Age
meanAge = round(combined["Age"].mean(skipna = True))
train['Age'].fillna(meanAge,inplace = True)
test['Age'].fillna(meanAge,inplace = True)


# In[ ]:


# Filling NA values for Cabin
proxyCabin = "Unknown"
train['Cabin'].fillna(proxyCabin,inplace = True)
test['Cabin'].fillna(proxyCabin,inplace = True)


# In[ ]:


# Filling NA values for Embarked
modeEmbark = combined["Embarked"].mode()[0]
train['Embarked'].fillna(modeEmbark,inplace = True)
test['Embarked'].fillna(modeEmbark,inplace = True)


# In[ ]:


# Filling NA values for Fare
meanFare = round(combined["Fare"].mean(skipna = True))
test['Fare'].fillna(meanFare,inplace = True)


# In[ ]:


train.info()
train.head()


# ## 3. Preprocessing data

# We will start off by dropping columns (Name, PassengerID, Ticket and Cabin) we are not using from the dataframe, and combining the SibSp and Parch columns.

# In[ ]:


# Dropping columns
train.drop(['Name','PassengerId','Ticket','Cabin'],axis = 1,inplace = True)
test.drop(['Name','Ticket','Cabin'],axis = 1,inplace = True)


# In[ ]:


train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
test['FamilySize'] = test['SibSp'] + test['Parch'] + 1

train.drop(['SibSp','Parch'],axis = 1,inplace = True)
test.drop(['SibSp','Parch'],axis = 1,inplace = True)


# In[ ]:


train.info()
train.head()


# We will now convert the categorical columns we want to use to binary variables.

# In[ ]:


# Encoding categorical columns 
trainPclassAdd = pd.get_dummies(train['Pclass'].reset_index(drop = True),prefix = 'Pclass',dtype = int)
trainEmbarkedAdd = pd.get_dummies(train['Embarked'].reset_index(drop = True).astype(str),prefix = 'Embarked',dtype = int)

testPclassAdd = pd.get_dummies(test['Pclass'].reset_index(drop = True),prefix = 'Pclass',dtype = int)
testEmbarkedAdd = pd.get_dummies(test['Embarked'].reset_index(drop = True).astype(str),prefix = 'Embarked',dtype = int)

train['Sex'] = train['Sex'].apply(lambda x: 1 if x == 'male' else 0)
test['Sex'] = test['Sex'].apply(lambda x: 1 if x == 'male' else 0)


# In[ ]:


train = pd.concat([train.reset_index(drop = True),trainPclassAdd,trainEmbarkedAdd],axis = 1)
test = pd.concat([test.reset_index(drop = True),testPclassAdd,testEmbarkedAdd],axis = 1)

train.drop(["Embarked","Pclass"],axis = 1, inplace = True)
test.drop(["Embarked","Pclass"],axis = 1, inplace = True)
train.head()


# In[ ]:


test.head()


# # 4. Model building
# We will test the following models for the prediction
# 1. Logistic regression
# 2. Support Vector Machine
# 3. Decision Tree Classifier 

# In[ ]:


X_train = train.drop(['Survived'],axis = 1)
y_train = train['Survived']

X_test = test.drop(['PassengerId'],axis = 1)
X_train.shape, y_train.shape, X_test.shape


# In[ ]:


# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train,y_train)
print("Training accuracy is " + str(logreg.score(X_train,y_train)* 100))
logreg.coef_[0]


# In[ ]:


# Support Vector Machine
svm = SVC()
svm.fit(X_train,y_train)
print("Training accuracy is " + str(svm.score(X_train,y_train)* 100))


# In[ ]:


# Decision tree classifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train,y_train)
print("Training accuracy is " + str(dtc.score(X_train,y_train)* 100))

