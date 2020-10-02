#!/usr/bin/env python
# coding: utf-8

# In[2]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import random
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ###### First of all, i want to thank you all that are viewing this kernel, this is my first kernel, and hopefully it helps me and more beginners in a significant way!

# # A first look in our data.

# Let's take a first look at our dataset to see how it looks.

# In[3]:


X_full = pd.read_csv('../input/train.csv')
X_full.columns


# Name and PassengerID are really not useful features.

# In[4]:


ID = X_full.PassengerId 
NAME = X_full.Name #Maybe we could use this in the future, so let's store this here.
X_full.drop(['PassengerId','Name'], inplace=True, axis=1)
X_full.head()


# In[5]:


X_full.describe(include='all')


# # Preprocessing and feature engineering.
# 
# We have some NaN values in our dataset, let's visualize it.

# In[6]:


X_full.isnull().sum()


# In[7]:


plt.figure(figsize=(15,8))
sns.heatmap(X_full.isnull(), cbar=False)


# Cabin and Ticket are not good predictors of survival rate and Cabin has too many Nan values, so let's drop it out.

# In[8]:


X_full.drop(['Cabin','Ticket'], inplace=True, axis=1)
X_full.columns


# There are only two rows that are missing the embarked column. Let's drop it too.

# In[9]:


X_full.drop(X_full[X_full.Embarked.isnull() == True].index, inplace=True)
plt.figure(figsize=(15,8))
sns.heatmap(X_full.isnull(), cbar=False)


# Right, now we need to do something with the age. But first let's check a heatmap of correlations of our numeric variables.

# In[10]:


plt.figure(figsize=(15,8))
sns.heatmap(X_full.corr(), cmap='magma', annot=True)


# * Pclass has an obvious negative correlation with Fare. Passengers who payed more were in better classes.
# * Fare and Pclass has significant correlations with Survival rates. Meaning that passengers who payed more had a better chance at surviving.
# * Parch and SibSp has great correlation for obvious reasons too.
# * Age and Pclass has a negative correlation, meaning that older the passenger, more likely he's paying for a better class.
# * As for SibSp and Parch correlations with Age, greater the Age, less likely the passenger to be with family.

# # Filling nan values in age

# In[11]:


s = X_full['Age'].isnull()
X_full_age = X_full.copy() #Do this on a copy to see the distributions comparison



X_full_age.Age[s] = X_full_age.Age[s].map(lambda x: random.randrange(X_full_age.Age.median() - (X_full_age.Age.median()*0.25),
                                                       X_full_age.Age.median() + (X_full_age.Age.median()*0.25)))


plt.figure(figsize=(20,6))
sns.distplot(X_full_age.Age, ax=plt.subplot(1,2,1))
sns.distplot(X_full.Age.dropna(), ax=plt.subplot(1,2,2))


# Well, i think this distribution comparison is pretty good!
# Now let's bin our ages.

# In[12]:


X_full_age['Age'] = pd.Series(pd.cut(X_full_age.Age,[0,10,20,30,40,50,60,70,80,90], labels= False, right=True))
X_full_age.head()


# # Categorical variables

#  Now let's take care of our categorical variables. Those are Sex, Embarked and Pclass.

# In[13]:


X_full_age.dtypes


# Sex and Embarked already are Categorical, let's set Pclass to categorical too.

# In[14]:


X_full_age.Pclass = pd.Categorical(X_full_age.Pclass)
X_full_age = pd.get_dummies(X_full_age)
X_full_age.head()


# In[15]:


X_full_age.isnull().sum()


# Now that we have no NaN values, lets combine SibSp and Parch to make a new feature named Family and drop the original values.

# In[16]:


X_full_age['Family'] = X_full_age.SibSp + X_full_age.Parch
X_full_age.drop(['SibSp','Parch'], axis=1, inplace=True)
X_full_age.head()


# In[17]:


plt.figure(figsize=(15,8))
sns.heatmap(X_full_age.corr(), annot=True, cmap='magma')


# #### Now it becomes a little harder to interpret this correlation heatmap, but let's start
#     
# * Sex_female has a significant correlation with Survived, so females are more likely to survive. While males are not (Sex_male has negative correlation with survived)
# * People who embarked at Cherboug had more survivability than people who embarked at Queenstown, Southampton is the place with less survivability, this is because people who embarked at Cherbough were more rich, paid larger fares and got better classes.
# *  Women were more likely to be travelling with family than men.

# In[18]:


ytrain = X_full_age.Survived
Xtrain = X_full_age.drop(['Survived'], axis=1)


# # Nested Cross Validation

# In[19]:


from sklearn.model_selection import cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

PGrid = {"C":[1,10,100],
        "gamma":[.01, 0.1],
        "kernel":['rbf'],
        "cache_size":[200,500,1000]}

model = SVC(random_state=1)
Xtrainsvm = StandardScaler().fit_transform(Xtrain)

gsearch = GridSearchCV(estimator=model, param_grid=PGrid, cv=5, iid=False)

score = cross_val_score(gsearch, X=Xtrainsvm, y=ytrain, cv=5)

score.mean()


# # Preprocessing the test data

# In[20]:


Xtest = pd.read_csv('../input/test.csv')
ID = Xtest['PassengerId']
Xtest.drop(['PassengerId','Name','Cabin','Ticket'], inplace=True, axis=1)
plt.figure(figsize=(15,8))
sns.heatmap(Xtest.isnull(), cbar=False)


# In[21]:


s = Xtest['Age'].isnull()

Xtest.Age[s] = Xtest.Age[s].map(lambda x: random.randrange(round(Xtest.Age.median() - (Xtest.Age.median()*0.25)),
                                                           round(Xtest.Age.median() + (Xtest.Age.median()*0.25))))

Xtest.isnull().sum()


# In[22]:


s = Xtest['Fare'].isnull()

Xtest.Fare[s] = Xtest.Fare.mean()

Xtest.isnull().sum()


# In[23]:


Xtest['Age'] = pd.Series(pd.cut(Xtest.Age,[0,10,20,30,40,50,60,70,80,90], labels= False, right=True))
Xtest.head()


# In[24]:


Xtest.Pclass = pd.Categorical(Xtest.Pclass)
Xtest = pd.get_dummies(Xtest)
Xtest.head()


# In[25]:


Xtest['Family'] = Xtest.SibSp + Xtest.Parch
Xtest.drop(['SibSp','Parch'], axis=1, inplace=True)
Xtest.head()


# In[26]:


Xtest = StandardScaler().fit_transform(Xtest)


# In[27]:


gsearch.fit(Xtrain, ytrain)

svmpredict = gsearch.predict(Xtest)


# # XGBoost
# 

# In[45]:


from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score


xgb = XGBClassifier(n_estimators=1000, learning_rate=0.1, random_state=0)
xgbscore = cross_val_score(xgb, Xtrainsvm, ytrain, cv=5)
print(xgbscore.mean())
xgb.fit(Xtrainsvm, ytrain)
predictions = xgb.predict(Xtest)


# # Saving submissions

# In[46]:


submission = pd.concat([ID,pd.Series(predictions.tolist())], axis=1)
submission.columns = ['PassengerId','Survived']
submission.to_csv('predictions.csv', index=False)
submission.head()


# ###### Could you all comment and tell me what could i do to improve my model accuracy, and if i did something wrong? I'm a real beginner, so your comments will be of great interest to me!
