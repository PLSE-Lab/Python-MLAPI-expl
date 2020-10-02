#!/usr/bin/env python
# coding: utf-8

# Importing libraries

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xb
from sklearn.linear_model import LogisticRegression
import warnings 
warnings.filterwarnings('ignore')
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


#  Reading input csv files

# In[ ]:


train_data=pd.read_csv('/kaggle/input/titanic/train.csv')
test_data=pd.read_csv('/kaggle/input/titanic/test.csv')


# * Shape of the dataset
# * Description
# * Columns 
# * Dataypes of columns

# In[ ]:


train_data.shape


# In[ ]:


train_data.describe()


# In[ ]:


train_data.columns


# In[ ]:


train_data.info()


# Data Preprocessing

# In[ ]:


train_data.isnull().sum()


# In[ ]:


test_data.isnull().sum()


# Plotting out the distribution of Age.

# In[ ]:


sns.distplot(train_data['Age'],kde=True)


# In[ ]:


train_data['Age'] = train_data['Age'].fillna(30)


# Out of 871 entries, 677 entries of Cabin are null. Hence it is better to drop that column.

# In[ ]:


train_data = train_data.drop(['Cabin'],axis=1)
test_data = test_data.drop(['Cabin'],axis=1)


# Embarked is a categorical feature which has only 2 null values. Hence the null values will be filled by calculating the mode.

# In[ ]:


train_data['Embarked'].value_counts()


# In[ ]:


train_data['Embarked']=train_data['Embarked'].fillna('S')


# Converting categorical features Embarked and Sex into numerical and float variable Fare into int 

# In[ ]:


train_data['Sex'] = pd.get_dummies(train_data['Sex'])
test_data['Sex'] = pd.get_dummies(test_data['Sex'])
train_data['Fare']=train_data['Fare'].astype('int32')
train_data['Embarked'] = pd.factorize(train_data['Embarked'])[0]
test_data['Embarked'] = pd.factorize(test_data['Embarked'])[0]
test_data['Fare']=test_data['Fare'].fillna(test_data['Fare'].mean())
test_data['Fare']=test_data['Fare'].astype('int32')


# Plotting graphs to get insights  

# In[ ]:


sns.heatmap(train_data.corr(),annot=True,linewidths=0.5)


# In[ ]:


sns.countplot(x=train_data['Pclass'],hue=train_data['Survived'])


# In[ ]:


sns.countplot(x=train_data['Embarked'],hue=train_data['Survived'])


# In[ ]:


sns.countplot(x=train_data['Age'],hue=train_data['Survived'])


# Converting Age into 5 groups as follows

# In[ ]:


for i in range(0,len(train_data)):
    if train_data['Age'][i] <= 15:
        train_data['Age'][i] = 0
    elif (train_data['Age'][i] > 15) & (train_data['Age'][i] <=35):
        train_data['Age'][i]=1
    elif (train_data['Age'][i] > 35) & (train_data['Age'][i] <=55):
        train_data['Age'][i]=2
    elif (train_data['Age'][i] > 55) & (train_data['Age'][i] <=75):
        train_data['Age'][i]=3
    else:
        train_data['Age'][i]=4


# In[ ]:


for i in range(0,len(test_data)):
    if test_data['Age'][i] <= 15:
        test_data['Age'][i] = 0
    elif (test_data['Age'][i] > 15) & (test_data['Age'][i] <=26):
        test_data['Age'][i]=1
    elif (test_data['Age'][i] > 35) & (test_data['Age'][i] <=55):
        test_data['Age'][i]=2
    elif (test_data['Age'][i] > 55) & (test_data['Age'][i] <=75):
        test_data['Age'][i]=3
    else:
        test_data['Age'][i]=4


# In[ ]:


sns.countplot(x=train_data['Fare'],hue=train_data['Pclass'])


# Converting Fare into 3 groups based on above visualization 

# In[ ]:


for i in range(0,len(train_data)):
    if train_data['Fare'][i] <= 50:
        train_data['Fare'][i] = 3
    elif (train_data['Fare'][i] > 50) & (train_data['Fare'][i] <=150 ):
        train_data['Fare'][i]=2
    else:
        train_data['Fare'][i]=1


# In[ ]:


for i in range(0,len(test_data)):
    if test_data['Fare'][i] <= 50:
        test_data['Fare'][i] = 3
    elif (test_data['Fare'][i] > 50) & (test_data['Fare'][i] <=150 ):
        test_data['Fare'][i]=2
    else:
        test_data['Fare'][i]=1


# Combining SibSp and Parch into a single feature Fam

# In[ ]:


train_data['Fam'] = train_data['Parch'] + train_data['SibSp']
test_data['Fam'] = test_data['Parch'] + test_data['SibSp']


# In[ ]:


sns.countplot(train_data['Fam'])


# Dividing Fam into 5 groups as follows

# In[ ]:


for i in range(0,len(train_data)):
    if train_data['Fam'][i] == 0:
        train_data['Fam'][i] = 0
    elif (train_data['Fam'][i] >= 1) & (train_data['Fam'][i] <=3):
        train_data['Fam'][i]=1
    elif (train_data['Fam'][i] >= 4) & (train_data['Fam'][i] <=6):
        train_data['Fam'][i]=2
    elif (train_data['Fam'][i] >= 7) & (train_data['Fam'][i] <=9):
        train_data['Fam'][i]=3
    else:
        train_data['Fam'][i]=4


# In[ ]:


for i in range(0,len(test_data)):
    if test_data['Fam'][i] == 0:
        test_data['Fam'][i] = 0
    elif (test_data['Fam'][i] >= 1) & (test_data['Fam'][i] <=3):
        test_data['Fam'][i]=1
    elif (test_data['Fam'][i] >= 4) & (test_data['Fam'][i] <=6):
        test_data['Fam'][i]=2
    elif (test_data['Fam'][i] >= 7) & (test_data['Fam'][i] <=9):
        test_data['Fam'][i]=3
    else:
        test_data['Fam'][i]=4


# Selecting X and Y and splitting the dataset

# In[ ]:


X = train_data[['Sex','Pclass','Age','Parch','Fam','Fare','Embarked']]
y = train_data[['Survived']]
X_test = test_data[['Sex','Pclass','Age','Parch','Fam','Fare','Embarked']]
train_X , test_X , train_y , test_y = train_test_split(X,y,test_size = 0.2,random_state=0)


# Training on different models

# In[ ]:


classifier = RandomForestClassifier(n_estimators=500,max_depth=3)
classifier.fit(train_X,train_y)
classifier.score(test_X,test_y)


# In[ ]:


xgb_model = xb.XGBClassifier(base_score=0.5,n_estimators=1000, learning_rate=0.05)
xgb_model.fit(train_X, train_y, early_stopping_rounds=5, 
             eval_set=[(test_X, test_y)], verbose=False)
xgb_model.score(test_X,test_y)


# In[ ]:


log_reg = LogisticRegression()
log_reg.fit(train_X,train_y)
log_reg.score(test_X,test_y)


# Making predictions and saving into output csv file

# In[ ]:


predictions = xgb_model.predict(X_test)


# In[ ]:


output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)

