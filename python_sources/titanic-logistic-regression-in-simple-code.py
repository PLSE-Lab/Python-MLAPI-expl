#!/usr/bin/env python
# coding: utf-8

# # UPVOTE IF YOU FIND EASY TO UNDERSTAND AND SIMPLE

# # Titanic Solution in Simple code in Simple Logistic Regression

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

#import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

#import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# **IMPORT LIBRARIES FOR DATA FIXING & EDA**

# In[ ]:



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# **GET DATA**

# In[ ]:


train_df = pd.read_csv('../input/titanic/train.csv')
test_df = pd.read_csv('../input/titanic/test.csv')


# In[ ]:


# Let's check head,info & describe for train as we'll be fixing data & doing EDA for train_df first

train_df.head()


# In[ ]:


train_df.info()


# In[ ]:


train_df.describe()


# # EXPLORATORY DATA ANALYSIS (EDA) FOR train DATA

# **FOR Missing VALUES**

# In[ ]:


train_df.isnull().sum()


# In[ ]:


sns.heatmap(train_df.isnull(),cmap='viridis',yticklabels=False)


# In[ ]:


sns.countplot(x='Survived',data=train_df,hue='Sex',palette='rainbow')


# In[ ]:


train_df[train_df['Survived']==1]['Survived'].sum()


# In[ ]:


sns.countplot(x='Survived',data=train_df,hue='Pclass',palette='plasma')


# In[ ]:


train_df['Age'].hist(bins=30,color='red',alpha=0.6)


# In[ ]:


train_df['Fare'].hist(bins=30,color='blue',alpha=0.6)


# In[ ]:


plt.figure(figsize=(10,6))
sns.boxplot(x='Pclass',y='Age',data=train_df,palette='prism')


# In[ ]:


plt.figure(figsize=(10,6))
sns.boxplot(x='Pclass',y='Fare',data=train_df,palette='prism_r')


# # FILLING MISSING VALUES FOR train data

# In[ ]:


train_df.isnull().sum()


# In[ ]:


class_1 = train_df[train_df['Pclass']==1]['Age'].mean()
class_2 = train_df[train_df['Pclass']==2]['Age'].mean()
class_3 = train_df[train_df['Pclass']==3]['Age'].mean()


# In[ ]:


train_df.loc[(train_df['Age'].isnull()) & (train_df['Pclass']==1), 'Age']=class_1
train_df.loc[(train_df['Age'].isnull()) & (train_df['Pclass']==2), 'Age']=class_2
train_df.loc[(train_df['Age'].isnull()) & (train_df['Pclass']==3), 'Age']=class_3


# In[ ]:


train_df.isnull().sum()


# In[ ]:


train_df.drop('Cabin',axis=1,inplace=True)


# In[ ]:


train_df.isnull().sum()


# In[ ]:


train_df[train_df['Embarked'].isnull()]


# In[ ]:


train_df['Embarked'].mode()[0]


# In[ ]:


train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])


# In[ ]:


train_df.isnull().sum()


# NOW WE'LL DO SOME OPERATIONS TO FIX test-df LIKE AS DID FOR train_df

# In[ ]:


test_df.head()


# In[ ]:


test_df.info()


# In[ ]:


test_df.describe()


# # EXPLORATORY DATA ANALYSIS (EDA) FOR test DATA

# **FOR MISSING VALUES**

# In[ ]:


test_df.isnull().sum()


# In[ ]:


plt.figure(figsize=(10,6))
sns.heatmap(test_df.isnull(),cmap='GnBu_r',cbar=True,yticklabels=False)


# In[ ]:


test_df['Age'].hist(bins=30,alpha=0.6,color='orange')


# In[ ]:


test_df['Fare'].hist(bins=30,alpha=0.6,color='green')


# In[ ]:


plt.figure(figsize=(10,6))
sns.boxplot(x='Pclass',y='Age',data=test_df,palette='nipy_spectral')


# In[ ]:


plt.figure(figsize=(10,6))
sns.boxplot(x='Pclass',y='Fare',data=test_df,palette='twilight')


# # FILLING MISSING VALUES FOR test DATA

# In[ ]:


test_df.isnull().sum()


# In[ ]:


class_1_test = test_df[test_df['Pclass']==1]['Age'].mean()
class_2_test = test_df[test_df['Pclass']==2]['Age'].mean()
class_3_test = test_df[test_df['Pclass']==3]['Age'].mean()


# In[ ]:


test_df.loc[(test_df['Age'].isnull()) & (test_df['Pclass']==1),'Age'] = class_1_test
test_df.loc[(test_df['Age'].isnull()) & (test_df['Pclass']==2),'Age'] = class_2_test
test_df.loc[(test_df['Age'].isnull()) & (test_df['Pclass']==3),'Age'] = class_3_test


# In[ ]:


test_df.isnull().sum()


# In[ ]:


test_df[test_df['Fare'].isnull()]


# In[ ]:


meanFare_test = test_df.groupby('Pclass').mean()['Fare']
meanFare_test


# In[ ]:


test_df['Fare'] = test_df['Fare'].fillna(meanFare_test[3])


# In[ ]:


test_df.isnull().sum()


# In[ ]:


test_df.drop('Cabin',axis=1,inplace=True)


# In[ ]:


test_df.isnull().sum()


# # CONVERTING CATEGORICAL VARIABLES

# **1. FOR train DATA**

# In[ ]:


Sex = pd.get_dummies(train_df['Sex'],drop_first=True)
Embark = pd.get_dummies(train_df['Embarked'],drop_first=True)


# In[ ]:


train_df.drop(['Sex','Name','Ticket','Embarked'],axis=1,inplace=True)


# In[ ]:


train_df = pd.concat([train_df,Sex,Embark],axis=1)


# In[ ]:


train_df.head()


# **2. FOR test DATA**

# In[ ]:


Sex = pd.get_dummies(test_df['Sex'],drop_first=True)
Embark = pd.get_dummies(test_df['Embarked'],drop_first=True)


# In[ ]:


test_df = pd.concat([test_df,Sex,Embark],axis=1)


# In[ ]:


test_df.drop(['Sex','Name','Ticket','Embarked',],axis=1,inplace=True)


# In[ ]:


test_df.head()


# # MODEL PREPROCESSING

# In[ ]:


X_train = train_df.drop(['Survived','PassengerId'], axis=1)
y_train = train_df['Survived']
X_test = test_df.drop('PassengerId', axis=1)

X_train.shape, y_train.shape, X_test.shape


# # CREATING ,TRAINING AND PREDICTING MODEL

# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


logmodel = LogisticRegression()


# In[ ]:


logmodel.fit(X_train,y_train)


# In[ ]:


predictions = logmodel.predict(X_test)


# # FILE SUBMISSION

# In[ ]:


submission = pd.DataFrame({'PassengerId': test_df['PassengerId'],'Survived': predictions})

submission.to_csv('Submission.csv', index = False)
submission.head()

