#!/usr/bin/env python
# coding: utf-8

# ![titanic.jpg](attachment:titanic.jpg)
# 
# 
# # Titanic Prediction.
# 
# This was my first attempt on a kaggle competition, this kernal is very very basic & uses Logistic Regression,the following steps were done in the following:
# 
# - Reading & Understanding the data.
# - Exploratory data analysis.
# - Missing value treatment.
# - Finally Model building.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


#Import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Import Logistic Regression
from sklearn.linear_model import LogisticRegression

#Import Scaler
from sklearn.preprocessing import StandardScaler


# In[ ]:


#read the data
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
train.head()


# ## Data Inspection

# In[ ]:


#wrting a function to inspect the data inspection
def data_inv(df):
    print('Number of Persons: ',df.shape[0])
    print('dataset variables: ',df.shape[1])
    print('-'*20)
    print('dateset columns: \n')
    print(df.columns)
    print('-'*20)
    print('data-type of each column: \n')
    print(df.dtypes)
    print('-'*20)
    print('missing rows in each column: \n')
    c=df.isnull().sum()
    print(c[c>0])
    print('-'*20)
    print('Missing vaules %age vise:\n')
    print((100*(df.isnull().sum()/len(df.index))))
    print('-'*20)
    print('Pictorial Representation:')
    plt.figure(figsize=(8,6))
    sns.heatmap(df.isnull(), yticklabels=False,cbar=False, cmap='viridis')
    plt.show()   
data_inv(train)


# In[ ]:


#inspecting the test data
data_inv(test)


# ## The Categorical Features are as follows
# 
#   - Pclass
#   - Sex
#   - SibSp ( # of siblings and spouse)
#   - Parch ( # of parents and children)
#   - Embarked
#   - Cabin

# > Age, Cabin & Embarked, has missing values so we will check the same in a while.

# In[ ]:


# Analysing the Target variable to check if the data is balanced
sns.set_style('whitegrid')
sns.countplot(x='Survived', data = train)
plt.show()


# ## Exploratory data analysis
# _In this process we'll try to check the independent variable with respect to dependent variable_

# In[ ]:


plt.figure(figsize=(12,12))
# Analying the Pclass
plt.subplot(2,3,1)
sns.countplot(x='Pclass', data = train)


#Analysing the sex
plt.subplot(2,3,2)
sns.countplot(x='Sex', data = train)


#Analysing the SibSP
plt.subplot(2,3,3)
sns.countplot(x='Pclass', data = train)
plt.show()

plt.figure(figsize=(12,12))
#Analysing the Parch
plt.subplot(2,3,4)
sns.countplot(x='Parch', data = train)
#Analysing the Embarkment
plt.subplot(2,3,5)
sns.countplot(x='Embarked', data = train)
#analysing the SibSp
plt.subplot(2,3,6)
sns.countplot(x='SibSp', data = train)
plt.show()


# In[ ]:


#checking the Survived 

# Writing the function for ploting the Categorical variable Vs the Traget Variable
def plotting(x):
    survived = train[train['Survived'] == 1][x].value_counts()
    not_survived = train[train['Survived'] == 0][x].value_counts()
    df = pd.DataFrame([survived,not_survived])
    df.index = ['Survived','not_survived']
    df.plot(kind='bar',stacked=True, figsize=(10,5))


# In[ ]:


plotting('Pclass')


# In[ ]:


plotting('Sex')


# In[ ]:


plotting('SibSp')


# In[ ]:


plotting('Parch')


# In[ ]:


plotting('Embarked')


# ## Mising Value Treatment
# 
# > Cabin variable has a lot of missing values, so dropping the same from both the test & train dataframes

# In[ ]:


#cabin has a lot of missing values in both the test & train data so dropping those
train.drop('Cabin', axis=1, inplace=True)
test.drop('Cabin', axis=1, inplace=True)


# > Age also has missing values, so to handle the age ,missing values, I am fetching the title of the name & based on the title, I'll try to impute the age.

# In[ ]:


# To handle the age, Extarcting the title (mr,mrs etc) from names col for age imputattion
train_test_df = [test, train]

for data in train_test_df:
    data['Tittle'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)


# In[ ]:


train.groupby("Tittle")["Age"].mean()


# In[ ]:


# fill missing age with median age for each title (Mr, Mrs, Miss, Others)
train["Age"].fillna(train.groupby("Tittle")["Age"].transform("median"), inplace=True)
test["Age"].fillna(test.groupby("Tittle")["Age"].transform("median"), inplace=True)


# > Now I have imputed the missing values & go ahead & check the data again.

# In[ ]:


#checking the ,missing value in train data
data_inv(train)


# In[ ]:


train.dropna(inplace=True)
data_inv(train)


# In[ ]:


#checking for missing values in test data
data_inv(test)


# In[ ]:


test.Age.fillna(test.Age.mean(), inplace=True)
test.Fare.fillna(test.Fare.mean(), inplace=True)


# In[ ]:


train.info()


# In[ ]:


train.head()


# # Data Preparation
# 
# > In this section, we do some data prepration techniques & drop the variable which will not help in the model building

# In[ ]:


train.drop(['Name', 'Ticket', 'Tittle'], axis =1, inplace=True)


# In[ ]:


train.head()


# In[ ]:


train_categorical = train.select_dtypes(include='object')
train_categorical.head()


# In[ ]:


train_categorical_dummies = pd.get_dummies(train_categorical,drop_first=True)
train_categorical_dummies.head()


# In[ ]:


train =  train.drop(train_categorical,axis=1)
train.shape


# In[ ]:


#concatenate
train = pd.concat([train, train_categorical_dummies], axis =1)
train.head()


# In[ ]:


test.head()


# In[ ]:


#run from above
test.drop(['Name', 'Ticket', 'Tittle'], axis =1, inplace=True)
test_categorical = test.select_dtypes(include='object')
test_categorical.head()


# In[ ]:


test.shape


# In[ ]:


test_categorical_dummies = pd.get_dummies(test_categorical,drop_first=True)
test_categorical_dummies.head()


# In[ ]:


test =  test.drop(test_categorical,axis=1)
train.shape


# In[ ]:


#concatenate
test = pd.concat([test, test_categorical_dummies], axis =1)
test.head()


# # Model Building
# ## Logistic Regression

# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


#Define X & y
X_train = train.drop(['Survived'], axis = 1)
y_train = train['Survived']


# In[ ]:


X_train.head()


# In[ ]:


test.head()


# In[ ]:


scaler = StandardScaler()
X_train[['Age', 'Fare']] = scaler.fit_transform(X_train[['Age', 'Fare']])
X_train.head()


# In[ ]:


X_test = test


# In[ ]:


X_test.head()


# In[ ]:


X_test[['Age', 'Fare']] = scaler.transform(X_test[['Age', 'Fare']])


# In[ ]:


X_test.isnull().sum()


# In[ ]:


logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[ ]:


X_train.info()


# In[ ]:


predictions = logmodel.predict(X_test)


# In[ ]:


predictions


# In[ ]:


output = pd.DataFrame({
                    'PassengerId': test.PassengerId,
                    'Survived' : predictions
})

output.to_csv('my_submission_logisticregression.csv', index=False)
print("Your submission was successfully saved!")


# In[ ]:




