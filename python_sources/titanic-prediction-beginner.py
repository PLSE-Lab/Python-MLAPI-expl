#!/usr/bin/env python
# coding: utf-8

# In[293]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[294]:


# loading the data
data=pd.read_csv("../input/train.csv")
# top 5 columns
data.head(10)


# In[295]:


# length of data
print(data.shape)


# In[296]:


print(data.describe())


# In[297]:


# columns with null values
data[data.Cabin.isnull()]
null_columns=[col for col in data.columns if data[col].isnull().sum()>0]
print(null_columns)


# In[298]:


# Removing unnecessary columns (by Intuition)
# Name
# Cabin and Embarked as they have null values
# Ticket as it is not signifying importance
data=data.drop(['Name','Cabin','Embarked','Ticket'],axis=1)
data.head()


# In[299]:


# basic visualization to see imbalance in sex and number of surviors
data.groupby(['Sex']).Survived.sum().plot.bar() # no. of people survived
#data.groupby(['Sex']).Survived.count().plot.bar() # no.of people present


# In[300]:


# basic visualization to number of surviors in different passenger classes
data.groupby(['Pclass']).Survived.sum().plot.bar()


# In[301]:


# basic visualization to number of surviors in different ages
data.groupby('Age').Survived.sum().plot.bar(figsize=(19,6)) # maximum number of survivors are at age around 15-49
#data['Age'].value_counts().plot.bar(figsize=(19,6)) # to see how many people are there for each age
#data['Age'][:10]


# In[302]:


# Handling Null values in Age column
# Replace them with mean as most of them are in the age around 24 makes sense and might not give much errors
data=data.fillna(round(data.Age.mean(),0)) 
data.head(6)


# In[303]:


# basic visualization to number of surviors in different 
data.groupby('Parch').Survived.sum().plot.bar(figsize=(19,6)) #singles have survived more


# In[304]:


# Converting categorical to numerical using dummies (one hot encoding)
#pd.get_dummies(data['Sex']'Pclass'])
col=['Sex','Pclass']
for i in col:
    data=pd.concat([data,pd.get_dummies(data[i])],axis=1) #,prefix=[i+'_']

data.head()


# In[305]:


# drop column Sex in data
data=data.drop(['Sex','Pclass'],axis=1)
data.head()


# In[306]:


# features and target variable
X=['Age','SibSp','Parch','Fare','female','male',1,2,3]
y=['Survived']
#data.columns
features=data[X]
target=data[y]
#features #visualizing the data


# In[307]:


# Logistic Regression from sklearn
from sklearn.linear_model import LogisticRegression
model=LogisticRegression().fit(features,target)

#data['Age'].dtype


# In[308]:


# load the test data
test_data=pd.read_csv("../input/test.csv")
original_test_data=test_data.copy()
test_data.head()
# Converting categorical to numerical using dummies (one hot encoding)
#pd.get_dummies(data['Sex']'Pclass'])
col=['Sex','Pclass']
for i in col:
    test_data=pd.concat([test_data,pd.get_dummies(test_data[i])],axis=1) #,prefix=[i+'_']


# In[309]:


# columns with null values
null_columns=[col for col in test_data.columns if test_data[col].isnull().sum()>0]
print(null_columns)

test_data.Age=test_data.Age.fillna(test_data.Age.mean())
test_data.Fare=test_data.Fare.fillna(test_data.Fare.mean())


# In[310]:


features_predict=test_data[X]
results_final=model.predict(features_predict)


# In[319]:


pid=pd.DataFrame(list(original_test_data['PassengerId'].values))
rf=pd.DataFrame(list(results_final))
#output=pd.DataFrame([list(original_test_data['PassengerId'].values),list(results_final)],columns=['PassengerId','Survived'])
output=pd.concat([pid,rf],axis=1)
output.columns=['PassengerId','Survived']
original_test_data.columns
output.head()
#len(results_final)
#len(output)
#len(original_test_data['PassengerId'].values)
#list(original_test_data['PassengerId'].values)


# In[320]:


# storing it to csv
output.to_csv("output.csv",index=False)


# In[318]:


import os
print(os.getcwd())


# In[314]:


os.chdir("/kaggle/working")


# In[ ]:





# In[ ]:




