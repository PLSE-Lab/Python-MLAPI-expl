#!/usr/bin/env python
# coding: utf-8

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


# **Description of Data**
# The sinking of the RMS Titanic is one of the most infamous shipwrecks in history. On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.
# 
# One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.
# 
# In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive. In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.

# In[ ]:


#importing pakages
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# In[ ]:


#importing dataset
dataset = pd.read_csv('/kaggle/input/titanic/train.csv')
dataset.head()


# In[ ]:


Dataset = "titanic"


# In[ ]:


dataset.shape


# In[ ]:


#check for missing values
sns.heatmap(dataset.isnull())


# In[ ]:


#to check number of people that survived the disaster and number of people who did not survive.
sns.countplot(x='Survived',data=dataset)


# In[ ]:


#number of people that survived based on their age.
sns.countplot(x='Survived',data=dataset,hue="Sex")


# In[ ]:


#number of people that survived based on their Pclass.
sns.countplot(x='Survived',data=dataset,hue="Pclass")


# In[ ]:


plt.figure(figsize=(12,7))
sns.boxplot(x='Pclass',y='Age',data=dataset,palette='winter')


# In[ ]:


#missing value imputation on age and Pclass
def impute_age(cols):
  Age = cols[0]
  Pclass = cols[1]
  
  if pd.isnull(Age):
    if Pclass == 1:
      return 37
    elif Pclass ==2:
      return 29
    else:
      return 24
  else:
    return Age


# In[ ]:


dataset['Age']= dataset[['Age','Pclass']].apply(impute_age,axis=1)


# In[ ]:


#dropping independent variable cabin
dataset.drop('Cabin',axis =1,inplace=True)


# In[ ]:


dataset.groupby('Embarked').size()


# In[ ]:


#imputing common value for embarked
common_value = 'S'
dataset['Embarked'] = dataset['Embarked'].fillna(common_value)
dataset.info()


# In[ ]:


dataset.info()


# In[ ]:


dataset.head()


# In[ ]:


sex = pd.get_dummies(dataset['Sex'],drop_first=True)
embark = pd.get_dummies(dataset['Embarked'],drop_first=True)


# In[ ]:


dataset.drop(['PassengerId','Sex','Embarked','Name','Ticket'],axis=1,inplace=True)


# In[ ]:


dataset = pd.concat([dataset,sex,embark],axis=1)


# In[ ]:


#Creating training data set
X_train = dataset.drop(['Survived'],axis =1 )
y_train = dataset['Survived']


# In[ ]:


#USING RANDOM FOREST CLASSIFIER
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train,y_train)
y_pred_rf=rf.predict(X_train)


# In[ ]:


from sklearn import metrics
print('Accuracy',metrics.accuracy_score(y_train,y_pred_rf))


# **Test Data**

# In[ ]:


#importing training data
data = pd.read_csv('/kaggle/input/titanic/test.csv')
data.shape


# In[ ]:


passengerID = data['PassengerId']


# In[ ]:


data.info()


# In[ ]:


sns.heatmap(data.isnull())


# In[ ]:


#missing value imputation on age and Pclass
def impute_age(cols):
  Age = cols[0]
  Pclass = cols[1]
  
  if pd.isnull(Age):
    if Pclass == 1:
      return 37
    elif Pclass ==2:
      return 29
    else:
      return 24
  else:
    return Age


# In[ ]:


data['Age']= data[['Age','Pclass']].apply(impute_age,axis=1)


# In[ ]:


sns.heatmap(data.isnull())


# In[ ]:


#dropping cabin from dataset
data.drop('Cabin',axis =1,inplace=True)


# In[ ]:


data['Age'] = data[['Age','Pclass']].apply(impute_age,axis=1)
data.fillna(method='ffill',inplace=True)


# In[ ]:


sex = pd.get_dummies(data['Sex'],drop_first=True)
embark = pd.get_dummies(data['Embarked'],drop_first=True)


# In[ ]:


data.drop(['PassengerId','Sex','Embarked','Name','Ticket'],axis=1,inplace=True)


# In[ ]:


data = pd.concat([data,sex,embark],axis=1)


# In[ ]:


data.head()


# In[ ]:


#predicting the people who survived on test data
y_pred_test = rf.predict(data)


# In[ ]:


y_pred_test


# In[ ]:


df = pd.DataFrame({'PassengerID':passengerID, 'Survived':y_pred_test})
df


# In[ ]:




