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


# In[ ]:


train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
train_data.head()


# In[ ]:


test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
test_data.head()


# In[ ]:


train_data.shape


# In[ ]:


train_data.info()


# In[ ]:


train_data.describe()


# In[ ]:


train_data['Age'].isnull().sum()


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


sns.pairplot(train_data)


# In[ ]:


sns.distplot(train_data['Age'],bins=40)


# In[ ]:


sns.distplot(train_data['Age'].dropna(),bins=40)


# In[ ]:


sns.distplot(train_data['Fare'])


# In[ ]:


sns.distplot(train_data['Pclass'])


# In[ ]:


sns.heatmap(train_data.isnull(),yticklabels=False)


# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train_data)


# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=train_data)


# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=train_data)


# In[ ]:


plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=train_data,palette='winter')


# In[ ]:


def missingAgeValue(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age


# In[ ]:


train_data['Age'] = train_data[['Age','Pclass']].apply(missingAgeValue,axis=1)


# In[ ]:


train_data['Age'].isnull().sum()


# In[ ]:


train_data.head()


# In[ ]:


train_data.drop('Cabin',axis=1,inplace=True)


# In[ ]:


train_data.isnull().sum()


# In[ ]:


train_data.dropna(inplace=True)


# In[ ]:


train_data.isnull().sum()


# In[ ]:


train_data.info()


# In[ ]:


y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])


# In[ ]:


from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X,y)


# In[ ]:


predictions = logmodel.predict(X_test)


# In[ ]:


output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")


# In[ ]:




