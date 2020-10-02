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


import matplotlib.pyplot as plt
import seaborn as sns
import math

train_data = pd.read_csv("../input/titanic/train.csv")
train_data.head(10)


# In[ ]:


print("number of passanger in the original dataset: "+ str(len(train_data.index)))


# ## Analyzing Data

# In[ ]:


sns.countplot(x="Survived", data= train_data)


# In[ ]:


sns.countplot(x="Survived", hue='Sex', data= train_data)


# In[ ]:


sns.countplot(x="Survived", hue="Pclass",data= train_data)


# In[ ]:


train_data["Age"].plot.hist()


# In[ ]:


train_data["Fare"].plot.hist(figsize=(10,5))


# In[ ]:


train_data.info()


# In[ ]:


sns.countplot(x = "SibSp", data= train_data)


# Data Wrangling

# In[ ]:


train_data.isnull()


# In[ ]:


train_data.isnull().sum()


# In[ ]:


sns.heatmap(train_data.isnull(), yticklabels=False)


# In[ ]:


sns.heatmap(train_data.isnull(), yticklabels=False, cmap="viridis")


# In[ ]:


sns.boxplot(x= "Pclass", y="Age", data = train_data)


# In[ ]:


train_data.drop("Cabin", axis=1, inplace=True)


# In[ ]:


train_data.dropna(inplace=True)


# In[ ]:


sns.heatmap(train_data.isnull(), yticklabels=False, cbar=False)


# In[ ]:


train_data.isnull().sum()


# In[ ]:


sex = pd.get_dummies(train_data['Sex'], drop_first =True)
embark = pd.get_dummies(train_data['Embarked'], drop_first =True)
pclass = pd.get_dummies(train_data['Pclass'], drop_first =True)


# In[ ]:


train_data = pd.concat([train_data, sex, embark, pclass], axis=1)
train_data.head()


# In[ ]:


train_data.drop(["PassengerId","Pclass","Sex","Name","Ticket", 'Embarked'], axis=1, inplace=True)


# In[ ]:


train_data.head()


# In[ ]:


X = train_data.drop('Survived', axis=1)
y = train_data['Survived']


# In[ ]:


X.head(20)


# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)


# In[ ]:


from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()


# In[ ]:


#X_train.head(10)

logmodel.fit(X_train, y_train)


# In[ ]:


prediction = logmodel.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


classification_report(y_test, prediction)


# In[ ]:


from sklearn.metrics import confusion_matrix


# In[ ]:


confusion_matrix(y_test, prediction)


# In[ ]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, prediction)


# In[ ]:


test_data = pd.read_csv("../input/titanic/test.csv")


# In[ ]:


test_data.isnull().sum()


# In[ ]:


Age_mean = test_data['Age'].mean()
fair_mean = test_data['Fare'].mean()
Age_mean


# In[ ]:


test_data['Age'] = test_data['Age'].fillna(Age_mean)
test_data['Fare'] = test_data['Fare'].fillna(fair_mean)


# In[ ]:


test_data.head()


# In[ ]:


#test_data.dropna(inplace=True)

sex_test = pd.get_dummies(test_data['Sex'], drop_first =True)
embark_test = pd.get_dummies(test_data['Embarked'], drop_first =True)
pclass_test = pd.get_dummies(test_data['Pclass'], drop_first =True)

test_data = pd.concat([test_data, sex_test, embark_test, pclass_test], axis=1)
test_data.head()


# In[ ]:


PassengerId = test_data['PassengerId']
test_data.drop(["PassengerId","Pclass","Sex","Name","Ticket", 'Embarked', 'Cabin'], axis=1, inplace=True)
test_data.head()


# In[ ]:


PassengerId.tail(25)

print(PassengerId.count())


# In[ ]:


test_data.isnull().sum()


# In[ ]:


test_predictrion = logmodel.predict(test_data)


# In[ ]:


print(test_predictrion)
test_predictrion = pd.DataFrame(test_predictrion)


# In[ ]:


Result = pd.concat([PassengerId, test_predictrion])


# In[ ]:


Result.to_csv('submission123.csv', sep='\t', encoding='utf-8')


# In[ ]:





# In[ ]:


PassengerId


# In[ ]:




