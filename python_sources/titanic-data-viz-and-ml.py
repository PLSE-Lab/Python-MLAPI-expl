#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train = pd.read_csv('/kaggle/input/titanic/train.csv')


# In[ ]:


train.head()


# In[ ]:


train.isnull()


# In[ ]:


sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')


# In[ ]:


sns.set_style('whitegrid')


# In[ ]:


sns.countplot(x='Survived', data=train)


# In[ ]:


sns.countplot(x='Survived',hue='Sex',data=train)


# In[ ]:


sns.countplot(x='Survived',hue='Pclass',data=train)


# In[ ]:


sns.distplot(train['Age'].dropna(), kde=False, bins=30)


# In[ ]:


train['Age'].plot.hist(bins=30)


# In[ ]:


train.info()


# In[ ]:


sns.countplot(x='SibSp', data=train)


# In[ ]:


train['Fare'].hist(bins=40, figsize=(10, 4))


# In[ ]:


import cufflinks as cf


# In[ ]:


cf.go_offline()


# In[ ]:


train['Fare'].iplot(kind='hist', bins=30)


# In[ ]:


plt.figure(figsize=(10, 7))
sns.boxplot(x='Pclass', y='Age', data=train)


# In[ ]:


def inpute_age(cols):
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


train['Age'] = train[['Age', 'Pclass']].apply(inpute_age, axis=1)


# In[ ]:


sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')


# In[ ]:


train.drop('Cabin', axis=1, inplace=True)


# In[ ]:


train.head()


# In[ ]:


sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')


# In[ ]:


sex = pd.get_dummies(train['Sex'], drop_first=True)


# In[ ]:


sex


# In[ ]:


embark = pd.get_dummies(train['Embarked'], drop_first=True)


# In[ ]:


embark.head()


# In[ ]:


train = pd.concat([train, sex, embark], axis=1)


# In[ ]:


train.head()


# In[ ]:


train.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)


# In[ ]:


train.head()


# In[ ]:


train.drop('PassengerId', axis=1, inplace=True)


# In[ ]:


train.head()


# In[ ]:


X = train.drop('Survived', axis=1)
y = train['Survived']


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


logmodel = LogisticRegression()


# In[ ]:


logmodel.fit(X_train, y_train)


# In[ ]:


predicitions = logmodel.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


print(classification_report(y_test, predicitions))


# In[ ]:


from sklearn.metrics import confusion_matrix


# In[ ]:


confusion_matrix(y_test, predicitions)


# In[ ]:




