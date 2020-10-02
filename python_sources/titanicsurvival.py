#!/usr/bin/env python
# coding: utf-8

# In[9]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[10]:


df = pd.read_csv('../input/train.csv')
df.head()


# In[11]:


df['Name'] = df.Name.map(lambda x: x.split(",")[1].split(".")[0])


# In[12]:


df.head()


# In[13]:


df.dtypes


# In[14]:


df[['Sex', 'Embarked', 'Name', 'Ticket','Cabin']].describe()


# In[15]:


def bar_chart(feature):
    survived = df[df['Survived']==1][feature].value_counts()
    dead = df[df['Survived'] == 0][feature].value_counts()
    df_res = pd.DataFrame([survived, dead])
    df_res.index = ['Survived','Dead']
    df_res.plot(kind='bar', stacked=True, figsize=(10,5))


# In[16]:


bar_chart('Sex')


# In[17]:


X = df[[ 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Fare', 'Parch', 'Embarked','Cabin','Ticket']]
y = df['Survived']


# In[18]:


X.isna().any()


# In[19]:


plt.figure(figsize=(12,7))
sns.boxplot(x='Pclass', y='Age', data=X, palette='winter')


# In[20]:


def impute_age(cols):
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


# In[21]:


X['Age'] = X[['Age','Pclass']].apply(impute_age, axis=1)


# In[22]:


sns.heatmap(X.isnull(),yticklabels=False, cbar=False, cmap='viridis')


# In[23]:


X.Cabin.isnull().sum() #204 not Nulls
X.Cabin.fillna(X.Cabin.mode()[0], inplace=True)
X.Embarked.fillna(X.Embarked.mode()[0], inplace=True)


# In[24]:


X.isna().any()


# In[25]:


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import category_encoders as ce


# In[26]:


print(X.Name.unique())
X.Name = LabelEncoder().fit_transform(X.Name)
print(X.head())


# In[27]:


embark = pd.get_dummies(X['Embarked'], drop_first=True)
sex = pd.get_dummies(X['Sex'], drop_first=True)


# In[28]:


X.head()


# In[29]:


X.drop(['Sex', 'Embarked'],axis=1, inplace=True)
X.head()


# In[30]:


X = pd.concat([X, sex, embark], axis=1)


# In[31]:


X.head()


# In[32]:


X.Cabin = LabelEncoder().fit_transform(X.Cabin)
print(X.head())


# In[33]:


X.Ticket = LabelEncoder().fit_transform(X.Ticket)
print(X.head())


# In[34]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42, shuffle=False)
print(X_train.head())


# In[35]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 
model = LogisticRegression()
model.fit(X_train, y_train)
model.predict(X)


# In[36]:


model.score(X,y)


# In[ ]:





# In[ ]:




