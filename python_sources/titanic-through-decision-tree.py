#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[2]:


data = pd.read_csv('../input/train.csv')
data1 = pd.read_csv('../input/test.csv')


# In[3]:


data


# In[4]:


data1


# In[5]:


data.describe()


# In[6]:


data = data.drop(['Ticket','Cabin'],axis = 1)
data1 = data1.drop(['Ticket','Cabin'],axis = 1)


# In[7]:


data.shape


# In[8]:


data1.shape


# In[9]:


data['Title'] = data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
pd.crosstab(data['Title'],data['Sex'])


# In[10]:


data1['Title'] = data1.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
pd.crosstab(data1['Title'],data1['Sex'])


# In[11]:


data['Title'] = data['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
data['Title'] = data['Title'].replace('Mlle', 'Miss')
data['Title'] = data['Title'].replace('Ms', 'Miss')
data['Title'] = data['Title'].replace('Mme', 'Mrs')
  


# In[12]:


data1['Title'] = data1['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
data1['Title'] = data1['Title'].replace('Mlle', 'Miss')
data1['Title'] = data1['Title'].replace('Ms', 'Miss')
data1['Title'] = data1['Title'].replace('Mme', 'Mrs')


# In[13]:


mapping = {'Master':1 , 'Miss':2 , 'Mr':3 , 'Mrs':4,'Rare':5}
data['Title'] = data['Title'].map(mapping)
data['Title'] = data['Title'].fillna(0)


# In[14]:


data


# In[15]:


mapping = {'Master':1 , 'Miss':2 , 'Mr':3 , 'Mrs':4,'Rare':5}
data1['Title'] = data1['Title'].map(mapping)
data1['Title'] = data1['Title'].fillna(0)


# In[16]:


data1


# In[17]:


data = data.drop(['Name','PassengerId'],axis = 1)
data1 = data1.drop(['Name'],axis = 1)


# In[18]:


data


# In[19]:


data1


# In[20]:


data['Sex'] = data['Sex'].map({'female':1 , 'male':0}).astype(int)
data1['Sex'] = data1['Sex'].map({'female':1 , 'male':0}).astype(int)


# In[21]:


data


# In[22]:


data1


# In[23]:


data['Agea'] = pd.cut(data['Age'],5)# no need here just to know what the function does.


# In[24]:


data


# In[25]:


data.loc[data['Age']<=16,'Age'] = 0
data.loc[ data['Age'] <= 16, 'Age'] = 0
data.loc[(data['Age'] > 16) & (data['Age'] <= 32), 'Age'] = 1
data.loc[(data['Age'] > 32) & (data['Age'] <= 48), 'Age'] = 2
data.loc[(data['Age'] > 48) & (data['Age'] <= 64), 'Age'] = 3
data.loc[ data['Age'] > 64, 'Age']


# In[26]:


data


# In[27]:


data = data.drop(['Agea'],axis = 1)


# In[28]:


data


# In[29]:


data['Age'].fillna(data['Age'].dropna().median(),inplace = True)
data


# In[30]:


data1['Age'].fillna(data1['Age'].dropna().median(),inplace = True)
data1


# In[31]:


data['Fare'].fillna(data['Fare'].dropna().median(),inplace = True)
data1['Fare'].fillna(data1['Fare'].dropna().median(),inplace = True)


# In[32]:


data,data1


# In[33]:


frequency = data.Embarked.dropna().mode()[0]
frequency


# In[34]:


data['Embarked'] = data['Embarked'].fillna(frequency)
data


# In[35]:


data1['Embarked'] = data['Embarked'].fillna(frequency)
data1


# In[36]:


data['Embarked'] = data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
data


# In[39]:


data1['Embarked'] = data1['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
data1


# In[38]:


data.isnull().any() # job done 


# In[40]:


data1.isnull().any() # job done 


# In[41]:


from sklearn.tree import DecisionTreeClassifier


# In[47]:


X_train = data.drop("Survived", axis=1)
Y_train = data["Survived"]
X_test  = data1.drop("PassengerId", axis=1).copy()
X_train


# In[44]:


classifier = DecisionTreeClassifier()
classifier.fit(X_train , Y_train)
prediction = classifier.predict(X_test)
prediction


# In[46]:


result = classifier.score(X_train, Y_train)
result


# In[52]:


gender_submission = pd.DataFrame({
        "PassengerId": data1["PassengerId"],
        "Survived": prediction
    })

