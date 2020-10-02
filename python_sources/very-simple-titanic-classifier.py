#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics.classification import accuracy_score, classification_report


# In[6]:


data = pd.read_csv('../input/train.csv')


# In[7]:


data_len = len(data)
for col in data.columns:
    print('{}: {}'.format(col, 100*sum(data[col].isna())/data_len))


# In[8]:


sns.distplot(data['Age'].dropna())


# In[9]:


data['Age'] = data['Age'].fillna(data['Age'].mean())


# In[10]:


data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)


# In[11]:


data.head()


# In[12]:


gender = {
    'female': 0,
    'male': 1
}
data['Sex'] = data['Sex'].map(lambda x: gender.get(x))


# In[13]:


data_len = len(data)
for col in data.columns:
    print('{}: {}'.format(col, 100*sum(data[col].isna())/data_len))


# In[14]:


X_train, X_test, Y_train, Y_test = train_test_split(data.drop('Survived', axis=1), data['Survived'], test_size=0.2)
classifier = LogisticRegression()


# In[15]:


classifier.fit(X_train, Y_train)


# In[16]:


prediction = classifier.predict(X_test)


# In[17]:


'Accuracy: {}'.format(accuracy_score(prediction, Y_test))


# In[18]:


print('Classification report')
print(classification_report(Y_test, prediction))

