#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


data = pd.read_csv("../input/winequality-red.csv")

data.head()


# In[3]:


data.isnull().sum()


# In[4]:


data.corr()


# In[5]:


list(data)


# In[6]:


x = data[['fixed acidity',
 'volatile acidity',
 'citric acid',
 'chlorides',
 'total sulfur dioxide',
 'density',
 'sulphates',
 'alcohol']]

y = data['quality']


# In[7]:


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)


# # Decision Tree

# In[8]:


from sklearn.tree import DecisionTreeClassifier


# In[9]:


clf = DecisionTreeClassifier()
clf.fit(x_train,y_train)


# In[10]:


a = clf.predict(x_test)

from sklearn.metrics import accuracy_score,confusion_matrix

print(accuracy_score(y_test,a))
print(confusion_matrix(y_test,a))


# # Random Forest

# In[11]:


from sklearn.ensemble import RandomForestClassifier


# In[12]:


clf = RandomForestClassifier()
clf.fit(x_train,y_train)
a = clf.predict(x_test)


# In[13]:


print(accuracy_score(y_test,a))
print(confusion_matrix(y_test,a))

