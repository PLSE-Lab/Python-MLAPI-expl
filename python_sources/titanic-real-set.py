#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
print(os.listdir("../input"))
import numpy as np
import pandas as pd 
import math as ma 
import matplotlib.pyplot as plt
import seaborn as sn
import sklearn as sk


# In[2]:


testset = pd.read_csv("../input/titanic3.csv")


# In[3]:


testset.info()


# In[4]:


sn.countplot(x ="survived", data = testset)


# In[5]:


sn.countplot(x = "survived", hue = "pclass", data = testset )


# In[6]:


sn.countplot(x = 'survived', hue = "sibsp", data = testset)


# In[7]:


sn.scatterplot(x = 'fare', y = 'survived', data = testset)


# In[8]:


testset.drop("cabin", axis=1, inplace=True)


# In[9]:


testset.head(2)


# In[10]:


testset.isnull().sum()


# In[11]:


sn.heatmap(testset.isnull(), yticklabels=False, cbar=True)


# In[13]:


sex = pd.get_dummies(testset['sex'], drop_first=True)


# In[14]:


embarked = pd.get_dummies(testset["embarked"], drop_first=True)
pcl = pd.get_dummies(testset["pclass"], drop_first=True)


# In[34]:


sibsp = pd.get_dummies(testset['sibsp'])


# In[35]:


titanic_data = pd.concat([testset, sex, embarked, pcl, sibsp], axis=1)


# In[36]:


titanic_data.head(20)


# In[37]:


titanic_data.drop(["sex", "embarked", "parch", "name", "ticket", 'boat', 'home.dest',], axis=1,inplace=True)


# In[38]:


titanic_data.drop(['body'], axis =1, inplace=True)


# In[39]:


titanic_data.head(2)


# In[40]:


titanic_data.drop(['fare', 'age'], axis =1, inplace=True)


# In[41]:


sn.heatmap(titanic_data.isnull(), yticklabels=False, cbar=True)


# In[43]:


#Training Data
X = titanic_data.drop("survived", axis=1)
y = titanic_data["survived"]


# In[44]:


from sklearn.model_selection import train_test_split


# In[54]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)


# In[55]:


from sklearn.linear_model import LogisticRegression


# In[56]:


logmodel = LogisticRegression()


# In[57]:


logmodel.fit(X_train, y_train)


# In[58]:


predictions = logmodel.predict(X_test)


# In[59]:


from sklearn.metrics import classification_report


# In[60]:


classification_report(y_test,predictions)


# In[61]:


from sklearn.metrics import confusion_matrix


# In[62]:


confusion_matrix(y_test, predictions)


# In[63]:


from sklearn.metrics import accuracy_score


# In[64]:


accuracy_score(y_test, predictions)


# In[65]:


##Testing accuracy
titanic_data2 = pd.concat([testset, sex], axis=1)


# In[66]:


titanic_data2.head(2)


# In[69]:


titanic_data2.drop(["sex", "embarked", "parch", "name", "ticket", 'boat', 'home.dest'], axis=1,inplace=True)


# In[70]:


titanic_data2.drop(['age', 'fare', 'body', 'sibsp'], axis = 1, inplace=True )


# In[71]:


titanic_data2.head(2)


# In[72]:


##Train Data 2
X = titanic_data2.drop("survived", axis=1)
y = titanic_data2["survived"]


# In[73]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)


# In[74]:


logmodel.fit(X_train, y_train)


# In[75]:


predictions = logmodel.predict(X_test)


# In[76]:


classification_report(y_test,predictions)


# In[77]:


confusion_matrix(y_test, predictions)


# In[78]:


accuracy_score(y_test, predictions)


# In[ ]:




