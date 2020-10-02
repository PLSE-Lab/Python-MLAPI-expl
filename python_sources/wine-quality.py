#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data = pd.read_csv("../input/winequality-red.csv")
data.head(10)


# In[3]:


data.info()


# In[4]:


data.describe()


# In[5]:


data['quality'].value_counts()


# In[6]:


data['quality'].value_counts().plot.bar()


# In[9]:


data.corr()


# In[11]:


data['quality'] = data['quality'].astype(int)

data['quality'].value_counts()


# In[12]:


y = data['quality']
X = data.drop('quality',axis=1)

from sklearn.model_selection import train_test_split , cross_val_score, GridSearchCV


from sklearn.ensemble import RandomForestClassifier

train_x,test_x,train_y,test_y = train_test_split(X,y)


# In[13]:


forest = RandomForestClassifier(n_estimators=400,random_state = 42)
forest.fit(train_x,train_y)
predicts = forest.predict(test_x)


# In[14]:


confusionMatrix = confusion_matrix(test_y,predicts)
confusionMatrix

