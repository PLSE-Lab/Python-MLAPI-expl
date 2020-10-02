#!/usr/bin/env python
# coding: utf-8

# 

# In[14]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[15]:


df = pd.read_csv("../input/data.csv")


# In[16]:


df.head(3)


# In[17]:


X=df[df.columns[2:32]]
y=df[df.columns[1]]


# In[18]:


X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=0, stratify=y)


# In[19]:


X_train.shape


# In[20]:


X_test.shape


# #K Nearest Neighbor

# In[21]:


knr = KNeighborsClassifier(n_neighbors=6).fit(X_train, y_train)
print("train score - " + str(knr.score(X_train, y_train)))
print("test score - " + str(knr.score(X_test, y_test)))


# #Logistic Regression

# In[22]:


lr1 = LogisticRegression(random_state=0).fit(X_train, y_train)
print("train score - " + str(lr1.score(X_train, y_train)))
print("test score - " + str(lr1.score(X_test, y_test)))


# ###After Tuning
# ####Increased Regularization Parameter (C) value

# In[23]:


lr2 = LogisticRegression(C=6, random_state=0).fit(X_train, y_train)
print("train score - " + str(lr2.score(X_train, y_train)))
print("test score - " + str(lr2.score(X_test, y_test)))


# #Linear SVC

# In[24]:


svc = LinearSVC(random_state=0).fit(X_train,y_train)
print("train score - " + str(svc.score(X_train, y_train)))
print("test score - " + str(svc.score(X_test, y_test)))


# ###After Tuning
# ####Increased Regularization Parameter (C) value

# In[25]:


svc = LinearSVC(C=3, random_state=0).fit(X_train,y_train)
print("train score - " + str(svc.score(X_train, y_train)))
print("test score - " + str(svc.score(X_test, y_test)))


# SVC Models are good when the data is scaled. Lets scale the data and build the model

# In[26]:


scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svc_scaled = LinearSVC(C=2, random_state=0).fit(X_train_scaled,y_train)
print("train score - " + str(svc_scaled.score(X_train_scaled, y_train)))
print("test score - " + str(svc_scaled.score(X_test_scaled, y_test)))


# #Decision Tree

# In[27]:


dec = DecisionTreeClassifier(random_state=0).fit(X_train, y_train)
print("train score - " + str(dec.score(X_train, y_train)))
print("test score - " + str(dec.score(X_test, y_test)))


# ###After Tuning
# ####Pre-pruning by reducing maximum depth to 5

# In[28]:


dec = DecisionTreeClassifier(max_depth=5, random_state=0).fit(X_train, y_train)
print("train score - " + str(dec.score(X_train, y_train)))
print("test score - " + str(dec.score(X_test, y_test)))


# #Random Forest

# In[29]:


forest = RandomForestClassifier(n_estimators=10, random_state=0).fit(X_train, y_train)
print("train score - " + str(forest.score(X_train, y_train)))
print("test score - " + str(forest.score(X_test, y_test)))


# ###After Tuning
# ####Increased n_estimators to 100 and max_features to 30. Max depth restricted to 5

# In[30]:


forest = RandomForestClassifier(n_estimators=100, max_features=30, max_depth=5, random_state=0).fit(X_train, y_train)
print("train score - " + str(forest.score(X_train, y_train)))
print("test score - " + str(forest.score(X_test, y_test)))


# #Gradient Boosting

# In[31]:


gb = GradientBoostingClassifier().fit(X_train, y_train)
print("train score - " + str(gb.score(X_train, y_train)))
print("test score - " + str(gb.score(X_test, y_test)))


# ###After Tuning
# ####Increased learning rate to 0.15 from default 0.1

# In[32]:


gb = GradientBoostingClassifier(random_state=0, learning_rate=0.15).fit(X_train, y_train)
print("train score - " + str(gb.score(X_train, y_train)))
print("test score - " + str(gb.score(X_test, y_test)))


# #Conclusion

# We can see Support Vector Machine (SVM) model with scaled data is giving a very good score on test data.

# In[34]:


print("Support Vector Machine test score - " + str(svc_scaled.score(X_test_scaled, y_test)))
print("Gradient Boosting test score - " + str(gb.score(X_test, y_test)))
print("Logistic Regression test score - " + str(lr2.score(X_test, y_test)))


# In[ ]:




