#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


#importing the datset


# In[ ]:


dataset = pd.read_csv('../input/Iris.csv')


# In[ ]:


dataset.head()


# In[ ]:


# ID is unrelevat collumn , so i want to drop it .
dataset =dataset.drop("Id",axis=1)


# In[ ]:


dataset.head()


# In[ ]:


#finding is there any null value in data set or not
dataset.isnull().values.any()


# In[ ]:


X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# In[ ]:


print(X)


# In[ ]:


print(y)


# In[ ]:


#for y converting categorical data , for the i use get_dummies method , you can use lable encoding an one hot encoding 
y=pd.get_dummies(y,columns=['Species'])


# In[ ]:


print (y)


# In[ ]:


# droping one dummy column 
y = y.iloc[:, 1:3].values


# In[ ]:


print (y)


# In[ ]:


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[ ]:


print (X_test)
print(y_test)
print(X_train)
print(y_train)


# In[ ]:


X.shape


# In[ ]:


X_train.shape


# In[ ]:


X_test.shape


# In[ ]:


y.shape


# In[ ]:


y_test.shape


# In[ ]:


y_train.shape


# In[ ]:


#i am using k-nearest Neighbours over here.
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=8)
classifier.fit(X_train, y_train)


# In[ ]:


y_pred = classifier.predict(X_test)


# In[ ]:


print (y_pred)


# In[ ]:


print(y_test)


# In[ ]:


# Summary of the predictions made by the classifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
print(classification_report(y_test, y_pred))


# In[ ]:


print(confusion_matrix(y_test.argmax(axis=1),y_pred.argmax(axis=1)))


# In[ ]:


# Accuracy score
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(y_pred,y_test))


# In[ ]:




