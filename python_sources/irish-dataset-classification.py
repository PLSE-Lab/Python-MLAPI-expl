#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/Iris_Data.csv")
data.head(5)


# In[ ]:


data.tail(5)


# In[17]:


print("Species")
print(data['species'].unique())


# In[ ]:


data.describe()


# In[18]:


#shows the relation between individual features with all the species
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.violinplot(x='species',y='sepal_length',data=data)
plt.subplot(2,2,2)
sns.violinplot(x='species',y='sepal_width',data=data)
plt.subplot(2,2,3)
sns.violinplot(x='species',y='petal_length',data=data)
plt.subplot(2,2,4)
sns.violinplot(x='species',y='petal_width',data=data)
plt.show()


# In[35]:


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from keras.utils import to_categorical
from sklearn import svm
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier


# In[20]:


plt.figure(figsize=(7,4)) 
sns.heatmap(data.corr(),annot=True,cmap='cubehelix_r') #draws  heatmap with input as the correlation matrix calculted by(iris.corr())
plt.show()


# In[21]:


train,test= train_test_split(data,test_size=0.3) #splitting the train_test data i.e test data by 30% of total data
print(train.shape) #printing the shape of the training data
print(test.shape)  #printing the shape of the test data


# In[25]:


#Assigning the train test data in X, y format of training and testing data
X_train=train[['sepal_length','sepal_width','petal_length','petal_width']]
X_test=test[['sepal_length','sepal_width','petal_length','petal_width']]
y_train=train.species
y_test=test.species


# In[26]:


#Implementing the trainning and testing datas in various models
#1. Support Vector Machine Algorithm
model=svm.SVC()
model.fit(X_train,y_train)
y_predict=model.predict(X_test)
accu=metrics.accuracy_score(y_predict,y_test)
print("The accuracy of SVM is:",accu)


# In[27]:


#2. Logistic Regression
model=LogisticRegression()
model.fit(X_train,y_train)
y_predict=model.predict(X_test)
accu=metrics.accuracy_score(y_predict,y_test)
print("The accuracy of Logistic Regression is:",accu)


# In[28]:


#3. KNeighborsClassifier using k=3
model= KNeighborsClassifier(n_neighbors=3)
model.fit(X_train,y_train)
y_predict=model.predict(X_test)
accu=metrics.accuracy_score(y_predict,y_test)
print("The accuracy of KNeighborsClassifier with 3 neighbors is:",accu)


# In[30]:


#3. cont. KNeighborsClassifier using various values of k
a_index=list(range(1,11))
a=pd.Series()
x=[1,2,3,4,5,6,7,8,9,10]
for i in list(range(1,11)):
    model=KNeighborsClassifier(n_neighbors=i) 
    model.fit(X_train,y_train)
    y_predict=model.predict(X_test)
    a=a.append(pd.Series(metrics.accuracy_score(y_predict,y_test)))
plt.plot(a_index, a)
plt.xticks(x)


# In[34]:


#4 Decision Tree Classifier

model= DecisionTreeClassifier()
model.fit(X_train,y_train)
y_predict=model.predict(X_test)
accu=metrics.accuracy_score(y_predict,y_test)
print("The accuracy of DecisionTreeClassifier is:",accu)

